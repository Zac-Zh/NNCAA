import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from models.cifar_cnn import CIFARCNN, train_cifar
from models.hn_classifier import HNClassifier
from models.ln_generator import LNGenerator
from utils.cifar_loader import load_cifar10_data, get_class_names
from utils.credibility import compute_neuron_credibility, split_neurons_by_credibility
from utils.visualization import visualize_adversarial_samples, visualize_feature_space, plot_decision_boundary_distance

def parse_args():
    parser = argparse.ArgumentParser(description='神经网络可信度评估与对抗训练 (CIFAR-10)')
    parser.add_argument('--batch-size', type=int, default=128, help='批量大小')
    parser.add_argument('--epochs', type=int, default=1, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='训练设备')
    parser.add_argument('--threshold-percentile', type=float, default=70, help='神经元可信度划分阈值（百分位数）')
    parser.add_argument('--max-perturbation', type=float, default=0.05, help='最大扰动范围')
    parser.add_argument('--visualize', action='store_true', help='是否可视化对抗样本')
    parser.add_argument('--save-dir', type=str, default='./results', help='结果保存目录')
    return parser.parse_args()

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total

def calculate_decision_boundary_distance(model, inputs, labels, device):
    """计算样本到决策边界的距离
    
    这里使用一个简化的方法：对于每个样本，计算其预测类别的概率与第二高概率类别之间的差距
    这个差距可以近似表示样本到决策边界的距离
    """
    model.eval()
    inputs, labels = inputs.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
        probabilities = F.softmax(outputs, dim=1)
        
        # 获取每个样本的预测类别和概率
        values, predictions = probabilities.max(1)
        
        # 创建掩码，将最大概率设为0
        mask = torch.zeros_like(probabilities).scatter_(1, predictions.unsqueeze(1), 1)
        masked_probs = probabilities * (1 - mask) + mask * (-1)
        
        # 获取第二高的概率
        second_values, _ = masked_probs.max(1)
        
        # 计算最高概率与第二高概率的差距作为到决策边界的距离
        distances = values - second_values
        
    return distances, predictions

def train_adversarial(hn_classifier, ln_generator, train_loader, test_loader, epochs, lr, device, 
                     visualize=False, save_dir='./results', class_names=None):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 优化器
    optimizer_c = torch.optim.Adam(hn_classifier.parameters(), lr=lr)
    optimizer_g = torch.optim.Adam(ln_generator.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # 记录训练过程中的指标
    metrics = {
        'c_loss': [],
        'g_loss': [],
        'test_acc': [],
        'adv_acc': []
    }
    
    for epoch in range(epochs):
        hn_classifier.train()
        ln_generator.train()
        total_c_loss = 0
        total_g_loss = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            batch_size = inputs.size(0)
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 训练生成器
            optimizer_g.zero_grad()
            adversarial_inputs, perturbations = ln_generator(inputs)
            outputs = hn_classifier(adversarial_inputs)
            g_loss = -criterion(outputs, labels)  # 最大化分类误差
            g_loss.backward()
            optimizer_g.step()
            total_g_loss += g_loss.item()
            
            # 训练分类器
            optimizer_c.zero_grad()
            outputs = hn_classifier(inputs)  # 在原始样本上训练
            c_loss = criterion(outputs, labels)
            c_loss.backward()
            optimizer_c.step()
            total_c_loss += c_loss.item()
        
        # 评估
        hn_classifier.eval()
        ln_generator.eval()
        
        # 在测试集上评估
        test_acc = evaluate_model(hn_classifier, test_loader, device)
        
        # 生成对抗样本并评估
        adv_correct = 0
        adv_total = 0
        all_inputs = []
        all_perturbations = []
        all_adv_inputs = []
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                adv_inputs, perturbations = ln_generator(inputs)
                outputs = hn_classifier(adv_inputs)
                _, predicted = outputs.max(1)
                
                adv_total += labels.size(0)
                adv_correct += predicted.eq(labels).sum().item()
                
                # 收集样本用于可视化
                if visualize and len(all_inputs) < 100:  # 只收集前100个样本
                    all_inputs.append(inputs.cpu())
                    all_perturbations.append(perturbations.cpu())
                    all_adv_inputs.append(adv_inputs.cpu())
                    all_labels.append(labels.cpu())
                    all_predictions.append(predicted.cpu())
        
        adv_acc = adv_correct / adv_total
        
        # 记录指标
        metrics['c_loss'].append(total_c_loss/len(train_loader))
        metrics['g_loss'].append(total_g_loss/len(train_loader))
        metrics['test_acc'].append(test_acc)
        metrics['adv_acc'].append(adv_acc)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'分类器损失: {total_c_loss/len(train_loader):.4f}')
        print(f'生成器损失: {total_g_loss/len(train_loader):.4f}')
        print(f'测试准确率: {test_acc:.4f}')
        print(f'对抗准确率: {adv_acc:.4f}')
        print(f'攻击成功率: {1-adv_acc:.4f}')
        
        # 可视化
        if visualize and (epoch + 1) % 5 == 0:  # 每5个epoch可视化一次
            # 合并收集的样本
            all_inputs = torch.cat(all_inputs, dim=0)
            all_perturbations = torch.cat(all_perturbations, dim=0)
            all_adv_inputs = torch.cat(all_adv_inputs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            all_predictions = torch.cat(all_predictions, dim=0)
            
            # 可视化对抗样本
            visualize_adversarial_samples(
                all_inputs, all_perturbations, all_adv_inputs,
                all_labels, all_predictions, class_names,
                num_samples=5, save_path=f'{save_dir}/adv_samples_epoch_{epoch+1}.png'
            )
            
            # 计算样本到决策边界的距离
            distances, predictions = calculate_decision_boundary_distance(
                hn_classifier, all_adv_inputs[:100].to(device), all_labels[:100].to(device), device
            )
            
            # 可视化决策边界距离
            plot_decision_boundary_distance(
                distances, all_labels[:100], predictions,
                class_names, save_path=f'{save_dir}/decision_boundary_epoch_{epoch+1}.png'
            )
            
            # 提取特征空间
            features = []
            with torch.no_grad():
                for i in range(0, len(all_adv_inputs), 100):
                    batch = all_adv_inputs[i:i+100].to(device)
                    # 使用倒数第二层作为特征
                    hn_classifier(batch)
                    batch_features = hn_classifier.activations['fc2']
                    features.append(batch_features.cpu())
            
            features = torch.cat(features, dim=0)
            
            # 可视化特征空间
            visualize_feature_space(
                features, all_labels[:len(features)],
                class_names, save_path=f'{save_dir}/feature_space_epoch_{epoch+1}.png'
            )
    
    # 绘制训练过程中的指标变化
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(metrics['c_loss'])
    plt.title('分类器损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(2, 2, 2)
    plt.plot(metrics['g_loss'])
    plt.title('生成器损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(2, 2, 3)
    plt.plot(metrics['test_acc'])
    plt.title('测试准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.subplot(2, 2, 4)
    plt.plot(metrics['adv_acc'])
    plt.title('对抗准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_metrics.png')
    
    return metrics

def main():
    args = parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载数据
    print('加载CIFAR-10数据集...')
    train_loader, test_loader = load_cifar10_data(args.batch_size)
    class_names = get_class_names()
    
    # 训练基准模型
    print('训练基准CNN模型...')
    baseline_model = CIFARCNN()
    baseline_model = train_cifar(
        baseline_model,
        train_loader,
        test_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device
    )
    
    # 计算神经元可信度
    print('计算神经元可信度...')
    credibility_scores = compute_neuron_credibility(
        baseline_model,
        train_loader,
        device=args.device
    )
    
    # 划分高低可信度神经元
    print('划分高低可信度神经元...')
    high_credibility_neurons, low_credibility_neurons = split_neurons_by_credibility(
        baseline_model,
        credibility_scores,
        threshold_percentile=args.threshold_percentile
    )
    
    # 构建HNs分类器和LNs生成器
    print('构建HNs分类器和LNs生成器...')
    hn_classifier = HNClassifier(baseline_model, high_credibility_neurons).to(args.device)
    ln_generator = LNGenerator(
        baseline_model,
        low_credibility_neurons,
        max_perturbation=args.max_perturbation
    ).to(args.device)
    
    # 对抗训练
    print('开始对抗训练...')
    metrics = train_adversarial(
        hn_classifier,
        ln_generator,
        train_loader,
        test_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        visualize=args.visualize,
        save_dir=args.save_dir,
        class_names=class_names
    )
    
    # 保存模型
    torch.save(hn_classifier.state_dict(), f'{args.save_dir}/hn_classifier_cifar.pth')
    torch.save(ln_generator.state_dict(), f'{args.save_dir}/ln_generator_cifar.pth')
    print('模型已保存')

if __name__ == '__main__':
    main()