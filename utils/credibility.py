import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
def compute_neuron_credibility(model, data_loader, device='cuda', num_samples=1000):
    """计算神经元可信度
    
    使用因果推断方法计算每个神经元的可信度，通过逐个屏蔽神经元并观察对模型输出的影响。
    
    Args:
        model: 预训练的基准CNN模型
        data_loader: 数据加载器
        device: 计算设备（'cuda'或'cpu'）
        num_samples: 用于计算可信度的样本数量
    
    Returns:
        credibility_scores: 包含每层神经元可信度分数的字典
    """
    model = model.to(device)
    model.eval()
    
    # 用于存储每层神经元的可信度分数
    credibility_scores = {}
    
    # 获取有限数量的样本用于计算可信度
    sample_inputs = []
    sample_labels = []
    sample_count = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            batch_size = inputs.size(0)
            if sample_count + batch_size > num_samples:
                # 只取需要的样本数量
                inputs = inputs[:num_samples - sample_count]
                labels = labels[:num_samples - sample_count]
            
            sample_inputs.append(inputs.to(device))
            sample_labels.append(labels.to(device))
            
            sample_count += inputs.size(0)
            if sample_count >= num_samples:
                break
    
    # 合并所有样本
    inputs = torch.cat(sample_inputs, dim=0)
    labels = torch.cat(sample_labels, dim=0)
    
    # 获取原始预测结果
    with torch.no_grad():
        original_outputs = model(inputs)
        _, original_preds = original_outputs.max(1)
        original_correct = original_preds.eq(labels).float()
    
    # 计算每层神经元的可信度
    for layer_name in ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2']:
        print(f'计算{layer_name}层神经元可信度...')
        
        # 获取该层的激活值
        _ = model(inputs)  # 前向传播以获取激活值
        activations = model.activations[layer_name]
        
        # 获取该层神经元数量
        if layer_name.startswith('conv'):
            # 卷积层的神经元对应于通道数
            num_neurons = activations.size(1)
            credibility_scores[layer_name] = torch.zeros(num_neurons, device=device)
            
            # 对每个通道（神经元）进行屏蔽
            for neuron_idx in tqdm(range(num_neurons)):
                # 复制激活值
                masked_activations = activations.clone()
                
                # 屏蔽当前神经元（将该通道的激活值设为0）
                masked_activations[:, neuron_idx] = 0
                
                # 计算屏蔽后的输出
                if layer_name == 'conv1':
                    # 从conv1层开始重新计算
                    x = masked_activations
                    x = F.relu(model.conv2(x))
                    x = model.pool(x)
                    x = F.relu(model.conv3(x))
                    x = F.relu(model.conv4(x))
                    x = model.pool(x)
                    x = x.view(-1, 128 * 8 * 8)
                    x = F.relu(model.fc1(x))
                    x = model.dropout(x)
                    x = F.relu(model.fc2(x))
                    x = model.dropout(x)
                    masked_outputs = model.fc3(x)
                elif layer_name == 'conv2':
                    # 从conv2层开始重新计算
                    x = masked_activations
                    x = model.pool(x)
                    x = F.relu(model.conv3(x))
                    x = F.relu(model.conv4(x))
                    x = model.pool(x)
                    x = x.view(-1, 128 * 8 * 8)
                    x = F.relu(model.fc1(x))
                    x = model.dropout(x)
                    x = F.relu(model.fc2(x))
                    x = model.dropout(x)
                    masked_outputs = model.fc3(x)
                elif layer_name == 'conv3':
                    # 从conv3层开始重新计算
                    x = masked_activations
                    x = F.relu(model.conv4(x))
                    x = model.pool(x)
                    x = x.view(-1, 128 * 8 * 8)
                    x = F.relu(model.fc1(x))
                    x = model.dropout(x)
                    x = F.relu(model.fc2(x))
                    x = model.dropout(x)
                    masked_outputs = model.fc3(x)
                elif layer_name == 'conv4':
                    # 从conv4层开始重新计算
                    x = masked_activations
                    x = model.pool(x)
                    x = x.view(-1, 128 * 8 * 8)
                    x = F.relu(model.fc1(x))
                    x = model.dropout(x)
                    x = F.relu(model.fc2(x))
                    x = model.dropout(x)
                    masked_outputs = model.fc3(x)
                
                # 计算屏蔽后的预测结果
                _, masked_preds = masked_outputs.max(1)
                masked_correct = masked_preds.eq(labels).float()
                
                # 计算可信度分数：屏蔽前后正确预测的变化比例
                # 可信度越高，表示该神经元对正确预测的贡献越大
                impact = (original_correct - masked_correct).mean().item()
                credibility_scores[layer_name][neuron_idx] = impact
        
        else:  # 全连接层
            num_neurons = activations.size(1)
            credibility_scores[layer_name] = torch.zeros(num_neurons, device=device)
            
            # 对每个神经元进行屏蔽
            for neuron_idx in tqdm(range(num_neurons)):
                # 复制激活值
                masked_activations = activations.clone()
                
                # 屏蔽当前神经元
                masked_activations[:, neuron_idx] = 0
                
                # 计算屏蔽后的输出
                if layer_name == 'fc1':
                    # 从fc1层开始重新计算
                    x = masked_activations
                    x = F.relu(model.fc2(x))
                    masked_outputs = model.fc3(x)
                elif layer_name == 'fc2':
                    # 从fc2层开始重新计算
                    x = masked_activations
                    masked_outputs = model.fc3(x)
                
                # 计算屏蔽后的预测结果
                _, masked_preds = masked_outputs.max(1)
                masked_correct = masked_preds.eq(labels).float()
                
                # 计算可信度分数
                impact = (original_correct - masked_correct).mean().item()
                credibility_scores[layer_name][neuron_idx] = impact
    
    return credibility_scores

def split_neurons_by_credibility(model, credibility_scores, threshold_percentile=70):
    """根据可信度分数将神经元划分为高可信度和低可信度两组
    
    Args:
        model: 预训练的基准CNN模型
        credibility_scores: 包含每层神经元可信度分数的字典
        threshold_percentile: 划分高低可信度神经元的百分位阈值
    
    Returns:
        high_credibility_neurons: 包含每层高可信度神经元索引的字典
        low_credibility_neurons: 包含每层低可信度神经元索引的字典
    """
    high_credibility_neurons = {}
    low_credibility_neurons = {}
    
    for layer_name, scores in credibility_scores.items():
        # 将分数转换为NumPy数组以便计算百分位数
        scores_np = scores.cpu().numpy()
        
        # 计算阈值（使用百分位数）
        threshold = np.percentile(scores_np, threshold_percentile)
        
        # 划分高低可信度神经元
        high_idx = torch.where(scores >= threshold)[0].cpu().numpy()
        low_idx = torch.where(scores < threshold)[0].cpu().numpy()
        
        high_credibility_neurons[layer_name] = high_idx
        low_credibility_neurons[layer_name] = low_idx
        
        print(f'{layer_name}层: 高可信度神经元 {len(high_idx)}个, 低可信度神经元 {len(low_idx)}个')
    
    return high_credibility_neurons, low_credibility_neurons

def visualize_credibility_distribution(credibility_scores):
    """可视化每层神经元的可信度分布
    
    Args:
        credibility_scores: 包含每层神经元可信度分数的字典
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(15, 10))
    
    for i, (layer_name, scores) in enumerate(credibility_scores.items(), 1):
        plt.subplot(2, 2, i)
        
        # 将分数转换为NumPy数组
        scores_np = scores.cpu().numpy()
        
        # 绘制分布图
        sns.histplot(scores_np, kde=True)
        plt.title(f'{layer_name}层神经元可信度分布')
        plt.xlabel('可信度分数')
        plt.ylabel('神经元数量')
        
        # 添加垂直线表示70%分位点
        threshold = np.percentile(scores_np, 70)
        plt.axvline(x=threshold, color='r', linestyle='--', 
                   label=f'70%分位点: {threshold:.4f}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('credibility_distribution.png')
    plt.close()
    
    print('可信度分布图已保存为credibility_distribution.png')