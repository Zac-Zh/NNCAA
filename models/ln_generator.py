import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cifar_cnn import CIFARCNN

class LNGenerator(nn.Module):
    """低可信度神经元对抗生成器
    
    这个模型使用从基准CNN中提取的低可信度神经元构建对抗生成器。
    它的目标是生成能够最大化高可信度神经元分类误差的样本。
    """
    
    def __init__(self, baseline_model, low_credibility_neurons, max_perturbation=0.1):
        """
        Args:
            baseline_model: 预训练的基准CNN模型
            low_credibility_neurons: 包含每层低可信度神经元索引的字典
            max_perturbation: 最大扰动范围
        """
        super(LNGenerator, self).__init__()
        
        # 复制CIFAR模型的结构
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        
        # 添加一个额外的全连接层，用于生成扰动
        self.perturbation_size = 3 * 32 * 32  # CIFAR-10图像大小
        self.fc_perturbation = nn.Linear(256, 3 * 32 * 32)  # 确保输出维度匹配图像大小
        
        # 添加维度验证断言
        assert self.fc_perturbation.out_features == 3*32*32, \
            f"全连接层输出维度应为{3*32*32}，实际为{self.fc_perturbation.out_features}"
        self.bn = nn.BatchNorm1d(3 * 32 * 32)  # 输入维度匹配扰动层输出
        
        # 加载预训练模型的参数（除了新添加的层）
        pretrained_dict = baseline_model.state_dict()
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        
        # 保存低可信度神经元的索引
        self.low_credibility_neurons = low_credibility_neurons
        
        # 最大扰动范围
        self.max_perturbation = max_perturbation
        
        # 用于记录每层激活值的字典
        self.activations = {}
        
    def forward(self, x):
        # 保存原始输入
        original_x = x.clone()
        
        # 第一个卷积层
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # 只保留低可信度神经元的激活值
        if 'conv1' in self.low_credibility_neurons:
            mask = torch.zeros(x.shape[1], device=x.device)
            mask[self.low_credibility_neurons['conv1']] = 1
            x = x * mask.view(1, -1, 1, 1)
        
        self.activations['conv1'] = x.clone().detach()
        
        # 第二个卷积层
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # 只保留低可信度神经元的激活值
        if 'conv2' in self.low_credibility_neurons:
            mask = torch.zeros(x.shape[1], device=x.device)
            mask[self.low_credibility_neurons['conv2']] = 1
            x = x * mask.view(1, -1, 1, 1)
        
        self.activations['conv2'] = x.clone().detach()
        
        # 第二个卷积块
        x = F.relu(self.conv3(x))
        if 'conv3' in self.low_credibility_neurons:
            mask = torch.zeros(x.shape[1], device=x.device)
            mask[self.low_credibility_neurons['conv3']] = 1
            x = x * mask.view(1, -1, 1, 1)
        
        self.activations['conv3'] = x.clone().detach()
        
        x = F.relu(self.conv4(x))
        if 'conv4' in self.low_credibility_neurons:
            mask = torch.zeros(x.shape[1], device=x.device)
            mask[self.low_credibility_neurons['conv4']] = 1
            x = x * mask.view(1, -1, 1, 1)
        
        self.activations['conv4'] = x.clone().detach()
        x = self.pool(x)
        
        # 将特征图展平为一维向量
        x = x.view(-1, 128 * 8 * 8)
        
        # 第一个全连接层
        x = self.fc1(x)
        x = F.relu(x)
        
        # 只保留低可信度神经元的激活值
        if 'fc1' in self.low_credibility_neurons:
            mask = torch.zeros(x.shape[1], device=x.device)
            mask[self.low_credibility_neurons['fc1']] = 1
            x = x * mask.view(1, -1)
        
        self.activations['fc1'] = x.clone().detach()
        
        # 第二个全连接层
        x = self.fc2(x)
        x = F.relu(x)
        
        # 只保留低可信度神经元的激活值
        if 'fc2' in self.low_credibility_neurons:
            mask = torch.zeros(x.shape[1], device=x.device)
            mask[self.low_credibility_neurons['fc2']] = 1
            x = x * mask.view(1, -1)
        
        self.activations['fc2'] = x.clone().detach()
        
        # 生成扰动
        assert x.size(1) == 256, f"FC2层输出维度应为256，实际得到{x.size(1)}"
        perturbation = self.fc_perturbation(x)
        
        # 维度验证
        assert perturbation.size(1) == 3*32*32, f"预期3072维，实际得到{perturbation.size(1)}维"
        
        # 重塑扰动为图像形状
        batch_size = x.size(0)
        perturbation = perturbation.view(batch_size, 3, 32, 32)
        
        # 应用激活函数和缩放
        perturbation = torch.tanh(perturbation) * self.max_perturbation
        
        # 最终形状验证
        assert perturbation.shape == (batch_size, 3, 32, 32), \
            f"最终形状错误: 预期({batch_size},3,32,32) 实际{perturbation.shape}"
        
        # 将扰动应用到原始输入上
        adversarial_x = original_x + perturbation
        
        # 最终输出验证
        assert adversarial_x.shape == original_x.shape, \
            f"对抗样本形状错误: 预期{original_x.shape} 实际{adversarial_x.shape}"
        
        # 确保生成的样本在有效范围内
        adversarial_x = torch.clamp(adversarial_x, 0, 1)
        
        return adversarial_x, perturbation
    
    def get_activations(self):
        """获取模型各层的激活值"""
        return self.activations

# 用于训练低可信度神经元对抗生成器的函数
def train_ln_generator(generator, classifier, train_loader, epochs=10, lr=0.001, device='cuda'):
    """训练低可信度神经元对抗生成器
    
    Args:
        generator: 低可信度神经元对抗生成器
        classifier: 高可信度神经元分类器
        train_loader: 训练数据加载器
        epochs: 训练轮数
        lr: 学习率
        device: 训练设备（'cuda'或'cpu'）
    
    Returns:
        训练好的生成器
    """
    # 将模型移动到指定设备
    generator = generator.to(device)
    classifier = classifier.to(device)
    
    # 固定分类器的参数
    for param in classifier.parameters():
        param.requires_grad = False
    
    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 使用Adam优化器
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    
    # 训练循环
    for epoch in range(epochs):
        generator.train()
        classifier.eval()
        
        running_loss = 0.0
        attack_success = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 生成对抗样本
            adversarial_inputs, _ = generator(inputs)
            
            # 使用分类器对对抗样本进行分类
            outputs = classifier(adversarial_inputs)
            
            # 计算损失（目标是最大化分类误差）
            # 这里我们使用负的交叉熵损失，因为我们想要最大化误分类
            loss = -criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 统计损失和攻击成功率
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            attack_success += (predicted != labels).sum().item()
        
        # 计算攻击成功率
        attack_success_rate = 100. * attack_success / total
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, '
              f'Attack Success Rate: {attack_success_rate:.2f}%')
    
    return generator

# 用于对抗训练的函数
def adversarial_training(classifier, generator, train_loader, test_loader, epochs=10, lr=0.001, lambda_adv=0.5, device='cuda'):
    """对抗训练函数
    
    Args:
        classifier: 高可信度神经元分类器
        generator: 低可信度神经元对抗生成器
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        epochs: 训练轮数
        lr: 学习率
        lambda_adv: 对抗损失的权重
        device: 训练设备（'cuda'或'cpu'）
    
    Returns:
        训练好的分类器
    """
    # 将模型移动到指定设备
    classifier = classifier.to(device)
    generator = generator.to(device)
    
    # 固定生成器的参数
    for param in generator.parameters():
        param.requires_grad = False
    
    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 使用Adam优化器
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    
    # 训练循环
    for epoch in range(epochs):
        classifier.train()
        generator.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 生成对抗样本
            adversarial_inputs, _ = generator(inputs)
            
            # 在原始样本上的前向传播
            outputs_original = classifier(inputs)
            
            # 在对抗样本上的前向传播
            outputs_adversarial = classifier(adversarial_inputs)
            
            # 计算原始损失和对抗损失
            loss_original = criterion(outputs_original, labels)
            loss_adversarial = criterion(outputs_adversarial, labels)
            
            # 总损失 = 原始损失 + lambda * 对抗损失
            loss = loss_original + lambda_adv * loss_adversarial
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 统计损失和准确率（在原始样本上）
            running_loss += loss.item()
            _, predicted = outputs_original.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # 计算训练集上的准确率
        train_accuracy = 100. * correct / total
        
        # 在测试集上评估模型
        classifier.eval()
        test_correct = 0
        test_total = 0
        adv_correct = 0
        adv_total = 0
        
        with torch.no_grad():
            # 在原始测试集上评估
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = classifier(inputs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
                
                # 在对抗样本上评估
                adversarial_inputs, _ = generator(inputs)
                outputs = classifier(adversarial_inputs)
                _, predicted = outputs.max(1)
                adv_total += labels.size(0)
                adv_correct += predicted.eq(labels).sum().item()
        
        # 计算测试集上的准确率
        test_accuracy = 100. * test_correct / test_total
        adv_accuracy = 100. * adv_correct / adv_total
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%, '
              f'Adv Acc: {adv_accuracy:.2f}%')
    
    return classifier