import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cifar_cnn import CIFARCNN

class HNClassifier(nn.Module):
    def __init__(self, baseline_model, high_credibility_neurons):
        super(HNClassifier, self).__init__()
        
        # 复制CIFAR模型的结构
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)
        
        # 加载预训练模型的参数
        self.load_state_dict(baseline_model.state_dict())
        
        # 保存高可信度神经元的索引
        self.high_credibility_neurons = high_credibility_neurons
        
        # 用于记录每层激活值的字典
        self.activations = {}
        
    def forward(self, x):
        # 第一个卷积块
        x = F.relu(self.conv1(x))
        if 'conv1' in self.high_credibility_neurons:
            mask = torch.zeros(x.shape[1], device=x.device)
            mask[self.high_credibility_neurons['conv1']] = 1
            x = x * mask.view(1, -1, 1, 1)
        self.activations['conv1'] = x.clone().detach()
        
        x = F.relu(self.conv2(x))
        if 'conv2' in self.high_credibility_neurons:
            mask = torch.zeros(x.shape[1], device=x.device)
            mask[self.high_credibility_neurons['conv2']] = 1
            x = x * mask.view(1, -1, 1, 1)
        self.activations['conv2'] = x.clone().detach()
        x = self.pool(x)
        
        # 第二个卷积块
        x = F.relu(self.conv3(x))
        if 'conv3' in self.high_credibility_neurons:
            mask = torch.zeros(x.shape[1], device=x.device)
            mask[self.high_credibility_neurons['conv3']] = 1
            x = x * mask.view(1, -1, 1, 1)
        self.activations['conv3'] = x.clone().detach()
        
        x = F.relu(self.conv4(x))
        if 'conv4' in self.high_credibility_neurons:
            mask = torch.zeros(x.shape[1], device=x.device)
            mask[self.high_credibility_neurons['conv4']] = 1
            x = x * mask.view(1, -1, 1, 1)
        self.activations['conv4'] = x.clone().detach()
        x = self.pool(x)
        
        # 全连接层
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        if 'fc1' in self.high_credibility_neurons:
            mask = torch.zeros(x.shape[1], device=x.device)
            mask[self.high_credibility_neurons['fc1']] = 1
            x = x * mask.view(1, -1)
        self.activations['fc1'] = x.clone().detach()
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        if 'fc2' in self.high_credibility_neurons:
            mask = torch.zeros(x.shape[1], device=x.device)
            mask[self.high_credibility_neurons['fc2']] = 1
            x = x * mask.view(1, -1)
        self.activations['fc2'] = x.clone().detach()
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x
    
    def get_activations(self):
        return self.activations

# 用于训练高可信度神经元分类器的函数
def train_hn_classifier(model, train_loader, test_loader, epochs=10, lr=0.001, device='cuda'):
    """训练高可信度神经元分类器
    
    Args:
        model: 高可信度神经元分类器
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        epochs: 训练轮数
        lr: 学习率
        device: 训练设备（'cuda'或'cpu'）
    
    Returns:
        训练好的模型
    """
    # 将模型移动到指定设备
    model = model.to(device)
    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 使用Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 统计损失和准确率
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # 计算训练集上的准确率
        train_accuracy = 100. * correct / total
        
        # 在测试集上评估模型
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        # 计算测试集上的准确率
        test_accuracy = 100. * test_correct / test_total
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')
    
    return model