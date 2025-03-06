import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    """基准CNN模型，用于MNIST分类任务
    
    这个模型采用类似LeNet的结构，包含两个卷积层和三个全连接层。
    在训练后，我们将计算每个神经元的可信度，并将网络划分为高可信度和低可信度两部分。
    """
    
    def __init__(self):
        super(BaselineCNN, self).__init__()
        # 第一个卷积层，输入通道1（灰度图），输出通道32，卷积核大小3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 第二个卷积层，输入通道32，输出通道64，卷积核大小3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 池化层，用于降低特征图的空间维度
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层1，输入维度64*7*7，输出维度128
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 全连接层2，输入维度128，输出维度64
        self.fc2 = nn.Linear(128, 64)
        # 输出层，输入维度64，输出维度10（对应10个数字类别）
        self.fc3 = nn.Linear(64, 10)
        # 用于记录每层激活值的字典，方便后续计算神经元可信度
        self.activations = {}
        # 用于记录每层梯度的字典，方便后续计算神经元可信度
        self.gradients = {}
        
    def forward(self, x):
        # 第一个卷积层 + ReLU激活 + 池化
        x = self.pool(F.relu(self.conv1(x)))
        self.activations['conv1'] = x.clone().detach()
        
        # 第二个卷积层 + ReLU激活 + 池化
        x = self.pool(F.relu(self.conv2(x)))
        self.activations['conv2'] = x.clone().detach()
        
        # 将特征图展平为一维向量
        x = x.view(-1, 64 * 7 * 7)
        
        # 第一个全连接层 + ReLU激活
        x = F.relu(self.fc1(x))
        self.activations['fc1'] = x.clone().detach()
        
        # 第二个全连接层 + ReLU激活
        x = F.relu(self.fc2(x))
        self.activations['fc2'] = x.clone().detach()
        
        # 输出层
        x = self.fc3(x)
        self.activations['fc3'] = x.clone().detach()
        
        return x
    
    def get_activations(self):
        """获取模型各层的激活值"""
        return self.activations
    
    def register_hooks(self):
        """注册梯度钩子，用于获取反向传播时的梯度信息"""
        def get_gradients(name):
            def hook(grad):
                self.gradients[name] = grad.clone().detach()
            return hook
        
        # 为每层激活值注册梯度钩子
        for name, activation in self.activations.items():
            if activation.requires_grad:
                activation.register_hook(get_gradients(name))
    
    def get_gradients(self):
        """获取模型各层的梯度"""
        return self.gradients

# 用于训练基准CNN模型的函数
def train_baseline(model, train_loader, test_loader, epochs=10, lr=0.001, device='cuda'):
    """训练基准CNN模型
    
    Args:
        model: 基准CNN模型
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

# 用于评估模型的函数
def evaluate_model(model, data_loader, device='cuda'):
    """评估模型在给定数据集上的性能
    
    Args:
        model: 要评估的模型
        data_loader: 数据加载器
        device: 评估设备（'cuda'或'cpu'）
    
    Returns:
        准确率
    """
    model = model.to(device)
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
    
    accuracy = 100. * correct / total
    return accuracy