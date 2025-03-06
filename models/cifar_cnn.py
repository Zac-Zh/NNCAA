import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFARCNN(nn.Module):
    """基准CNN模型，用于CIFAR-10分类任务
    
    这个模型采用更深的卷积网络结构，以适应CIFAR-10数据集的复杂性。
    在训练后，我们将计算每个神经元的可信度，并将网络划分为高可信度和低可信度两部分。
    """
    
    def __init__(self):
        super(CIFARCNN, self).__init__()
        # 第一个卷积块
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # 第二个卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        
        # Dropout层
        self.dropout = nn.Dropout(0.5)
        
        # 用于记录每层激活值的字典
        self.activations = {}
        # 用于记录每层梯度的字典
        self.gradients = {}
        
    def forward(self, x):
        # 第一个卷积块
        x = F.relu(self.conv1(x))
        self.activations['conv1'] = x.clone().detach()
        x = F.relu(self.conv2(x))
        self.activations['conv2'] = x.clone().detach()
        x = self.pool(x)
        
        # 第二个卷积块
        x = F.relu(self.conv3(x))
        self.activations['conv3'] = x.clone().detach()
        x = F.relu(self.conv4(x))
        self.activations['conv4'] = x.clone().detach()
        x = self.pool(x)
        
        # 展平
        x = x.view(-1, 128 * 8 * 8)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        self.activations['fc1'] = x.clone().detach()
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        self.activations['fc2'] = x.clone().detach()
        x = self.dropout(x)
        
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
def train_cifar(model, train_loader, test_loader, epochs=10, lr=0.001, device='cuda'):
    """训练CIFAR-10基准CNN模型
    
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
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch: {epoch + 1}, Batch: {batch_idx + 1}, Loss: {total_loss/(batch_idx+1):.3f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        # 在测试集上评估
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = 100.*correct/total
        print(f'\nTest set: Average loss: {test_loss/len(test_loader):.3f}, '
              f'Accuracy: {acc:.2f}%\n')
        
        # 更新学习率
        scheduler.step(acc)
        
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'cifar_cnn_best.pth')
    
    # 加载最佳模型
    model.load_state_dict(torch.load('cifar_cnn_best.pth'))
    return model