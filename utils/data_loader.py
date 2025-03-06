import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_mnist_data(batch_size=128, num_workers=4):
    """加载MNIST数据集
    
    Args:
        batch_size: 批量大小
        num_workers: 数据加载的线程数
    
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    # 定义数据预处理转换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的标准化参数
    ])
    
    # 加载训练集
    train_dataset = datasets.MNIST(
        root='./data',  # 数据保存路径
        train=True,  # 训练集
        download=True,  # 如果数据不存在则下载
        transform=transform  # 应用数据转换
    )
    
    # 加载测试集
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 打乱训练数据
        num_workers=num_workers,
        pin_memory=True  # 使用固定内存，可以加速GPU训练
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试数据不需要打乱
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

def get_sample_data(data_loader, num_samples=100):
    """从数据加载器中获取指定数量的样本
    
    Args:
        data_loader: 数据加载器
        num_samples: 需要的样本数量
    
    Returns:
        inputs: 输入数据张量
        labels: 标签张量
    """
    inputs = []
    labels = []
    sample_count = 0
    
    for batch_inputs, batch_labels in data_loader:
        batch_size = batch_inputs.size(0)
        if sample_count + batch_size > num_samples:
            # 只取需要的样本数量
            inputs.append(batch_inputs[:num_samples - sample_count])
            labels.append(batch_labels[:num_samples - sample_count])
            break
        else:
            inputs.append(batch_inputs)
            labels.append(batch_labels)
            sample_count += batch_size
        
        if sample_count >= num_samples:
            break
    
    # 合并所有样本
    inputs = torch.cat(inputs, dim=0)
    labels = torch.cat(labels, dim=0)
    
    return inputs, labels

def create_noisy_samples(inputs, noise_std=0.1):
    """为输入数据添加高斯噪声
    
    Args:
        inputs: 输入数据张量
        noise_std: 噪声的标准差
    
    Returns:
        noisy_inputs: 添加噪声后的数据张量
    """
    noise = torch.randn_like(inputs) * noise_std
    noisy_inputs = inputs + noise
    # 将像素值裁剪到[0,1]范围
    noisy_inputs = torch.clamp(noisy_inputs, 0, 1)
    return noisy_inputs