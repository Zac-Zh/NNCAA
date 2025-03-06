import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_cifar10_data(batch_size=128, num_workers=4):
    """加载CIFAR-10数据集
    
    Args:
        batch_size: 批量大小
        num_workers: 数据加载的线程数
    
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    # 定义数据预处理转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10标准化参数
    ])
    
    # 加载训练集
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # 加载测试集
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

def get_class_names():
    """获取CIFAR-10数据集的类别名称"""
    return ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']