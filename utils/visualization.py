import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from sklearn.manifold import TSNE

def visualize_adversarial_samples(original_images, perturbations, adversarial_images,
                                original_labels, predicted_labels, class_names=None,
                                num_samples=5, save_path=None):
    """可视化对抗样本
    
    Args:
        original_images: 原始图像张量 (N, C, H, W)
        perturbations: 扰动张量 (N, C, H, W)
        adversarial_images: 对抗样本张量 (N, C, H, W)
        original_labels: 原始标签
        predicted_labels: 预测标签
        class_names: 类别名称列表
        num_samples: 要显示的样本数量
        save_path: 保存图像的路径
    """
    # 确保只显示指定数量的样本
    original_images = original_images[:num_samples]
    perturbations = perturbations[:num_samples]
    adversarial_images = adversarial_images[:num_samples]
    original_labels = original_labels[:num_samples]
    predicted_labels = predicted_labels[:num_samples]
    
    # 创建图像网格
    fig, axes = plt.subplots(3, num_samples, figsize=(3*num_samples, 9))
    
    # 设置标题
    fig.suptitle('对抗样本分析', fontsize=16)
    
    # 归一化扰动以便可视化
    perturbations = (perturbations - perturbations.min()) / (perturbations.max() - perturbations.min())
    
    # 显示原始图像、扰动和对抗样本
    for i in range(num_samples):
        # 原始图像
        img = original_images[i].permute(1, 2, 0).cpu().numpy()
        if img.shape[2] == 1:  # 如果是灰度图
            img = img.squeeze()
        axes[0, i].imshow(img)
        label = class_names[original_labels[i]] if class_names else str(original_labels[i].item())
        axes[0, i].set_title(f'原始: {label}')
        axes[0, i].axis('off')
        
        # 扰动
        pert = perturbations[i].permute(1, 2, 0).cpu().numpy()
        if pert.shape[2] == 1:  # 如果是灰度图
            pert = pert.squeeze()
        axes[1, i].imshow(pert)
        axes[1, i].set_title('扰动')
        axes[1, i].axis('off')
        
        # 对抗样本
        adv = adversarial_images[i].permute(1, 2, 0).cpu().numpy()
        if adv.shape[2] == 1:  # 如果是灰度图
            adv = adv.squeeze()
        axes[2, i].imshow(adv)
        pred_label = class_names[predicted_labels[i]] if class_names else str(predicted_labels[i].item())
        axes[2, i].set_title(f'对抗: {pred_label}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_feature_space(features, labels, class_names=None, save_path=None):
    """使用t-SNE可视化特征空间
    
    Args:
        features: 特征向量 (N, D)
        labels: 标签 (N,)
        class_names: 类别名称列表
        save_path: 保存图像的路径
    """
    # 使用t-SNE降维到2D
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features.cpu().numpy())
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels.cpu().numpy(), cmap='tab10')
    
    # 添加图例
    if class_names:
        plt.legend(handles=scatter.legend_elements()[0], labels=class_names,
                  title='类别', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title('特征空间可视化 (t-SNE)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_decision_boundary_distance(distances, labels, predictions, class_names=None, save_path=None):
    """绘制样本到决策边界的距离分布
    
    Args:
        distances: 样本到决策边界的距离
        labels: 原始标签
        predictions: 模型预测
        class_names: 类别名称列表
        save_path: 保存图像的路径
    """
    plt.figure(figsize=(10, 6))
    
    # 区分正确分类和错误分类的样本
    correct_mask = labels == predictions
    correct_distances = distances[correct_mask]
    wrong_distances = distances[~correct_mask]
    
    # 绘制直方图
    plt.hist(correct_distances.cpu().numpy(), bins=30, alpha=0.5, label='正确分类',
             color='blue', density=True)
    plt.hist(wrong_distances.cpu().numpy(), bins=30, alpha=0.5, label='错误分类',
             color='red', density=True)
    
    plt.xlabel('到决策边界的距离')
    plt.ylabel('密度')
    plt.title('样本到决策边界的距离分布')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()