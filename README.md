# 神经网络可信度评估与对抗训练

本项目实现了基于神经元可信度的对抗训练框架，通过区分高可信度神经元(HNs)和低可信度神经元(LNs)，构建对抗性训练机制以提高模型的泛化能力和鲁棒性。

## 项目结构

```
├── main.py                 # 主程序入口（MNIST数据集）
├── main_cifar.py           # CIFAR-10数据集的主程序入口
├── models/                 # 模型定义
│   ├── baseline_cnn.py     # 基准CNN模型（MNIST）
│   ├── cifar_cnn.py        # 基准CNN模型（CIFAR-10）
│   ├── hn_classifier.py    # 高可信度神经元分类器
│   └── ln_generator.py     # 低可信度神经元对抗生成器
├── utils/                  # 工具函数
│   ├── credibility.py      # 神经元可信度计算
│   ├── data_loader.py      # MNIST数据加载与预处理
│   ├── cifar_loader.py     # CIFAR-10数据加载与预处理
│   └── visualization.py    # 结果可视化
└── requirements.txt        # 项目依赖
```

## 实验流程

1. **预训练阶段**：训练基础CNN模型用于分类任务（MNIST或CIFAR-10）
2. **神经元可信度计算**：使用因果推断计算每个神经元的可信度
3. **网络划分**：将神经元划分为高可信度(HNs)和低可信度(LNs)两组
4. **对抗训练**：
   - HNs组成分类器(Classifier)
   - LNs组成对抗网络(Adversarial Generator)
   - 交替训练两个网络

## 运行方式

```bash
# 安装依赖
pip install -r requirements.txt

# 运行MNIST实验
python main.py

# 运行CIFAR-10实验（带可视化）
python main_cifar.py --visualize --save-dir ./results_cifar
```

## 可视化分析

本项目提供了多种可视化工具来分析对抗样本：

1. **对抗样本可视化**：展示原始图像、扰动和生成的对抗样本
2. **决策边界距离分析**：计算样本到决策边界的距离，分析对抗样本是否接近决策边界
3. **特征空间可视化**：使用t-SNE降维展示样本在特征空间中的分布

## 评估指标

- 基准分类准确率：HNs在正常测试集上的准确率
- 对抗准确率：HNs在LNs生成的样本上的准确率
- 泛化能力：HNs在其他噪声或对抗扰动上的鲁棒性
- 攻击成功率：LNs生成样本导致HNs误分类的比例