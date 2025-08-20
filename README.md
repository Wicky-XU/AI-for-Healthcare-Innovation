# COVID-19肺部CT图像分类项目

基于VGG16深度学习的COVID-19肺部CT图像智能分类系统

## 项目简介

本项目使用深度学习技术对肺部CT图像进行COVID-19诊断分类，基于预训练的VGG16模型构建，支持10轮和50轮两种训练配置，提供完整的数据处理、模型训练、预测分析和结果可视化功能。

### 主要特性

- **双版本训练**：支持10轮快速训练和50轮完整训练
- **智能预处理**：自动数据分割、增强和验证  
- **高级架构**：基于VGG16的增强分类网络
- **全面监控**：实时训练监控和性能分析
- **预测分析**：详细的置信度和不确定性评估
- **可视化**：丰富的图表和统计分析
- **模块化设计**：清晰的代码结构便于维护

## 项目结构

```
AI-for-Healthcare-Innovation/
├── README.md                         # 项目文档（本文件）
├── requirements.txt                  # Python依赖包列表
├── .gitignore                       # Git忽略文件配置
│
├── src/                             # 模块化源代码
│   ├── __init__.py                  # Python包初始化
│   ├── config.py                    # 配置管理模块
│   ├── data_utils.py                # 数据处理工具
│   ├── model_utils.py               # 模型构建和训练
│   ├── prediction_utils.py          # 预测和评估
│   ├── visualization.py             # 可视化工具
│   └── main.py                     # 主程序入口
│
├── notebooks/                       # Jupyter笔记本
│   ├── train_10_epochs.ipynb       # 10轮快速训练版本
│   └── train_50_epochs.ipynb       # 50轮完整训练版本
│
├── docs/                           # 文档文件夹
│   ├── python_basics.ipynb         # Python基础操作教程
│   └── proj_tech_classification.ipynb  # 项目技术分类模板
│
├── data/                           # 数据文件夹（本地存在，Git中隐藏）
│   ├── train_covid19/             # 训练数据
│   │   ├── yes/                   # COVID-19阳性CT图像
│   │   └── no/                    # COVID-19阴性CT图像
│   ├── test_healthcare/           # 测试数据
│   │   └── test/                  # 待预测CT图像
│   └── processed/                 # 处理后数据
│       ├── Train_covid/           # 训练集
│       └── Val_covid/             # 验证集
│
└── models/                         # 训练好的模型（本地存在，Git中隐藏）
    ├── covid_classifier_vgg16_10epochs.h5     # 10轮训练模型
    ├── covid_classifier_vgg16_50epochs.h5     # 50轮训练模型
    ├── best_model_10epochs.h5                 # 10轮最佳检查点
    ├── best_model_50epochs.h5                 # 50轮最佳检查点
    ├── training_history_10epochs.pkl          # 10轮训练历史
    └── training_history_50epochs_enhanced.pkl # 50轮训练历史
```

## 快速开始

### 环境要求

- Python 3.7+ (推荐3.8或3.9)
- 最少8GB内存，推荐16GB以上
- GPU可选但推荐使用，需要6GB以上显存
- 至少10GB可用存储空间

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/Wicky-XU/AI-for-Healthcare-Innovation.git
cd AI-for-Healthcare-Innovation

# 2. 创建虚拟环境
python -m venv venv

# 3. 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. 安装依赖
pip install -r requirements.txt
```

### 数据准备

请将数据按以下结构组织：

```
data/
├── train_covid19/
│   ├── yes/          # COVID-19阳性CT图像
│   └── no/           # COVID-19阴性CT图像
└── test_healthcare/
    └── test/         # 待预测CT图像
```

### 运行项目

**使用Jupyter Notebook（推荐新手）：**

```bash
jupyter notebook
```

然后在浏览器中打开：
- `notebooks/train_10_epochs.ipynb` - 快速训练版本
- `notebooks/train_50_epochs.ipynb` - 完整训练版本

**使用Python脚本（推荐生产环境）：**

```bash
# 快速训练（10轮）
python src/main.py --mode train --version 10_epochs

# 完整训练（50轮）  
python src/main.py --mode train --version 50_epochs

# 预测分析
python src/main.py --mode predict --version 10_epochs

# 模型对比
python src/main.py --mode compare
```

## 文件说明

### 核心代码模块

- **config.py** - 项目配置管理，包含所有超参数和路径设置
- **data_utils.py** - 数据处理工具，负责图像加载、预处理和数据分割
- **model_utils.py** - 模型构建和训练功能，包含VGG16架构定义和训练流程
- **prediction_utils.py** - 预测和评估模块，提供模型预测和结果分析功能
- **visualization.py** - 可视化工具，生成训练曲线和预测结果图表
- **main.py** - 主程序入口，整合所有功能的命令行界面

### Jupyter笔记本

- **train_10_epochs.ipynb** - 10轮快速训练版本，适合概念验证和快速测试，训练时间约30-60分钟
- **train_50_epochs.ipynb** - 50轮完整训练版本，包含增强的数据处理和更深的分类器，适合生产部署

### 文档

- **python_basics.ipynb** - Python基础操作教程，包含项目所需的基础编程知识
- **proj_tech_classification.ipynb** - 项目技术分类模板，展示深度学习在医学影像分类中的应用方法

## 配置参数

主要配置在 `src/config.py` 中：

```python
# 图像参数
IMG_SIZE = (224, 224)          # VGG16输入尺寸
BATCH_SIZE = 32                # 批处理大小
LEARNING_RATE = 1e-3           # 学习率

# 训练参数
NUM_EPOCHS_10 = 10             # 快速版本轮数
NUM_EPOCHS_50 = 50             # 完整版本轮数
TRAIN_VAL_SPLIT = 0.6          # 训练验证分割比例

# 数据增强
ROTATION_RANGE = 20            # 图像旋转范围
WIDTH_SHIFT_RANGE = 0.2        # 宽度偏移
HEIGHT_SHIFT_RANGE = 0.2       # 高度偏移
ZOOM_RANGE = 0.2               # 缩放范围
```

## 版本对比

| 特性 | 10轮版本 | 50轮版本 |
|------|----------|----------|
| 训练时间 | 30-60分钟 | 2-4小时 |
| 数据增强 | 基础增强 | 增强版策略 |
| 模型架构 | 标准分类器 | 4层深度分类器 |
| 早停机制 | 5轮耐心 | 10轮耐心 |
| 学习率调度 | 固定学习率 | 自适应调整 |
| 监控指标 | 准确率和损失 | 全面指标监控 |
| 适用场景 | 快速验证 | 生产部署 |

## 性能评估

### 评估指标

- **准确率**：正确分类的样本比例
- **精确度**：预测为阳性中真正阳性的比例  
- **召回率**：实际阳性中被正确识别的比例
- **F1分数**：精确度和召回率的调和平均
- **置信度**：模型预测的确信程度
- **不确定性**：预测结果的不确定性评估

### 预期性能

- **10轮版本**：验证准确率通常在80-90%，适合快速原型验证
- **50轮版本**：验证准确率可达85-95%，适合实际应用部署

## 常见问题

**GPU内存不足**：在config.py中减少BATCH_SIZE到16或更小

**依赖安装问题**：确保使用Python 3.7+，建议在虚拟环境中安装

**数据路径错误**：检查data文件夹结构是否正确，确保图像文件格式为jpg/png

**训练中断恢复**：程序会自动保存最佳模型检查点，可以从中断处继续

## 重要声明

本项目仅用于研究和教育目的。在任何医学应用之前，需要经过严格的临床验证。诊断决策应始终咨询专业医生。使用时请确保遵守相关数据保护法规，保护患者隐私。

## 致谢

感谢帝国理工学院（Imperial College London）数据科学研究所提供的学习平台和技术参考，也特别感谢项目课程中诸位教授的悉心指导和Group5其他4名团队成员的鼎力支持。

同时感谢VGG团队开发的优秀预训练模型，TensorFlow和Keras开发团队提供的深度学习框架，以及开源社区的无私贡献。

本项目源自2022年8月在帝国理工学院参加的AI与数据科学在线夏校课程，并在后续学习中不断完善和发展。

---

**技术栈**: Python • TensorFlow • VGG16 • Jupyter • OpenCV

**应用领域**: 医学影像分析 • 深度学习 • 计算机视觉

**项目目标**: 探索AI在医学影像诊断中的应用潜力
