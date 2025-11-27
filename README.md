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
│   ├── data.py                      # 数据处理工具
│   ├── model.py                     # 模型构建和训练
│   ├── utils.py                     # 可视化和预测工具
│   └── main.py                      # 主程序入口
│
├── images/                          # 结果可视化图片（上传到Git）
│   ├── data_distribution.png        # 数据集分布图
│   ├── training_history_10epochs.png # 训练曲线图
│   └── predictions.png              # 预测结果可视化
│
├── data/                            # 数据文件夹（本地存在，Git中隐藏）
│   ├── train_covid19/              # 训练数据
│   │   ├── yes/                    # COVID-19阳性CT图像
│   │   └── no/                     # COVID-19阴性CT图像
│   ├── test_healthcare/            # 测试数据
│   │   └── test/                   # 待预测CT图像
│   └── processed/                  # 处理后数据
│       ├── Train_covid/            # 训练集
│       └── Val_covid/              # 验证集
│
└── models/                          # 训练好的模型（本地存在，Git中隐藏）
    ├── covid_vgg16_10epochs.h5     # 10轮训练模型
    ├── covid_vgg16_50epochs.h5     # 50轮训练模型
    ├── history_10epochs.pkl        # 10轮训练历史
    └── history_50epochs.pkl        # 50轮训练历史
```

**说明**：
- `src/` - 包含所有源代码，已上传到Git
- `images/` - 训练和预测生成的可视化图片，已上传到Git用于结果展示
- `data/` 和 `models/` - 由于文件过大，仅保存在本地，不上传到Git

## 快速开始

### 环境要求

- Python 3.7+ (推荐3.10)
- 最少8GB内存，推荐16GB以上
- GPU可选但推荐使用，需要6GB以上显存
- 至少10GB可用存储空间

### 安装步骤
```bash
# 1. 克隆项目
git clone https://github.com/Wicky-XU/AI-for-Healthcare-Innovation.git
cd AI-for-Healthcare-Innovation

# 2. 安装依赖（推荐使用国内镜像加速）
pip install tensorflow numpy matplotlib opencv-python scikit-learn Pillow

# 或使用清华镜像
pip install tensorflow numpy matplotlib opencv-python scikit-learn Pillow -i https://pypi.tuna.tsinghua.edu.cn/simple
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

**使用命令行（推荐）：**
```bash
# 快速训练（10轮，约15-30分钟）
python -m src.main --mode train --epochs 10

# 完整训练（50轮，约2-4小时）  
python -m src.main --mode train --epochs 50

# 预测分析
python -m src.main --mode predict --epochs 10

# 模型对比
python -m src.main --mode compare
```

**在Python脚本中使用：**
```python
from src import data, model, utils

# 训练流程
train_dir, val_dir = data.split_data()
train_gen, val_gen = data.create_generators(train_dir, val_dir)
covid_model = model.build_model(version='simple')
history = model.train_model(covid_model, train_gen, val_gen, epochs=10)

# 可视化
utils.plot_training_history(10)

# 预测
test_images, filenames = data.load_test_images()
predictions = utils.predict_images(test_images, filenames, epochs=10)
```

## 文件说明

### 核心代码模块

- **config.py** - 项目配置管理，包含所有超参数和路径设置
- **data.py** - 数据处理工具，负责图像加载、预处理和数据分割
- **model.py** - 模型构建和训练功能，包含VGG16架构定义和训练流程
- **utils.py** - 工具模块，提供可视化和预测功能
- **main.py** - 主程序入口，提供命令行界面

### 可视化输出

运行训练或预测后，程序会自动在 `images/` 文件夹生成以下可视化图片：
- **data_distribution.png** - 训练集和验证集的类别分布图
- **training_history_Xepochs.png** - 训练过程的准确率和损失曲线
- **predictions.png** - 测试集预测结果的可视化展示

这些图片会自动保存，可用于结果分析和报告展示。

## 配置参数

主要配置在 `src/config.py` 中：
```python
# 图像参数
IMG_SIZE = (224, 224)          # VGG16输入尺寸
BATCH_SIZE = 32                # 批处理大小
LEARNING_RATE = 1e-3           # 学习率

# 训练参数
EPOCHS_10 = 10                 # 快速版本轮数
EPOCHS_50 = 50                 # 完整版本轮数
TRAIN_SPLIT = 0.6              # 训练验证分割比例

# 数据增强
AUGMENTATION = {
    'rotation_range': 20,       # 图像旋转范围
    'width_shift_range': 0.2,   # 宽度偏移
    'height_shift_range': 0.2,  # 高度偏移
    'zoom_range': 0.2,          # 缩放范围
    'horizontal_flip': True     # 水平翻转
}
```

## 版本对比

| 特性 | 10轮版本 | 50轮版本 |
|------|----------|----------|
| 训练时间 | 15-30分钟 | 2-4小时 |
| 数据增强 | 关闭 | 开启 |
| 模型架构 | 2层分类器 | 4层深度分类器 |
| 早停机制 | 5轮耐心 | 10轮耐心 |
| 学习率调度 | 固定学习率 | 自适应调整 |
| 适用场景 | 快速验证 | 生产部署 |

## 性能评估

### 评估指标

- **准确率**：正确分类的样本比例
- **损失值**：模型预测误差
- **训练曲线**：准确率和损失随轮次变化
- **置信度**：模型预测的确信程度

### 实际性能

根据最近一次10轮训练结果：
- **训练集准确率**：93.40%
- **验证集准确率**：95.24%
- **训练样本**：500张（253阳性 + 247阴性）
- **验证样本**：336张（170阳性 + 166阴性）

## 常见问题

**Q: GPU内存不足怎么办？**  
A: 在config.py中减少BATCH_SIZE到16或8

**Q: 训练时电脑发热/变慢？**  
A: 这是正常现象，建议：
- 确保散热良好
- 降低BATCH_SIZE减少负载
- 使用Google Colab训练

**Q: 如何切换训练版本？**  
A: 使用 `--epochs` 参数：
```bash
python -m src.main --mode train --epochs 10  # 10轮
python -m src.main --mode train --epochs 50  # 50轮
```

**Q: 依赖安装失败？**  
A: 建议使用国内镜像：
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow numpy matplotlib opencv-python
```

**Q: 输出信息显示为英文？**  
A: 为了避免编码问题和提升国际化程度，代码输出和可视化图表均使用英文，不影响使用

## 代码优化说明

本项目采用模块化重构，相比原notebook版本：
- 代码量减少43.5%（975行 → 551行）
- 消除80%重复代码
- 平均文件大小92行，易于阅读维护
- 10轮和50轮版本共享核心代码
- 支持命令行、Python脚本、Jupyter多种调用方式

## 重要声明

本项目仅用于研究和教育目的。在任何医学应用之前，需要经过严格的临床验证。诊断决策应始终咨询专业医生。使用时请确保遵守相关数据保护法规，保护患者隐私。

## 致谢

感谢帝国理工学院（Imperial College London）数据科学研究所提供的学习平台和技术参考，也特别感谢项目课程中诸位教授的悉心指导和Group5其他4名团队成员的鼎力支持。

同时感谢VGG团队开发的优秀预训练模型，TensorFlow和Keras开发团队提供的深度学习框架，以及开源社区的无私贡献。

本项目源自2022年8月在帝国理工学院参加的AI与数据科学在线夏校课程，并在后续学习中不断完善和发展。

---

**技术栈**: Python · TensorFlow · VGG16 · OpenCV

**应用领域**: 医学影像分析 · 深度学习 · 计算机视觉

**项目目标**: 探索AI在医学影像诊断中的应用潜力

**项目版本**: 2.0（模块化精简版）