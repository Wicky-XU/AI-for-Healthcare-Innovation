"""
COVID-19 肺部CT图像分类项目 - 配置管理模块
================================================

项目配置 - 统一管理项目的所有配置参数，包括路径、模型参数、训练参数等
基于notebook中的配置重构而成

"""

import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# 数据路径
TRAIN_DIR = os.path.join(DATA_DIR, 'train_covid19')
TEST_DIR = os.path.join(DATA_DIR, 'test_healthcare', 'test')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# 模型参数
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# 训练参数
EPOCHS_10 = 10
EPOCHS_50 = 50
TRAIN_SPLIT = 0.6
RANDOM_SEED = 42

# 数据增强参数
AUGMENTATION = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

# 模型保存配置
def get_model_path(epochs):
    """获取模型保存路径"""
    return os.path.join(MODEL_DIR, f'covid_vgg16_{epochs}epochs.h5')

def get_history_path(epochs):
    """获取训练历史保存路径"""
    return os.path.join(MODEL_DIR, f'history_{epochs}epochs.pkl')