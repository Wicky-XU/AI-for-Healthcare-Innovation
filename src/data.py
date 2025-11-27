"""
COVID-19 肺部CT图像分类项目 - 数据处理工具模块
=============================================

数据处理模块 - 负责数据加载、预处理、分割、增强和验证等功能
基于notebook中的数据处理流程重构而成

"""

import os
import shutil
import random
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from . import config

def check_and_setup():
    """检查数据并创建必要的目录"""
    # 检查训练数据
    if not os.path.exists(config.TRAIN_DIR):
        raise FileNotFoundError(f"训练数据目录不存在: {config.TRAIN_DIR}")
    
    yes_count = len(os.listdir(os.path.join(config.TRAIN_DIR, 'yes')))
    no_count = len(os.listdir(os.path.join(config.TRAIN_DIR, 'no')))
    
    print(f"找到训练数据 - COVID阳性: {yes_count}, COVID阴性: {no_count}")
    
    # 创建必要目录
    for dir_path in [config.MODEL_DIR, config.PROCESSED_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    return yes_count, no_count

def split_data(split_ratio=None):
    """分割训练集和验证集"""
    if split_ratio is None:
        split_ratio = config.TRAIN_SPLIT
    
    train_dir = os.path.join(config.PROCESSED_DIR, 'Train_covid')
    val_dir = os.path.join(config.PROCESSED_DIR, 'Val_covid')
    
    # 清空并重建目录
    for base_dir in [train_dir, val_dir]:
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
        for label in ['yes', 'no']:
            os.makedirs(os.path.join(base_dir, label), exist_ok=True)
    
    # 分割数据
    random.seed(config.RANDOM_SEED)
    for label in ['yes', 'no']:
        source_dir = os.path.join(config.TRAIN_DIR, label)
        files = os.listdir(source_dir)
        random.shuffle(files)
        
        split_idx = int(len(files) * split_ratio)
        train_files = files[:split_idx]
        val_files = files[split_idx:]
        
        # 复制文件
        for f in train_files:
            shutil.copy2(
                os.path.join(source_dir, f),
                os.path.join(train_dir, label, f)
            )
        for f in val_files:
            shutil.copy2(
                os.path.join(source_dir, f),
                os.path.join(val_dir, label, f)
            )
    
    print(f"数据分割完成 - 训练集: {split_ratio*100:.0f}%, 验证集: {(1-split_ratio)*100:.0f}%")
    return train_dir, val_dir

def create_generators(train_dir, val_dir, augment=True):
    """创建数据生成器"""
    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            **config.AUGMENTATION
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )
    
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    print(f"数据生成器创建完成 - 训练样本: {train_gen.samples}, 验证样本: {val_gen.samples}")
    return train_gen, val_gen

def load_test_images():
    """加载测试图像"""
    import cv2
    
    if not os.path.exists(config.TEST_DIR):
        print("测试数据目录不存在")
        return [], []
    
    test_images = []
    filenames = []
    
    for filename in os.listdir(config.TEST_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(config.TEST_DIR, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, config.IMG_SIZE)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                test_images.append(img / 255.0)
                filenames.append(filename)
    
    print(f"加载了 {len(test_images)} 张测试图像")
    return np.array(test_images), filenames