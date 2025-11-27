"""
COVID-19 肺部CT图像分类项目 - 模型构建和训练模块
===============================================

模型构建和训练模块 - 负责模型架构构建、训练配置、回调设置和训练执行等功能
基于notebook中的模型相关代码重构而成

"""

import pickle
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from . import config

def build_model(version='simple'):
    """构建VGG16迁移学习模型
    
    Args:
        version: 'simple' (10轮) 或 'enhanced' (50轮)
    """
    # 加载预训练VGG16（不含顶层）
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(*config.IMG_SIZE, 3)
    )
    
    # 冻结VGG16层
    base_model.trainable = False
    
    # 添加自定义分类层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    if version == 'enhanced':
        # 50轮版本：更深的分类器
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
    else:
        # 10轮版本：简单分类器
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
    
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"模型构建完成 ({version}版本)")
    print(f"  - 总参数: {model.count_params():,}")
    print(f"  - 可训练参数: {sum([np.prod(v.shape) for v in model.trainable_weights]):,}")
    
    return model

def get_callbacks(epochs, version='simple'):
    """获取训练回调函数"""
    patience = 10 if epochs == 50 else 5
    
    callbacks = [
        ModelCheckpoint(
            config.get_model_path(epochs),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    if version == 'enhanced':
        callbacks.append(
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        )
    
    return callbacks

def train_model(model, train_gen, val_gen, epochs, version='simple'):
    """训练模型"""
    callbacks = get_callbacks(epochs, version)
    
    print(f"\n开始训练 ({epochs}轮)...")
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # 保存训练历史
    with open(config.get_history_path(epochs), 'wb') as f:
        pickle.dump(history.history, f)
    
    print(f"\n训练完成! 模型已保存至: {config.get_model_path(epochs)}")
    
    return history