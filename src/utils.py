"""
COVID-19 肺部CT图像分类项目 - 预测和评估模块
==========================================

工具模块 - 负责可视化、模型预测、结果分析、评估指标计算和结果保存等功能
基于notebook中的预测相关代码重构而成

"""

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from . import config

# 可视化函数

def plot_training_history(epochs):
    """绘制训练历史曲线"""
    with open(config.get_history_path(epochs), 'rb') as f:
        history = pickle.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 准确率曲线
    axes[0].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 损失曲线
    axes[1].plot(history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存到images文件夹
    os.makedirs('images', exist_ok=True)
    plt.savefig(f'images/training_history_{epochs}epochs.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 打印最终指标
    print(f"\nTraining Results ({epochs} epochs):")
    print(f"  Best Validation Accuracy: {max(history['val_accuracy']):.4f}")
    print(f"  Final Training Accuracy: {history['accuracy'][-1]:.4f}")
    print(f"  Final Validation Accuracy: {history['val_accuracy'][-1]:.4f}")

def plot_predictions(images, filenames, predictions, top_n=10):
    """可视化预测结果"""
    n_display = min(top_n, len(images))
    cols = 5
    rows = (n_display + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    axes = axes.flatten() if n_display > 1 else [axes]
    
    for idx in range(n_display):
        axes[idx].imshow(images[idx])
        prob = predictions[idx][0]
        label = 'COVID-19 Positive' if prob > 0.5 else 'COVID-19 Negative'
        color = 'red' if prob > 0.5 else 'green'
        confidence = prob if prob > 0.5 else 1 - prob
        
        axes[idx].set_title(
            f'{label}\nConfidence: {confidence:.2%}',
            color=color,
            fontweight='bold'
        )
        axes[idx].axis('off')
    
    # 隐藏多余子图
    for idx in range(n_display, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # 保存到images文件夹
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/predictions.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_class_distribution(train_dir, val_dir):
    """绘制数据集类别分布"""
    train_yes = len(os.listdir(os.path.join(train_dir, 'yes')))
    train_no = len(os.listdir(os.path.join(train_dir, 'no')))
    val_yes = len(os.listdir(os.path.join(val_dir, 'yes')))
    val_no = len(os.listdir(os.path.join(val_dir, 'no')))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 训练集分布
    axes[0].bar(['Positive', 'Negative'], [train_yes, train_no], color=['#ff6b6b', '#4ecdc4'])
    axes[0].set_title('Training Set Distribution', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of Samples')
    for i, v in enumerate([train_yes, train_no]):
        axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')
    
    # 验证集分布
    axes[1].bar(['Positive', 'Negative'], [val_yes, val_no], color=['#ff6b6b', '#4ecdc4'])
    axes[1].set_title('Validation Set Distribution', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Number of Samples')
    for i, v in enumerate([val_yes, val_no]):
        axes[1].text(i, v + 2, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    # 保存到images文件夹
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/data_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

# 预测函数

def predict_images(test_images, filenames, epochs):
    """对测试图像进行预测"""
    model_path = config.get_model_path(epochs)
    
    print(f"\nLoading model: {model_path}")
    model = load_model(model_path)
    
    print(f"Predicting {len(test_images)} images...")
    predictions = model.predict(test_images, verbose=0)
    
    # 打印预测结果
    print("\nPrediction Results:")
    print("-" * 70)
    print(f"{'Filename':<30} {'Prediction':<20} {'Confidence':<10}")
    print("-" * 70)
    
    for filename, pred in zip(filenames, predictions):
        prob = pred[0]
        label = 'COVID-19 Positive' if prob > 0.5 else 'COVID-19 Negative'
        confidence = prob if prob > 0.5 else 1 - prob
        print(f"{filename:<30} {label:<20} {confidence:>8.2%}")
    
    print("-" * 70)
    
    # 统计
    positive_count = np.sum(predictions > 0.5)
    print(f"\nPrediction Statistics:")
    print(f"  COVID-19 Positive: {positive_count} ({positive_count/len(predictions)*100:.1f}%)")
    print(f"  COVID-19 Negative: {len(predictions)-positive_count} ({(1-positive_count/len(predictions))*100:.1f}%)")
    
    return predictions

def compare_models():
    """比较10轮和50轮模型性能"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, epochs in enumerate([10, 50]):
        try:
            with open(config.get_history_path(epochs), 'rb') as f:
                history = pickle.load(f)
            
            axes[idx].plot(history['accuracy'], label='Training', linewidth=2)
            axes[idx].plot(history['val_accuracy'], label='Validation', linewidth=2)
            axes[idx].set_title(f'{epochs} Epochs Training', fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel('Accuracy')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            
            final_acc = history['val_accuracy'][-1]
            axes[idx].text(0.5, 0.95, f'Final Val Accuracy: {final_acc:.4f}',
                          transform=axes[idx].transAxes,
                          ha='center', va='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        except FileNotFoundError:
            axes[idx].text(0.5, 0.5, f'{epochs} epochs training history not found',
                          transform=axes[idx].transAxes,
                          ha='center', va='center')
    
    plt.tight_layout()
    
    # 保存到images文件夹
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()