#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具函数模块
提供项目所需的通用工具函数
"""

import os
import logging
from pathlib import Path
from datetime import datetime


def setup_logging(log_level=logging.INFO, log_file=None):
    """
    设置日志系统
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=log_level,
            format=log_format,
            filename=log_file,
            filemode='a'
        )
    else:
        logging.basicConfig(
            level=log_level,
            format=log_format
        )
    
    print("日志系统初始化完成")


def validate_input_file(input_file):
    """
    验证输入文件
    
    Args:
        input_file: 输入文件路径
        
    Returns:
        bool: 文件是否有效
    """
    file_path = Path(input_file)
    
    if not file_path.exists():
        print(f"错误: 输入文件不存在 - {input_file}")
        return False
    
    # 检查文件扩展名
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    if file_path.suffix.lower() not in valid_extensions:
        print(f"警告: 文件扩展名可能不受支持 - {file_path.suffix}")
    
    # 检查文件大小
    file_size = file_path.stat().st_size
    if file_size == 0:
        print(f"错误: 文件为空 - {input_file}")
        return False
    
    print(f"输入文件验证通过: {input_file} ({file_size / 1024 / 1024:.1f} MB)")
    return True


def prepare_output_directory(output_file):
    """
    准备输出目录
    
    Args:
        output_file: 输出文件路径
        
    Returns:
        bool: 目录创建是否成功
    """
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"输出目录准备完成: {output_path.parent}")
        return True
    except Exception as e:
        print(f"输出目录创建失败: {e}")
        return False


def save_transcript_results(results, output_file, input_file=None):
    """
    保存转录结果到文件
    
    Args:
        results: 转录结果列表
        output_file: 输出文件路径
        input_file: 输入文件路径（可选）
        
    Returns:
        bool: 保存是否成功
    """
    try:
        # 准备输出目录
        if not prepare_output_directory(output_file):
            return False
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入文件头
            f.write("# Auto Transcript Results\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if input_file:
                f.write(f"# Input: {input_file}\n")
            f.write("# " + "="*50 + "\n\n")
            
            # 写入转录结果
            for line in results:
                f.write(line + '\n')
        
        print(f"转录结果已保存: {output_file}")
        print(f"总计 {len(results)} 个转录片段")
        return True
        
    except Exception as e:
        print(f"保存转录结果失败: {e}")
        return False


def format_timestamp(seconds):
    """
    格式化时间戳
    
    Args:
        seconds: 秒数
        
    Returns:
        str: 格式化的时间戳 (HH:MM:SS)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def check_dependencies():
    """
    检查项目依赖
    
    Returns:
        bool: 依赖是否完整
    """
    required_packages = [
        'cv2',
        'PIL',
        'numpy',
        'moviepy',
        'pydub',
        'speech_recognition',
        'googletrans'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            elif package == 'numpy':
                import numpy
            elif package == 'moviepy':
                from moviepy.editor import VideoFileClip
            elif package == 'pydub':
                from pydub import AudioSegment
            elif package == 'speech_recognition':
                import speech_recognition
            elif package == 'googletrans':
                from googletrans import Translator
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install opencv-python pillow numpy moviepy pydub SpeechRecognition googletrans==4.0.0rc1")
        return False
    
    print("所有依赖包检查通过")
    return True


def get_project_info():
    """
    获取项目信息
    
    Returns:
        dict: 项目信息
    """
    return {
        'name': 'Auto Transcript Project',
        'version': '1.0.0',
        'description': '自动视频转录系统',
        'author': 'Your Name',
        'features': [
            '人脸识别',
            '语音识别', 
            '自动翻译',
            '时间戳标记'
        ]
    }


def print_project_banner():
    """打印项目横幅"""
    info = get_project_info()
    
    print("=" * 60)
    print(f"{info['name']} v{info['version']}")
    print(info['description'])
    print("=" * 60)
    print("功能特点:")
    for feature in info['features']:
        print(f"  - {feature}")
    print("=" * 60)


def cleanup_temp_files(temp_dir="data/temp"):
    """
    清理临时文件
    
    Args:
        temp_dir: 临时文件目录
    """
    try:
        temp_path = Path(temp_dir)
        if temp_path.exists():
            for file in temp_path.glob("*"):
                if file.is_file():
                    file.unlink()
            print(f"临时文件清理完成: {temp_dir}")
    except Exception as e:
        print(f"临时文件清理失败: {e}")


if __name__ == "__main__":
    print("工具函数模块测试")
    
    # 测试项目信息
    print_project_banner()
    
    # 测试依赖检查
    check_dependencies()
    
    # 测试时间戳格式化
    print(f"时间戳测试: {format_timestamp(125)} (应该是 00:02:05)")