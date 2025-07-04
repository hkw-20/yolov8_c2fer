import os
import yaml
import torch
from ultralytics import YOLO
from ultralytics.nn.modules.custom import C2fER, ECA, OBB  # 导入自定义模块

def main():
    print("==================================================")
    print("YOLOv8 Training with C2fER Module")
    print("==================================================")
    
    # 创建自定义模型配置文件
    create_custom_config()
    
    # 准备数据集
    prepare_dataset()
    
    # 训练模型
    train_model()
    
    print("✅ Training completed successfully!")
    print("==================================================")

def create_custom_config():
    """创建包含 C2fER 模块的自定义模型配置"""
    config = {
        'nc': 80,  # 类别数 (COCO数据集)
        'depth_multiple': 0.33,  # 模型深度倍数
        'width_multiple': 0.50,  # 层通道数倍数
        'backbone': [
            [-1, 1, 'Conv', [64, 3, 2]],  # 0-P1/2
            [-1, 1, 'Conv', [128, 3, 2]],  # 1-P2/4
            [-1, 3, 'C2fER', [128]],       # 2 (使用C2fER替代C2f)
            [-1, 1, 'Conv', [256, 3, 2]],  # 3-P3/8
            [-1, 6, 'C2fER', [256]],       # 4 (使用C2fER替代C2f)
            [-1, 1, 'Conv', [512, 3, 2]],  # 5-P4/16
            [-1, 6, 'C2fER', [512]],       # 6 (使用C2fER替代C2f)
            [-1, 1, 'Conv', [1024, 3, 2]], # 7-P5/32
            [-1, 3, 'C2fER', [1024]],      # 8 (使用C2fER替代C2f)
            [-1, 1, 'SPPF', [1024, 5]],   # 9
        ],
        'head': [
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 6], 1, 'Concat', [1]],  # cat backbone P4
            [-1, 3, 'C2fER', [512]],      # 12 (使用C2fER替代C2f)
            
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 4], 1, 'Concat', [1]],  # cat backbone P3
            [-1, 3, 'C2fER', [256]],      # 15 (使用C2fER替代C2f)
            [-1, 1, 'ECA'],               # 添加ECA注意力模块
            
            [-1, 1, 'Conv', [256, 3, 2]],
            [[-1, 12], 1, 'Concat', [1]],  # cat head P4
            [-1, 3, 'C2fER', [512]],       # 19 (使用C2fER替代C2f)
            [-1, 1, 'ECA'],                # 添加ECA注意力模块
            
            [-1, 1, 'Conv', [512, 3, 2]],
            [[-1, 9], 1, 'Concat', [1]],   # cat head P5
            [-1, 3, 'C2fER', [1024]],      # 23 (使用C2fER替代C2f)
            [-1, 1, 'ECA'],                # 添加ECA注意力模块
            
            [[15, 19, 23], 1, 'Detect', ['nc']],  # Detect(P3, P4, P5)
        ]
    }
    
    # 保存配置文件
    with open('yolov8-c2fer.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print("✅ Created custom model configuration: yolov8-c2fer.yaml")

def prepare_dataset():
    """准备或下载数据集"""
    # 这里使用COCO数据集示例，您可以使用自己的数据集
    dataset_dir = 'datasets'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    # 下载数据集配置文件
    coco_yaml = 'https://github.com/ultralytics/ultralytics/raw/main/ultralytics/datasets/coco.yaml'
    os.system(f'wget -P {dataset_dir} {coco_yaml}')
    
    # 修改数据集路径（根据您的实际路径调整）
    with open(f'{dataset_dir}/coco.yaml', 'r') as f:
        data = yaml.safe_load(f)
    
    # 更新路径（这里假设数据集已下载）
    data['path'] = '/root/datasets/coco'
    
    with open(f'{dataset_dir}/coco.yaml', 'w') as f:
        yaml.dump(data, f)
    
    print("✅ Dataset prepared: COCO")

def train_model():
    """训练模型"""
    # 检查GPU是否可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {device}")
    
    # 创建模型
    model = YOLO('yolov8-c2fer.yaml')  # 从自定义配置创建
    
    # 训练参数
    train_args = {
        'data': 'datasets/hkw.yaml',
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'device': device,
        'workers': 8,
        'optimizer': 'auto',  # 自动选择最佳优化器
        'name': 'yolov8_c2fer',
        'exist_ok': True,     # 允许覆盖现有结果
        'patience': 50,       # 早停耐心值
        'lr0': 0.01,          # 初始学习率
        'lrf': 0.01,          # 最终学习率 (lr0 * lrf)
        'weight_decay': 0.0005,
        'save_period': 10,    # 每10个epoch保存一次
        'pretrained': True,    # 使用预训练权重
    }
    
    # 开始训练
    print("⏳ Starting training...")
    results = model.train(**train_args)
    
    # 保存最终模型
    model.save('yolov8_c2fer_final.pt')
    print("💾 Saved final model: yolov8_c2fer_final.pt")
    
    # 评估模型
    metrics = model.val()
    print(f"📊 mAP50-95: {metrics.box.map:.4f}")
    print(f"📊 mAP50: {metrics.box.map50:.4f}")
    
    # 导出模型为ONNX格式
    model.export(format='onnx')
    print("📤 Exported model to ONNX format")

if __name__ == "__main__":
    main()