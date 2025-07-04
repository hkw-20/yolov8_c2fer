import os
import yaml
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.tasks import make_divisible
import traceback

# 修复通道数的ERBottleneck
class ERBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c1 = int(c1)  # 确保整数
        self.c2 = int(c2)  # 确保整数
        c_ = max(32, int(c2 * e))  # 确保至少32
        c_ = make_divisible(c_, 8)  # 确保能被8整除
        
        # 确保g是正整数
        g = max(1, int(g)) if g else 1
        
        self.cv1 = Conv(self.c1, c_, 1, 1)
        self.cv2 = Conv(c_, self.c2, 3, 1, g=g)
        self.add = shortcut and self.c1 == self.c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

# 完全重写C2fER以解决通道数问题
class C2fER(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c1 = int(c1)  # 确保整数
        self.c2 = int(c2)  # 确保整数
        n = max(1, int(n))  # 确保至少有一个模块
        
        # 计算隐藏通道数并确保至少为32
        self.c = max(32, int(c2 * max(e, 0.25)))  # 确保最小比例
        self.c = make_divisible(self.c, 8)  # 确保能被8整除
        
        # 修复输入输出通道数，确保与卷积层兼容
        self.cv1 = Conv(self.c1, 2 * self.c, 1, 1)
        
        # 计算拼接后的输入通道数
        input_channels = (2 + n) * self.c
        self.cv2 = Conv(int(input_channels), int(self.c2), 1)  # 确保整数
        
        # 使用ERBottleneck构建模块列表
        self.m = nn.ModuleList([
            ERBottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)
        ])
        
        # 调试信息，显示实际使用的通道数
        print(f"🔧 C2fER initialized: c1={self.c1}, c2={self.c2}, n={n}, shortcut={shortcut}, hidden_c={self.c}")
        print(f"   - cv1: in={self.c1}, out={2 * self.c}")
        print(f"   - cv2: in={input_channels}, out={self.c2}")

    def forward(self, x):
        # 对cv1的输出在通道维度上分块
        y = list(self.cv1(x).split([self.c, self.c], 1))
        
        # 依次通过各个ERBottleneck模块
        for module in self.m:
            y.append(module(y[-1]))
        
        # 拼接所有结果并通过cv2
        return self.cv2(torch.cat(y, 1))

# 动态注册自定义模块到ultralytics
from ultralytics.nn import tasks
tasks.C2f = C2fER  # 使用C2f名称注册
tasks.ERBottleneck = ERBottleneck

def main():
    try:
        print("==================================================")
        print("YOLOv8 Training with C2fER Module (20 classes)")
        print("==================================================")
        
        create_custom_config()
        dataset_path = prepare_dataset()
        train_model(dataset_path)
        
        print("✅ Training completed successfully!")
        print("==================================================")
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        traceback.print_exc()
        print("💡 Possible solutions:")
        print("1. Verify channel calculations in create_custom_config()")
        print("2. Check the model configuration in yolov8-c2fer.yaml")
        print("3. Ensure C2fER and ERBottleneck channel calculations are correct")

def create_custom_config():
    # 明确设置宽度和深度乘数
    width_multiple = 0.50  # 调整此值可改变模型宽度
    depth_multiple = 0.33  # 调整此值可改变模型深度
    
    # 定义基础通道数
    base_channels = {
        0: 64,   # Conv0
        1: 128,  # Conv1
        3: 256,  # Conv3
        5: 512,  # Conv5
        7: 1024, # Conv7
        9: 1024  # SPPF9
    }
    
    # 应用宽度乘数并确保通道数合规
    def calculate_channels(base_ch):
        value = int(base_ch * width_multiple)
        return make_divisible(max(32, value), 8)
    
    # 重新计算各层实际极通道数，确保一致性
    ch0 = calculate_channels(base_channels[0])   # Conv0输出通道
    ch1 = calculate_channels(base_channels[1])   # Conv1输出通道
    ch2 = ch1                                    # C2fER2输出通道
    ch3 = calculate_channels(base_channels[3])   # Conv3输出通道
    ch4 = ch3                                    # C2fER4输出通道
    ch5 = calculate_channels(base_channels[5])   # Conv5输出通道
    ch6 = ch5                                    # C2fER6输出通道
    ch7 = calculate_channels(base_channels[7])   # Conv7输出通道
    ch8 = ch7                                    # C2fER8输出通道
    ch9 = calculate_channels(base_channels[9])   # SPPF9输出通道
    
    # 计算Head部分通道数 - 使用相同的计算方式
    # 注意：这里使用基础通道数而不是缩放后的值
    ch12 = calculate_channels(256)   # C2fER12输出
    ch15 = calculate_channels(128)   # C2fER15输出
    ch16 = ch15                      # Conv16输出
    ch18 = calculate_channels(256)   # C2fER18输出
    ch19 = ch18                      # Conv19输出
    ch21 = calculate_channels(512)   # C2fER21输出
    
    # 计算瓶颈层数量
    n2 = max(1, int(3 * depth_multiple))
    n4 = max(1, int(6 * depth_multiple))
    n6 = max(1, int(6 * depth_multiple))
    n8 = max(1, int(3 * depth_multiple))
    n12 = max(1, int(3 * depth_multiple))
    n15 = max(1, int(3 * depth_multiple))
    n18 = max(1, int(3 * depth_multiple))
    n21 = max(1, int(3 * depth_multiple))
    
    # 构建模型配置 - 关键修复：正确传递所有参数
    config = {
        'nc': 20,
        'depth_multiple': depth_multiple,
        'width_multiple': width_multiple,
        'backbone': [
            [-1, 1, 'Conv', [ch0, 3, 2]],  # 0-P1/2
            [-1, 1, 'Conv', [ch1, 3, 2]],  # 1-P2/4
            [-1, 1, 'C2f', [ch2, n2, True]],  # 2
            [-1, 1, 'Conv', [ch3, 3, 2]],  # 3-P3/8
            [-1, 1, 'C2f', [ch4, n4, True]],  # 4
            [-1, 1, 'Conv', [ch5, 3, 2]],  # 5-P4/16
            [-1, 1, 'C2f', [ch6, n6, True]],  # 6
            [-1, 1, 'Conv', [ch7, 3, 2]],  # 7-P5/32
            [-1, 1, 'C2f', [ch8, n8, True]],  # 8
            [-1, 1, 'SPPF', [ch9, 5]],  # 9
        ],
        'head': [
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],  # 10
            [[-1, 6], 1, 'Concat', [1]],  # 11 cat backbone P4
            [-1, 1, 'C2f', [ch12, n12, False]],  # 12
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],  # 13
            [[-1, 4], 1, 'Concat', [1]],  # 14 cat backbone P3
            [-1, 1, 'C2f', [ch15, n15, False]],  # 15 (P3/8-small)
            [-1, 1, 'Conv', [ch16, 3, 2]],  # 16
            [[-1, 12], 1, 'Concat', [1]],  # 17 cat head P4
            [-1, 1, 'C2f', [ch18, n18, False]],  # 18 (P4/16-medium)
            [-1, 1, 'Conv', [ch19, 3, 2]],  # 19
            [[-1, 9], 1, 'Concat', [1]],  # 20 cat head P5
            [-1, 1, 'C2f', [ch21, n21, False]],  # 21 (P5/32-large)
            [[15, 18, 21], 1, 'Detect', [20]],  # 22 Detect(P3, P4, P5)
        ]
    }
    
    # 保存模型配置到YAML文件
    with open('yolov8-c2fer.yaml', 'w') as f:
        f.write(f"# YOLOv8 C2fER Configuration with {config['nc']} classes\n")
        f.write(f"nc: {config['nc']}\n")
        f.write(f"depth_multiple: {config['depth_multiple']}\n")
        f.write(f"width_multiple: {config['width_multiple']}\n\n")
        f.write("backbone:\n")
        for i, layer in enumerate(config['backbone']):
            f.write(f"  - {layer}  # {i}\n")
        f.write("\nhead:\n")
        for i, layer in enumerate(config['head']):
            f.write(f"  - {layer}  # {10+i}\n")
    
    print("✅ Created custom model configuration: yolov8-c2fer.yaml (20 classes)")

def prepare_dataset():
    dataset_path = '/root/ultralytics-8.3.27/hkw.yaml'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset config not found: {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        data = yaml.safe_load(f)
        names = data['names']
    
    print(f"✅ Using custom dataset: {dataset_path}")
    print(f"   - Classes: {names}")
    return dataset_path

def train_model(dataset_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {device}")
    
    if device == 'cuda':
        # 打印GPU内存信息
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated_mem = torch.cuda.memory_allocated(0) / 1e9
        cached_mem = torch.cuda.memory_reserved(0) / 1e9
        free_mem = total_mem - allocated_mem - cached_mem
        print(f"💻 GPU Memory: Total={total_mem:.2f}GB, Allocated={allocated_mem:.2f}GB, Cached={cached_mem:.2f}GB, Free={free_mem:.2f}GB")
    
    # 确保自定义模块已注册
    from ultralytics.nn import tasks
    tasks.C2f = C2fER
    tasks.ERBottleneck = ERBottleneck
    
    model_path = os.path.abspath('yolov8-c2fer.yaml')
    print(f"🛠️ Creating model from: {model_path}")
    
    # 关键修复：正确加载自定义模型
    try:
        # 创建模型并确保在正确设备上
        model = YOLO(model_path)
        
        # 加载预训练权重（可选）
        pretrained_path = '/root/ultralytics-8.3.27/yolov8n.pt'
        pretrained_exists = os.path.exists(pretrained_path)
        
        if pretrained_exists:
            # 只加载匹配的权重
            model.load(pretrained_path)
            print(f"✅ Loaded pretrained weights from: {pretrained_path}")
        else:
            print("⚠️ Pretrained weights not found, starting from scratch")
        
        # 将模型移动到设备 - 解决设备不匹配问题
        model.model.to(device)
        print(f"✅ Model moved to {device}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        traceback.print_exc()
        return
    
    # 验证模型结构 - 使用更小的输入尺寸
    print("🔍 Verifying model structure with sample input...")
    try:
        # 创建样本并确保在模型所在设备上
        sample = torch.randn(1, 3, 160, 160).to(next(model.model.parameters()).device)
        
        # 直接获取模型输出
        with torch.no_grad():
            output = model.model(sample)
            
            # 处理不同类型的输出
            if isinstance(output, (list, tuple)):
                shapes = [o.shape for o in output]
                print(f"✅ Model structure verified. Output shapes: {shapes}")
            else:
                print(f"✅ Model structure verified. Output shape: {output.shape}")
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        traceback.print_exc()
        print("💡 Try solutions:")
        print("1. Check channel calculations in C2fER and ERBottleneck")
        print("2. Verify the model configuration in yolov8-c2fer.yaml")
        print("3. Reduce model complexity by lowering width_multiple")
        return
    
    # 训练参数配置（降低显存消耗）
    train_args = {
        'data': dataset_path,
        'epochs': 40,
        'imgsz': 320,  # 降低分辨率以节省显存
        'batch': 8,     # 减小批量大小
        'device': device,
        'workers': 4,   # 减少工作进程
        'name': 'yolov8_c2fer_20cls',
        'exist_ok': True,
        'patience': 50,
        'lr0': 0.01,
        'lrf': 0.01,
        'weight_decay': 0.0005,
        'save_period': 10,
        'pretrained': pretrained_exists,  # 根据权重是否存在设置
    }
    
    print("⏳ Starting training...")
    try:
        model.train(**train_args)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        traceback.print_exc()
        return
    
    # 保存最终模型
    model.save('yolov8_c2fer_final_20cls.pt')
    print("💾 Saved final model: yolov8_c2fer_final_20cls.pt")
    
    # 评估模型性能
    try:
        metrics = model.val()
        print(f"📊 mAP50-95: {metrics.box.map:.4f}")
        print(f"📊 mAP50: {metrics.box.map50:.4f}")
    except Exception as e:
        print(f"⚠️ Validation failed: {e}")
    
    # 保存训练结果
    save_training_results()

def save_training_results():
    import shutil
    results_dir = os.path.join('runs', 'detect', 'yolov8_c2fer_20cls')
    os.makedirs(results_dir, exist_ok=True)
    
    # 复制模型和配置文件到结果目录
    for file in ['yolov8_c2fer_final_20cls.pt', 'yolov8-c2fer.yaml']:
        if os.path.exists(file):
            shutil.copy(file, results_dir)
            print(f"📋 Copied {file} to {results_dir}")
        else:
            print(f"⚠️ File not found: {file}")
    
    print(f"✅ Training results saved to {results_dir}")

if __name__ == "__main__":
    main()