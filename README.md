# AdversarialToolbox  
**对抗攻击生成与模型鲁棒性评估框架**  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)

一站式解决方案，支持快速生成对抗样本、评估模型脆弱性，并提供可视化分析工具。

## 🚀 项目背景  
深度神经网络的对抗鲁棒性是AI安全的核心问题。现有工具库常面临攻击方法实现不一致、评估流程碎片化等问题。AdversarialToolbox 通过标准化接口和自动化流程，帮助研究者和开发者：  
- 🔍 **快速复现**论文中的攻击方法  
- ⚡ **高效评估**模型在实际对抗场景中的表现  
- 🧩 **无缝扩展**自定义攻击算法与防御策略  

## ✨ 功能特性  
| 模块         | 功能描述                                                                 |  
|--------------|--------------------------------------------------------------------------|  
| **攻击生成** | 支持FGSM/PGD/CW/LBAP等12+方法，提供定向/非定向攻击模式                   |  
| **评估分析** | 自动化计算ASR、置信度分布，生成L2/L∞扰动分析报告                         |  
| **可视化**   | 交互式对比原始样本与对抗样本，支持热力图显示关键扰动区域                 |  
| **部署支持** | 提供ONNX/TensorRT导出接口，支持边缘设备对抗测试                          |  

## 🛠 快速开始  
### 安装依赖
```bash
pip install -r requirements.txt
```
### 生成对抗样本（以LBAP攻击为例）
```python
import torch
from model_zoo import ModelZoo
from dataset_zoo import DatasetZoo
from attacks.lbap import LBAP

### 初始化模型与数据
model = ModelZoo().load('resnet50').cuda().eval()
dataset = DatasetZoo().load('imagenet_val', path='data/imagenet')

### 配置LBAP攻击参数
attack = LBAP(
    model, 
    eps=16/255, 
    steps=50,
    decay=0.9,
    n=10,               # 多扰动路径数量
    random_mixup_num=6  # 随机混合样本数
)

### 对单张图像生成对抗样本
image, label = dataset[0]
image = image.unsqueeze(0).cuda()
target_label = 123  # 假设目标类别为123（需根据实际攻击目标设置）

adv_image = attack(image, target_label)

### 保存结果
save_image(adv_image, 'results/lbap_adv.png')

### 在ImageNet验证集上测试ResNet50对LBAP攻击的鲁棒性
python evaluate_attack.py \
  --model resnet50 \
  --attack lbap \
  --dataset imagenet_val \
  --eps 16 \
  --batch_size 32 \
  --output_dir ./results
```

---


## 🔫 支持的攻击方法  
| 方法名称       | 类型     | 定向攻击 | 关键参数 | 配置文件示例 |  
|----------------|----------|----------|----------|--------------|  
| **LBAP**       | 基于混合 | ✔️       | `eps=16/255`, `steps=50`, `n=10` | [lbap.yaml](configs/lbap.yaml) |  
| **DI-FGSM**    | 迭代优化 | ✔️       | `eps=8/255`, `steps=20`, `diversity_prob=0.7` | [difgsm.yaml](configs/difgsm.yaml) |  
| **TI-FGSM**    | 平移不变 | ✔️       | `kernel_size=5`, `sigma=3.0` | [tifgsm.yaml](configs/tifgsm.yaml) |  
