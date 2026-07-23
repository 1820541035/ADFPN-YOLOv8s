# ADFPN-YOLOv8s：面向低空无人机视角的弱小目标检测模型

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Access-blue)](https://ieeexplore.ieee.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-orange)](https://www.python.org/)

**卢思怡、张静\*、霍建文、谭立国、刘志鹏、戈尧、王川江**

*投稿至 IEEE Access*

> [[English](README.md)] |

---

## 概述

低空无人机视角下的弱小目标检测面临三重核心挑战：(i) 跨层级特征融合过程中弱小目标信息持续衰减；(ii) 复杂背景对前景目标的严重干扰；(iii) 极小目标的浅层细粒度特征表征不足。本文提出 **ADFPN-YOLOv8s**，在 YOLOv8s 基础上引入三项互补创新：

- **ADFPN**（基于注意力的动态加权特征金字塔网络）— 三路动态融合金字塔，将无参数注意力 SimAM 与可学习 Leaky ReLU 归一化权重相结合，实现跨层级特征的择优融合，在增强小目标信号的同时保留多尺度上下文信息。
- **MASP**（多尺度注意力空间金字塔池化模块）— 以 EMA 注意力增强的小尺度池化结构，在抑制复杂背景噪声的同时保留小目标的细粒度细节。
- **P2 高分辨率检测层**— 直接利用浅层高分辨率空间细节，减小最小可检测目标尺寸，弥补深层网络对小目标特征的丢失。

基于 VisDrone2019-DET 的系统场景划分实验表明，三个模块在不同检测场景中发挥互补而非冗余的作用。

---

## 实验对比结果（Δ vs. 基线 YOLOv8s）

以下所有表格中，数值均为**相对于基线 YOLOv8s 的绝对值变化**。正数表示提升，负数表示下降。

### 表一. YOLO 系列模型在 VisDrone2019-DET 上的对比结果

| 模型 | ΔmAP@0.5 | ΔmAP@0.5:0.95 | Δ参数量 (M) | ΔFPS |
|------|:--------:|:-------------:|:-----------:|:----:|
| YOLOv5s | −0.4% | −1.8% | −2.0 | −40.4 |
| YOLOv6s | −2.6% | −2.9% | +5.1 | −32.9 |
| YOLOv8s *(基线)* | — | — | — | — |
| YOLOv9s | −0.7% | −1.9% | −4.0 | −28.1 |
| YOLOv10s | −1.7% | −2.4% | −3.1 | −34.6 |
| YOLOv12s | −2.7% | −3.1% | −1.8 | −11.2 |
| YOLOv13s | −2.9% | −3.4% | −2.1 | −20.4 |
| **ADFPN-YOLOv8s (本文)** | **+3.5%** | **+1.4%** | **−0.1** | −21.7 |

### 表二. 改进模块对比结果（Δ vs. YOLOv8s + PANet）

| 分组 | 模块 | ΔmAP@0.5 | Δ参数量 (M) |
|:---:|------|:--------:|:-----------:|
| (a) 金字塔 | YOLOv8s + BiFPN | −1.2% | +0.1 |
| | **YOLOv8s + ADFPN (本文)** | **+0.9%** | +0.6 |
| (b) 池化 | YOLOv8s + SPP | +0.2% | +0.1 |
| | YOLOv8s + SimSPPF | −1.5% | +0.6 |
| | **YOLOv8s + MASP(4) (本文)** | **+0.6%** | +0.1 |
| (c) 注意力 | YOLOv8s + EMA | −1.0% | +0.03 |
| | YOLOv8s + CBAM | −0.2% | +0.4 |
| | YOLOv8s + CA | −1.3% | +0.1 |

### 表三. 跨数据集泛化实验 — HIT-UAV 与 SeaDronesSee

| 模型 | HIT-UAV ΔmAP@0.5 | HIT-UAV ΔmAP@0.5:0.95 | SeaDronesSee ΔmAP@0.5 | SeaDronesSee ΔmAP@0.5:0.95 |
|------|:----------------:|:---------------------:|:---------------------:|:---------------------------:|
| YOLOv10s | +1.3% | +2.5% | −0.4% | −0.1% |
| YOLOv12s | +0.6% | +1.4% | +8.1% | +2.2% |
| YOLOv13s | +0.8% | +0.8% | +8.5% | +2.6% |
| RT-DETR | −2.6% | −2.3% | +2.4% | +4.3% |
| **ADFPN-YOLOv8s** | **+4.4%** | **+6.6%** | **+10.5%** | **+7.0%** |

### 表四. 极端环境鲁棒性 — VisDrone2019-Dark 与 HazyDet

| 模型 | VisDrone2019-Dark ΔmAP@0.5 | HazyDet ΔmAP@0.5 |
|------|:--------------------------:|:-----------------:|
| YOLOv10s | +1.1% | +3.8% |
| YOLOv12s | −1.5% | −3.3% |
| YOLOv13s | −5.6% | −3.4% |
| RT-DETR | +1.6% | −17.6% |
| **ADFPN-YOLOv8s** | **+2.2%** | **+11.3%** |

### 表五. 嵌入式部署 — Jetson AGX Orin (TensorRT FP16)

| 模型 | ΔmAP@0.5 | ΔFPS | ΔGPU 利用率 |
|------|:--------:|:----:|:-----------:|
| YOLOv8s *(基线)* | — | — | — |
| **ADFPN-YOLOv8s (本文)** | **+5.1%** | −8 | +5% |

*所有模型均在 640x640 输入分辨率下评估，FPS 仅统计推理时间。*

---

## 核心亮点

- **分层动态融合**：ADFPN 权重呈现"深层通用、浅层自适应"的分层特性 — P4/P5 权重跨数据集高度一致（CV < 2%），P2 权重随场景特征自适应调节（CV = 16–30%）。
- **场景互补设计**：在四种场景类型（复杂背景、空旷场景、多尺度、一般城市）上的系统分析验证了三个模块在不同检测瓶颈上的互补分工。
- **轻量实时**：在增强检测能力的同时参数量反低于基线 YOLOv8s，嵌入式 Jetson AGX Orin 上实现实时推理。
- **跨域鲁棒**：在覆盖城市航拍、海洋、红外、低照度、雾霾等六种数据集的评估中验证了泛化能力。

---

## 快速开始

### 环境安装

```bash
git clone https://github.com/yourusername/ADFPN-YOLOv8s.git
cd ADFPN-YOLOv8s
pip install -r requirements.txt
```

### 推理

```python
from models import ADFPN_YOLOv8s
model = ADFPN_YOLOv8s(weights="weights/adfpn_yolov8s_visdrone.pt")
results = model("path/to/image.jpg")
results.show()
```

### Jetson 端 TensorRT 部署

```bash
python tools/export.py --weights adfpn_yolov8s_visdrone.pt --format engine --fp16
python tools/infer_jetson.py --engine adfpn_yolov8s_fp16.engine --source camera
```

---

## 引用

```bibtex
@article{lu2026adfpn,
  title={ADFPN--YOLOv8s: Small Object Detection from Low--Altitude UAV Perspective},
  author={Lu, Siyi and Zhang, Jing and Huo, Jianwen and Tan, Liguo and Liu, Zhipeng and Ge, Yao and Wang, Chuanjiang},
  journal={IEEE Access},
  year={2026},
  doi={10.1109/ACCESS.2026.xxxxx}
}
```

## 联系方式

- **通信作者**：张静 — [zhangjing@swust.edu.cn](mailto:zhangjing@swust.edu.cn)
- **单位**：西南科技大学 信息与控制工程学院，四川绵阳 621010

## 许可证

MIT License.
