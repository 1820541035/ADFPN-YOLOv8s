# ADFPN-YOLOv8s：面向低空无人机视角的弱小目标检测模型

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Access-blue)](https://ieeexplore.ieee.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-orange)](https://www.python.org/)

**卢思怡、张静\*、霍建文、谭立国、刘志鹏、戈尧、王川江**

*投稿至 IEEE Access[[English](README.md)] 

---


## 部署性能

| 平台 | 精度 | mAP@0.5 | FPS | GPU 利用率 |
|------|------|---------|-----|-----------|
| RTX 3060 (桌面端) | FP32 | 44.3% | 114.3 | — |
| Jetson AGX Orin | FP16 | 53.7% | 40 | 68% |
| Jetson AGX Orin | INT8 | 48.9% | 54 | 61% |

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
model.eval()
results = model("path/to/image.jpg")
results.show()
```

### VisDrone2019-DET 训练

```bash
python tools/train.py \
    --model adfpn_yolov8s \
    --data datasets/visdrone.yaml \
    --epochs 100 \
    --batch 16 \
    --img 640
```

### Jetson 端 TensorRT 部署

```bash
python tools/export.py --weights weights/adfpn_yolov8s_visdrone.pt --format engine --fp16
python tools/infer_jetson.py --engine adfpn_yolov8s_fp16.engine --source camera
```

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

本项目基于 MIT License 开源。
