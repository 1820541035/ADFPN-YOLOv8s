# ADFPN-YOLOv8s: Small Object Detection from Low-Altitude UAV Perspective

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Access-blue)](https://ieeexplore.ieee.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-orange)](https://www.python.org/)

**Siyi Lu, Jing Zhang\*, Jianwen Huo, Liguo Tan, Zhipeng Liu, Yao Ge, Chuanjiang Wang**

*Submitted to IEEE Access*

> [[中文文档](README_CN.md)] | 

---

## Overview

Small object detection from low-altitude UAV imagery faces three fundamental challenges: (i) progressive loss of weak features during cross-level fusion, (ii) severe interference from cluttered backgrounds, and (iii) insufficient fine-grained representations for tiny objects. We propose **ADFPN-YOLOv8s**, an improved detection framework built upon YOLOv8s, featuring three complementary innovations:

- **ADFPN** (Attention-based Dynamic Feature Pyramid Network) — A three-branch dynamic fusion pyramid that combines SimAM parameter-free attention with learnable Leaky ReLU-normalized weights, enabling selective cross-level fusion that amplifies small-object signals while preserving multi-scale context.
- **MASP** (Multi-scale Attention Spatial Pyramid Pooling) — A small-kernel pooling structure enhanced by EMA attention, designed to suppress cluttered backgrounds while retaining fine-grained contextual details.
- **P2 Detection Layer** — A high-resolution detection branch that directly exploits shallow spatial details, reducing the minimum detectable object size and compensating for the loss of extremely small object features in deep networks.

The three modules play complementary rather than redundant roles across different detection scenarios, as demonstrated by systematic scene-partition experiments on VisDrone2019-DET.

---

## Comparison Results (Δ vs. Baseline YOLOv8s)

All values are reported as **absolute differences from the baseline YOLOv8s**. Positive values indicate improvement; negative values indicate degradation.

### TABLE I. Comparison of YOLO Series Models on VisDrone2019-DET

| Model | ΔmAP@0.5 | ΔmAP@0.5:0.95 | ΔParams (M) | ΔFPS |
|-------|:--------:|:-------------:|:-----------:|:----:|
| YOLOv5s | −0.4% | −1.8% | −2.0 | −40.4 |
| YOLOv6s | −2.6% | −2.9% | +5.1 | −32.9 |
| YOLOv8s *(baseline)* | — | — | — | — |
| YOLOv9s | −0.7% | −1.9% | −4.0 | −28.1 |
| YOLOv10s | −1.7% | −2.4% | −3.1 | −34.6 |
| YOLOv12s | −2.7% | −3.1% | −1.8 | −11.2 |
| YOLOv13s | −2.9% | −3.4% | −2.1 | −20.4 |
| **ADFPN-YOLOv8s (Ours)** | **+3.5%** | **+1.4%** | **−0.1** | −21.7 |

### TABLE II. Comparison of Improved Modules (Δ vs. YOLOv8s + PANet)

| Group | Module | ΔmAP@0.5 | ΔParams (M) |
|:-----:|--------|:--------:|:-----------:|
| (a) Pyramid | YOLOv8s + BiFPN | −1.2% | +0.1 |
| | **YOLOv8s + ADFPN (Ours)** | **+0.9%** | +0.6 |
| (b) Pooling | YOLOv8s + SPP | +0.2% | +0.1 |
| | YOLOv8s + SimSPPF | −1.5% | +0.6 |
| | **YOLOv8s + MASP(4) (Ours)** | **+0.6%** | +0.1 |
| (c) Attention | YOLOv8s + EMA | −1.0% | +0.03 |
| | YOLOv8s + CBAM | −0.2% | +0.4 |
| | YOLOv8s + CA | −1.3% | +0.1 |

### TABLE III. Cross-Dataset Generalization — HIT-UAV & SeaDronesSee

| Model | HIT-UAV ΔmAP@0.5 | HIT-UAV ΔmAP@0.5:0.95 | SeaDronesSee ΔmAP@0.5 | SeaDronesSee ΔmAP@0.5:0.95 |
|-------|:----------------:|:---------------------:|:---------------------:|:---------------------------:|
| YOLOv10s | +1.3% | +2.5% | −0.4% | −0.1% |
| YOLOv12s | +0.6% | +1.4% | +8.1% | +2.2% |
| YOLOv13s | +0.8% | +0.8% | +8.5% | +2.6% |
| RT-DETR | −2.6% | −2.3% | +2.4% | +4.3% |
| **ADFPN-YOLOv8s** | **+4.4%** | **+6.6%** | **+10.5%** | **+7.0%** |

### TABLE IV. Robustness Under Extreme Conditions — VisDrone2019-Dark & HazyDet

| Model | VisDrone2019-Dark ΔmAP@0.5 | HazyDet ΔmAP@0.5 |
|-------|:--------------------------:|:-----------------:|
| YOLOv10s | +1.1% | +3.8% |
| YOLOv12s | −1.5% | −3.3% |
| YOLOv13s | −5.6% | −3.4% |
| RT-DETR | +1.6% | −17.6% |
| **ADFPN-YOLOv8s** | **+2.2%** | **+11.3%** |

### TABLE V. Deployment — Jetson AGX Orin (TensorRT FP16)

| Model | ΔmAP@0.5 | ΔFPS | ΔGPU Utilization |
|-------|:--------:|:----:|:-----------------:|
| YOLOv8s *(baseline)* | — | — | — |
| **ADFPN-YOLOv8s (Ours)** | **+5.1%** | −8 | +5% |

*All models evaluated at 640x640 input resolution. FPS includes inference time only.*

---

## Key Features

- **Hierarchical dynamic fusion**: ADFPN weights exhibit "deep universality, shallow adaptivity" across scenes — P4/P5 weights converge near-identically across datasets (CV < 2%), while P2 weights adapt to scene characteristics (CV = 16–30%).
- **Scene-complementary design**: Systematic analysis across four scene types (complex background, open scene, multi-scale, general urban) proves the three modules address distinct detection bottlenecks.
- **Lightweight and real-time**: Despite architectural enhancements, the model maintains fewer parameters than the baseline YOLOv8s and achieves real-time inference on embedded Jetson AGX Orin.
- **Cross-domain robustness**: Validated across six datasets spanning urban, maritime, infrared, low-light, and hazy conditions.

---

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/ADFPN-YOLOv8s.git
cd ADFPN-YOLOv8s
pip install -r requirements.txt
```

### Inference

```python
from models import ADFPN_YOLOv8s
model = ADFPN_YOLOv8s(weights="weights/adfpn_yolov8s_visdrone.pt")
results = model("path/to/image.jpg")
results.show()
```

### TensorRT Deployment (Jetson)

```bash
python tools/export.py --weights adfpn_yolov8s_visdrone.pt --format engine --fp16
python tools/infer_jetson.py --engine adfpn_yolov8s_fp16.engine --source camera
```

---

## Citation

```bibtex
@article{lu2026adfpn,
  title={ADFPN--YOLOv8s: Small Object Detection from Low--Altitude UAV Perspective},
  author={Lu, Siyi and Zhang, Jing and Huo, Jianwen and Tan, Liguo and Liu, Zhipeng and Ge, Yao and Wang, Chuanjiang},
  journal={IEEE Access},
  year={2026},
  doi={10.1109/ACCESS.2026.xxxxx}
}
```

## Contact

- **Corresponding Author**: Jing Zhang — [zhangjing@swust.edu.cn](mailto:zhangjing@swust.edu.cn)
- **Institution**: School of Information and Control Engineering, Southwest University of Science and Technology, Mianyang, Sichuan 621010, China

## License

MIT License.
