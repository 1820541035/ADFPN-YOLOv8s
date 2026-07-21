# ADFPN-YOLOv8s: Small Object Detection from Low-Altitude UAV Perspective

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Access-blue)](https://ieeexplore.ieee.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-orange)](https://www.python.org/)

**Siyi Lu, Jing Zhang\*, Jianwen Huo, Liguo Tan, Zhipeng Liu, Yao Ge, Chuanjiang Wang**

*Submitted to IEEE Access 

---


## Environment and Deployment

| Platform | Precision | mAP@0.5 | FPS | GPU Util. |
|----------|-----------|---------|-----|-----------|
| RTX 3060 (Desktop) | FP32 | 44.3% | 114.3 | -- |
| Jetson AGX Orin | FP16 | 53.7% | 40 | 68% |
| Jetson AGX Orin | INT8 | 48.9% | 54 | 61% |

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
model.eval()
results = model("path/to/image.jpg")
results.show()
```

### Training on VisDrone2019-DET

```bash
python tools/train.py \
    --model adfpn_yolov8s \
    --data datasets/visdrone.yaml \
    --epochs 100 \
    --batch 16 \
    --img 640
```

### TensorRT Deployment on Jetson

```bash
python tools/export.py --weights weights/adfpn_yolov8s_visdrone.pt --format engine --fp16
python tools/infer_jetson.py --engine adfpn_yolov8s_fp16.engine --source camera
```

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

- **Corresponding Author**: Jing Zhang -- [zhangjing@swust.edu.cn](mailto:zhangjing@swust.edu.cn)
- **Institution**: School of Information and Control Engineering, Southwest University of Science and Technology, Mianyang, Sichuan 621010, China

## License

This project is licensed under the MIT License.
