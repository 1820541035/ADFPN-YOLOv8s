#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VisDrone2019-DET 图像级子集制作脚本

按图像级指标筛选两个子集 (与 make_subsets.py 的目标级子集互补):
  lowlight  : 按全局平均灰度 μ_gray 升序排序, 取亮度最低的 20% 图像
  cluttered : 按每图实例数 (有效目标数) 降序排序, 取实例最多的 25% 图像

图像级子集的标注文件原样保留, 不做目标过滤; 统计报告会给出每个子集的
实例总数、平均每图实例数与类别分布。

平均灰度 μ_gray: 将图像转为灰度图 (ITU-R 601-2 luma) 后取全图像素均值;
有 numpy 时用 numpy 精确计算, 否则用 PIL ImageStat (结果一致)。
实例数: 标注中有效目标行数, 规则同 make_subsets.py (剔除 score=0/category=0
的忽略区域与宽高为 0 的退化框)。

边界并列时按文件名排序, 保证结果确定可复现; 每个子集取 ceil(N × pct/100) 张。

用法示例:
  python make_image_subsets.py                           # 两个子集, 复制图片
  python make_image_subsets.py --dry-run                 # 只计算指标并打印报告
  python make_image_subsets.py --subsets lowlight        # 只做低光照子集
  python make_image_subsets.py --pct-lowlight 20 --pct-cluttered 25
  python make_image_subsets.py --mode symlink            # 符号链接省磁盘
  python make_image_subsets.py --mode list               # 只生成图片清单
"""

import argparse
import csv
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

CATEGORY_NAMES = {
    1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van",
    6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus",
    10: "motor", 11: "others",
}

# ---------- 平均灰度计算 (numpy 优先, PIL 兜底) ----------
try:
    import numpy as np

    def mean_gray(img_path):
        with Image.open(img_path) as im:
            return float(np.asarray(im.convert("L"), dtype=np.float64).mean())

except ImportError:
    from PIL import ImageStat

    def mean_gray(img_path):
        with Image.open(img_path) as im:
            return float(ImageStat.Stat(im.convert("L")).mean[0])


# ---------- 实例数统计 ----------
def count_instances(ann_path):
    """统计标注文件中有效目标, 返回 (数量, 类别计数)。

    剔除忽略区域与退化框, 规则同 make_subsets.py。
    """
    n, cats = 0, Counter()
    for line in ann_path.read_text(encoding="utf-8").splitlines():
        parts = line.split(",")
        if len(parts) < 8:
            continue
        try:
            x, y, w, h, score, cat = (int(float(p)) for p in parts[:6])
        except ValueError:
            continue
        if score > 0 and cat != 0 and w > 0 and h > 0:
            n += 1
            cats[cat] += 1
    return n, cats


# ---------- 选取与拷贝 ----------
def select(rows, key_fn, reverse, pct):
    """rows: (stem, mean_gray, instances); 按指标排序后取前 pct% (ceil)。"""
    ordered = sorted(rows, key=lambda r: (key_fn(r), r[0]), reverse=reverse)
    k = math.ceil(len(ordered) * pct / 100)
    return ordered[:k]


def copy_image(src, dst_dir, mode):
    dst = dst_dir / src.name
    if dst.exists():
        return
    if mode == "symlink":
        try:
            os.symlink(src.resolve(), dst)
        except OSError:  # Windows 无权限创建符号链接时回退为复制
            shutil.copyfile(src, dst)
    else:
        shutil.copyfile(src, dst)


# ---------- 报告 ----------
def build_report(rows, selected, args, missing_img):
    low, clut = selected["lowlight"], selected["cluttered"]
    lines = ["=" * 60, "图像级子集筛选统计报告", "=" * 60,
             f"有效图像总数: {len(rows)}  缺图片跳过: {missing_img}",
             f"输出模式: {args.mode}"]
    if low:
        grays = [r[1] for r in low]
        total = sum(r[2] for r in low)
        cats = sum((r[3] for r in low), Counter())
        lines += ["", "-" * 60,
                  f"[lowlight]  全局平均灰度 μ_gray 最低的 {args.pct_lowlight}%",
                  f"  图片数: {len(low)}  入选阈值: μ_gray ≤ {max(grays):.2f}",
                  f"  μ_gray 范围: [{min(grays):.2f}, {max(grays):.2f}]",
                  f"  实例总数: {total}  平均每图: {total / len(low):.1f}",
                  f"  类别分布: {format_cats(cats)}"]
    if clut:
        counts = [r[2] for r in clut]
        total = sum(counts)
        cats = sum((r[3] for r in clut), Counter())
        lines += ["", "-" * 60,
                  f"[cluttered]  每图实例数最多的 {args.pct_cluttered}%",
                  f"  图片数: {len(clut)}  入选阈值: 实例数 ≥ {min(counts)}",
                  f"  实例数范围: [{min(counts)}, {max(counts)}]",
                  f"  实例总数: {total}  平均每图: {total / len(clut):.1f}",
                  f"  类别分布: {format_cats(cats)}"]
    if low and clut:
        overlap = len(set(r[0] for r in low) & set(r[0] for r in clut))
        lines += ["", f"两子集重叠图片数: {overlap}"]
    lines.append("=" * 60)
    return "\n".join(lines)


def format_cats(cats):
    return ", ".join(f"{CATEGORY_NAMES.get(c, c)}:{n}"
                     for c, n in sorted(cats.items()))


# ---------- 主流程 ----------
def main():
    if hasattr(sys.stdout, "reconfigure"):  # Windows GBK 控制台 -> UTF-8
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="VisDrone2019-DET 图像级子集制作脚本",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--root", default=str(Path(__file__).parent),
                    help="数据集根目录 (含 annotations/ 与 images/)")
    ap.add_argument("--ann-dir", default=None, help="标注目录 (默认 root/annotations)")
    ap.add_argument("--img-dir", default=None, help="图片目录 (默认 root/images)")
    ap.add_argument("--out", default=None, help="输出目录 (默认 root/image_subsets)")
    ap.add_argument("--subsets", nargs="+", choices=["lowlight", "cluttered"],
                    default=["lowlight", "cluttered"], help="要生成的子集")
    ap.add_argument("--pct-lowlight", type=float, default=20.0,
                    help="低光照子集取最暗的百分之多少 (默认 20)")
    ap.add_argument("--pct-cluttered", type=float, default=25.0,
                    help="杂乱背景子集取实例最多的百分之多少 (默认 25)")
    ap.add_argument("--mode", choices=["copy", "symlink", "list"], default="copy",
                    help="copy=复制图片; symlink=符号链接; list=只生成清单不复制图片")
    ap.add_argument("--dry-run", action="store_true", help="只计算指标并打印报告, 不写文件")
    args = ap.parse_args()

    root = Path(args.root)
    ann_dir = Path(args.ann_dir) if args.ann_dir else root / "annotations"
    img_dir = Path(args.img_dir) if args.img_dir else root / "images"
    out_root = Path(args.out) if args.out else root / "image_subsets"

    if not ann_dir.is_dir():
        sys.exit(f"标注目录不存在: {ann_dir}")
    if not img_dir.is_dir():
        sys.exit(f"图片目录不存在: {img_dir}")

    print(f"标注目录: {ann_dir}")
    print(f"图片目录: {img_dir}")
    print(f"输出目录: {out_root}  模式: {args.mode}  "
          f"{'(dry-run)' if args.dry_run else ''}")

    # 1) 逐图计算指标: (stem, mean_gray, instances, 类别计数)
    rows, missing_img = [], 0
    ann_files = sorted(ann_dir.glob("*.txt"))
    print(f"共 {len(ann_files)} 张图, 计算平均灰度与实例数...")
    for i, ann_path in enumerate(ann_files, 1):
        stem = ann_path.stem
        img_path = img_dir / (stem + ".jpg")
        if not img_path.exists():
            missing_img += 1
            continue
        n, cats = count_instances(ann_path)
        rows.append((stem, mean_gray(img_path), n, cats))
        if i % 500 == 0:
            print(f"  已处理 {i}/{len(ann_files)} ...")

    # 2) 按指标排序取百分比
    selected = {}
    if "lowlight" in args.subsets:
        selected["lowlight"] = select(rows, key_fn=lambda r: r[1], reverse=False,
                                      pct=args.pct_lowlight)
    if "cluttered" in args.subsets:
        selected["cluttered"] = select(rows, key_fn=lambda r: r[2], reverse=True,
                                       pct=args.pct_cluttered)

    # 3) 输出
    for name, picked in selected.items():
        if args.dry_run:
            continue
        (out_root / name / "annotations").mkdir(parents=True, exist_ok=True)
        if args.mode in ("copy", "symlink"):
            (out_root / name / "images").mkdir(parents=True, exist_ok=True)
        stems = []
        for stem, _, _, _ in picked:
            shutil.copyfile(ann_dir / (stem + ".txt"),
                            out_root / name / "annotations" / (stem + ".txt"))
            if args.mode in ("copy", "symlink"):
                copy_image(img_dir / (stem + ".jpg"),
                           out_root / name / "images", args.mode)
            else:
                stems.append(stem)
        if args.mode == "list":
            (out_root / name / "list.txt").write_text(
                "\n".join(sorted(stems)) + "\n", encoding="utf-8")

    report = build_report(rows, selected, args, missing_img)
    print("\n" + report)

    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "report.txt").write_text(report + "\n", encoding="utf-8")
        with (out_root / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "mean_gray", "instances"])
            w.writerows(rows)
        print(f"\n完成。输出目录: {out_root}")


if __name__ == "__main__":
    main()
