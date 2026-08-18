#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VisDrone2019-DET 子集制作脚本

按目标级条件从 VisDrone2019-DET 训练集筛选出三个子集:
  tiny         : 目标框面积 < 32×32 像素
  occluded     : 遮挡等级 ≥ 1 (部分遮挡或严重遮挡)
  longdistance : 面积 < 32×32 且 目标中心 y_c < H/2 (图像上半部分)

标注格式 (每行 8 列):
  bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion
  - score=0 或 category=0 的行是"忽略区域", 不属于目标, 一律剔除
  - occlusion: 0=无遮挡, 1=部分遮挡, 2=严重遮挡

用法示例:
  python make_subsets.py                     # 全部三个子集, 复制图片到 subsets/
  python make_subsets.py --dry-run           # 只统计各子集规模, 不写任何文件
  python make_subsets.py --subsets tiny      # 只做 tiny 子集
  python make_subsets.py --mode symlink      # 用符号链接节省磁盘 (失败自动回退为复制)
  python make_subsets.py --mode list         # 只生成图片清单 + 过滤后的标注, 不复制图片
  python make_subsets.py --annot full        # 标注保留全部目标, 仅按"图内是否含匹配目标"筛图
"""

import argparse
import os
import shutil
import statistics
import sys
from collections import Counter
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("缺少依赖 Pillow，请先执行: pip install pillow")

# ---------- 子集条件常量 ----------
TINY_AREA = 32 * 32      # 32×32 = 1024 像素
HORIZON_FRAC = 0.5       # 图像中线 y = H/2

CATEGORY_NAMES = {
    1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van",
    6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus",
    10: "motor", 11: "others",
}

# 子集名 -> (说明, 判定函数(目标, 图像高))
def _is_tiny(obj, h):
    return obj["w"] * obj["h"] < TINY_AREA


def _is_occluded(obj, h):
    return obj["occ"] >= 1


def _is_longdistance(obj, h):
    yc = obj["y"] + obj["h"] / 2.0
    return _is_tiny(obj, h) and yc < h * HORIZON_FRAC


SUBSETS = {
    "tiny":         ("面积 < 32×32",              _is_tiny),
    "occluded":     ("遮挡等级 ≥ 1",               _is_occluded),
    "longdistance": ("面积 < 32×32 且 yc < H/2",  _is_longdistance),
}


# ---------- 读取与判定 ----------
def parse_annotation_file(path):
    """解析标注文件, 返回 (有效目标列表, 无效行数)。剔除忽略区域行。"""
    objs, bad = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split(",")
        if len(parts) < 8:
            bad += 1
            continue
        try:
            x, y, w, h, score, cat, trunc, occ = (int(float(p)) for p in parts[:8])
        except ValueError:
            bad += 1
            continue
        if score > 0 and cat != 0 and w > 0 and h > 0:
            # 忽略区域 (score=0 / category=0) 与退化框 (宽/高为 0) 不算目标
            objs.append({"x": x, "y": y, "w": w, "h": h,
                         "score": score, "cat": cat, "trunc": trunc, "occ": occ})
    return objs, bad


def image_size(path, cache):
    """读取图片真实尺寸 (W, H), 按文件名缓存。只读文件头, 速度很快。"""
    stem = path.stem
    if stem not in cache:
        with Image.open(path) as im:
            cache[stem] = im.size  # (W, H)
    return cache[stem]


def fmt(obj):
    return f"{obj['x']},{obj['y']},{obj['w']},{obj['h']}," \
           f"{obj['score']},{obj['cat']},{obj['trunc']},{obj['occ']}"


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


# ---------- 统计报告 ----------
def build_report(selected, stats, union, args, total_files, total_objs,
                 bad_lines, no_obj_files, missing_img):
    occ_names = {0: "无遮挡", 1: "部分遮挡", 2: "严重遮挡"}
    lines = ["=" * 60, "子集筛选统计报告", "=" * 60,
             f"标注文件总数: {total_files}  有效目标总数: {total_objs}",
             f"无目标的标注文件: {no_obj_files}  缺图片跳过: {missing_img}  "
             f"无效标注行: {bad_lines}",
             f"标注保留模式: {'只保留匹配目标' if args.annot == 'filter' else '保留全部有效目标'}",
             f"子集并集覆盖图片数: {len(union)}"]
    for name in selected:
        s = stats[name]
        lines += ["", "-" * 60,
                  f"[{name}]  {SUBSETS[name][0]}",
                  f"  图片数: {s['images']}   匹配目标数: {s['boxes']}"]
        if s["areas"]:
            lines.append(
                f"  目标面积: min={min(s['areas'])}, median={statistics.median(s['areas']):.0f}, "
                f"max={max(s['areas'])}")
        cats = ", ".join(f"{CATEGORY_NAMES.get(c, c)}:{n}"
                         for c, n in sorted(s["cats"].items()))
        lines.append(f"  类别分布: {cats}")
        if name == "occluded":
            occs = ", ".join(f"{occ_names.get(o, o)}:{n}"
                             for o, n in sorted(s["occs"].items()))
            lines.append(f"  遮挡分布: {occs}")
    lines.append("=" * 60)
    return "\n".join(lines)


# ---------- 主流程 ----------
def main():
    if hasattr(sys.stdout, "reconfigure"):  # Windows GBK 控制台 -> UTF-8
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="VisDrone2019-DET 子集制作脚本",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--root", default=str(Path(__file__).parent),
                    help="数据集根目录 (含 annotations/ 与 images/)")
    ap.add_argument("--ann-dir", default=None, help="标注目录 (默认 root/annotations)")
    ap.add_argument("--img-dir", default=None, help="图片目录 (默认 root/images)")
    ap.add_argument("--out", default=None, help="输出目录 (默认 root/subsets)")
    ap.add_argument("--subsets", nargs="+", choices=list(SUBSETS), default=list(SUBSETS),
                    help="要生成的子集, 默认全部三个")
    ap.add_argument("--mode", choices=["copy", "symlink", "list"], default="copy",
                    help="copy=复制图片; symlink=符号链接; list=只生成清单不复制图片")
    ap.add_argument("--annot", choices=["filter", "full"], default="filter",
                    help="filter=标注只保留匹配目标; full=标注保留全部目标, 仅筛图")
    ap.add_argument("--dry-run", action="store_true", help="只统计规模, 不写任何文件")
    args = ap.parse_args()

    root = Path(args.root)
    ann_dir = Path(args.ann_dir) if args.ann_dir else root / "annotations"
    img_dir = Path(args.img_dir) if args.img_dir else root / "images"
    out_root = Path(args.out) if args.out else root / "subsets"

    if not ann_dir.is_dir():
        sys.exit(f"标注目录不存在: {ann_dir}")
    if not img_dir.is_dir():
        sys.exit(f"图片目录不存在: {img_dir}")

    selected = [s for s in ["tiny", "occluded", "longdistance"] if s in args.subsets]
    print(f"标注目录: {ann_dir}")
    print(f"图片目录: {img_dir}")
    print(f"输出目录: {out_root}  模式: {args.mode}  标注: {args.annot}  "
          f"{'(dry-run)' if args.dry_run else ''}")

    size_cache = {}
    stats = {n: {"images": 0, "boxes": 0, "cats": Counter(),
                 "areas": [], "occs": Counter()} for n in selected}
    list_stems = {n: [] for n in selected}
    union = set()
    missing_img = no_obj_files = bad_lines = total_objs = 0

    ann_files = sorted(ann_dir.glob("*.txt"))
    total_files = len(ann_files)
    print(f"共 {total_files} 个标注文件, 开始扫描...")

    for i, ann_path in enumerate(ann_files, 1):
        stem = ann_path.stem
        img_path = img_dir / (stem + ".jpg")
        if not img_path.exists():
            missing_img += 1
            continue
        _, h = image_size(img_path, size_cache)
        objs, bad = parse_annotation_file(ann_path)
        bad_lines += bad
        total_objs += len(objs)
        if not objs:
            no_obj_files += 1
            continue

        for name in selected:
            keep = [o for o in objs if SUBSETS[name][1](o, h)]
            if not keep:
                continue
            s = stats[name]
            s["images"] += 1
            s["boxes"] += len(keep)
            for o in keep:
                s["cats"][o["cat"]] += 1
                s["areas"].append(o["w"] * o["h"])
                s["occs"][o["occ"]] += 1
            union.add(stem)

            if not args.dry_run:
                out_objs = objs if args.annot == "full" else keep
                (out_root / name / "annotations").mkdir(parents=True, exist_ok=True)
                (out_root / name / "annotations" / (stem + ".txt")).write_text(
                    "\n".join(fmt(o) for o in out_objs) + "\n", encoding="utf-8")
                if args.mode in ("copy", "symlink"):
                    (out_root / name / "images").mkdir(parents=True, exist_ok=True)
                    copy_image(img_path, out_root / name / "images", args.mode)
                else:
                    list_stems[name].append(stem)

        if i % 1000 == 0:
            print(f"  已扫描 {i}/{total_files} ...")

    if not args.dry_run and args.mode == "list":
        for name in selected:
            (out_root / name).mkdir(parents=True, exist_ok=True)
            (out_root / name / "list.txt").write_text(
                "\n".join(sorted(list_stems[name])) + "\n", encoding="utf-8")

    report = build_report(selected, stats, union, args, total_files, total_objs,
                          bad_lines, no_obj_files, missing_img)
    print("\n" + report)
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "report.txt").write_text(report + "\n", encoding="utf-8")
        print(f"\n完成。输出目录: {out_root}")


if __name__ == "__main__":
    main()
