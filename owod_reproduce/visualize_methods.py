"""
OWOD Method Comparison Visualization
=====================================
3개의 랜덤 샘플 이미지에 대해 두 가지 unknown detection 방법을 시각화.

Layout:
  Row 0 : Method 1 — "object" token
  Row 1 : Method 2 — P_u formula
  Col 0-2: 3 random images (fixed seed)

Box color:
  Blue  : known prediction
  Red   : unknown prediction
  (텍스트: class name + confidence)

Usage:
  conda activate py310
  cd /Users/jisu/Desktop/dev/cli/YOLO-World
  python owod_reproduce/visualize_methods.py            # Task 1, seed=42, unk_thr=0.25
  python owod_reproduce/visualize_methods.py --task 2 --seed 7
  python owod_reproduce/visualize_methods.py --outfile result.png
"""

import sys
import warnings
import argparse
warnings.filterwarnings("ignore")

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "mmyolo"))

import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.utils import get_test_pipeline_cfg

# ── Re-use helpers from owod_eval_pu ─────────────────────────────────────────
from owod_reproduce.owod_eval_pu import (
    TASK_SPLITS, OBJECTIVENESS_PROMPTS, CLASS_ALIAS,
    patch_head_for_full_scores, compute_pu, build_test_pipeline,
    resolve_image_path, select_quickeval_ids,
)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT   = Path("/Users/jisu/Desktop/dev/cli/OW_OVD/reproduce_owod/data")
CHECKPOINT  = str(Path("/Users/jisu/Desktop/dev/cli/OW_OVD/reproduce_owod") /
                  "yolo_world_l_stage2-b3e3dc3f.pth")
CONFIG_FILE = str(ROOT / "configs/pretrain" /
                  "yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py")
IMAGE_LIST  = DATA_ROOT / "owod_all_task_test_nodup.txt"


# ── Inference helpers ─────────────────────────────────────────────────────────

def infer_raw(model, pipeline, img_path, texts, device):
    """Return raw InstanceData (post-NMS, rescaled)."""
    data_info = dict(img_id=0, img_path=str(img_path), texts=texts)
    data_info = pipeline(data_info)
    batch = dict(inputs=data_info["inputs"].unsqueeze(0),
                 data_samples=[data_info["data_samples"]])
    with torch.no_grad():
        output = model.test_step(batch)[0]
    return output.pred_instances


def predict_method1(pred, known_classes, known_thr=0.05, unknown_thr=0.001):
    """
    Method 1: "object" token → unknown
    texts = K known + ["object"],  object_idx = K
    """
    n_known    = len(known_classes)
    object_idx = n_known

    boxes  = pred.bboxes.cpu().numpy()
    scores = pred.scores.cpu().numpy()
    labels = pred.labels.cpu().numpy()

    is_obj = labels == object_idx
    keep   = ((~is_obj) & (scores >= known_thr)) | \
              (is_obj    & (scores >= unknown_thr))
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    known_boxes, known_scores, known_names = [], [], []
    unk_boxes, unk_scores = [], []

    for box, sc, lbl in zip(boxes, scores, labels):
        if int(lbl) == object_idx:
            unk_boxes.append(box)
            unk_scores.append(sc)
        else:
            known_boxes.append(box)
            known_scores.append(sc)
            known_names.append(known_classes[int(lbl)])

    return (np.array(known_boxes).reshape(-1, 4),
            np.array(known_scores),
            known_names,
            np.array(unk_boxes).reshape(-1, 4),
            np.array(unk_scores))


def predict_method2(pred, known_classes, unk_thr=0.25, pb_min=0.10,
                    known_keep_thr=0.05, gamma=3):
    """
    Method 2: P_u formula → unknown
    texts = K known + 10 objectiveness prompts
    """
    n_known  = len(known_classes)
    n_attr   = len(OBJECTIVENESS_PROMPTS)

    boxes  = pred.bboxes.cpu().numpy()
    scores = pred.scores.cpu().numpy()
    labels = pred.labels.cpu().numpy()

    if hasattr(pred, 'scores_all') and pred.scores_all.shape[-1] == n_known + n_attr:
        p_u, p_b, _, _ = compute_pu(pred.scores_all.cpu(), n_known, gamma=gamma)
        p_u_np = p_u.numpy()
        p_b_np = p_b.numpy()
    else:
        p_u_np = np.zeros(len(boxes))
        p_b_np = np.zeros(len(boxes))

    known_boxes, known_scores, known_names = [], [], []
    unk_boxes, unk_scores = [], []

    for i, (box, sc, lbl) in enumerate(zip(boxes, scores, labels)):
        is_known_cls = int(lbl) < n_known

        if is_known_cls and sc >= known_keep_thr:
            known_boxes.append(box)
            known_scores.append(sc)
            known_names.append(known_classes[int(lbl)])

        if p_u_np[i] >= unk_thr and p_b_np[i] >= pb_min:
            unk_boxes.append(box)
            unk_scores.append(p_u_np[i])

    return (np.array(known_boxes).reshape(-1, 4),
            np.array(known_scores),
            known_names,
            np.array(unk_boxes).reshape(-1, 4),
            np.array(unk_scores))


# ── Drawing ───────────────────────────────────────────────────────────────────

KNOWN_COLOR = "#4A90D9"    # blue
UNK_COLOR   = "#E74C3C"    # red
TEXT_BG     = (0.0, 0.0, 0.0, 0.55)

def draw_boxes(ax, img_arr, known_boxes, known_scores, known_names,
               unk_boxes, unk_scores, title="", max_known=30, max_unk=20):
    ax.imshow(img_arr)
    ax.set_title(title, fontsize=9, fontweight='bold', pad=3)
    ax.axis('off')

    def draw(boxes, scores, names, color, prefix="", max_n=30):
        if len(boxes) == 0:
            return
        # sort by score descending, keep top-N
        ord_ = np.argsort(-scores)[:max_n]
        for i in ord_:
            x1, y1, x2, y2 = boxes[i]
            w, h = x2 - x1, y2 - y1
            rect = mpatches.FancyBboxPatch(
                (x1, y1), w, h,
                boxstyle="square,pad=0",
                linewidth=1.5, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            if names:
                label = f"{names[i]} {scores[i]:.2f}"
            else:
                label = f"{prefix}{scores[i]:.2f}"
            ax.text(x1 + 2, y1 - 3, label,
                    fontsize=5.5, color='white', fontweight='bold',
                    bbox=dict(boxstyle='square,pad=0.15', fc=color,
                              ec='none', alpha=0.85))

    draw(known_boxes, known_scores, known_names, KNOWN_COLOR, max_n=max_known)
    draw(unk_boxes, unk_scores, [], UNK_COLOR, prefix="unk ", max_n=max_unk)

    # legend
    handles = [
        mpatches.Patch(color=KNOWN_COLOR, label=f"known ({len(known_boxes)})"),
        mpatches.Patch(color=UNK_COLOR,   label=f"unknown ({len(unk_boxes)})"),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=6,
              framealpha=0.7, handlelength=1.2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",    type=int, choices=[1,2,3,4], default=1)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--n",       type=int, default=3,
                        help="시각화할 이미지 수")
    parser.add_argument("--unk-thr", type=float, default=0.25)
    parser.add_argument("--pb-min",  type=float, default=0.10)
    parser.add_argument("--known-thr", type=float, default=0.05)
    parser.add_argument("--unknown-thr", type=float, default=0.001,
                        help="Method 1: object token threshold")
    parser.add_argument("--gamma",   type=int, default=3)
    parser.add_argument("--device",  default="cpu")
    parser.add_argument("--outfile", default=None,
                        help="저장 경로 (기본: vis_task{N}_seed{S}.png)")
    parser.add_argument("--pool",    type=int, default=200,
                        help="랜덤 샘플링할 image pool 크기")
    args = parser.parse_args()

    task_id = args.task
    prev_classes  = [c for split in TASK_SPLITS[:task_id-1] for c in split]
    cur_classes   = TASK_SPLITS[task_id-1]
    known_classes = prev_classes + cur_classes
    n_known       = len(known_classes)
    n_attr        = len(OBJECTIVENESS_PROMPTS)

    texts_m1 = [[c] for c in known_classes] + [["object"]]
    texts_m2 = [[c] for c in known_classes] + [[p] for p in OBJECTIVENESS_PROMPTS]

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading model …")
    cfg = Config.fromfile(CONFIG_FILE)
    from mmyolo.registry import MODELS as MMYOLO_MODELS
    from mmengine.runner import load_checkpoint
    from mmengine.registry import init_default_scope
    init_default_scope("mmyolo")
    model = MMYOLO_MODELS.build(cfg.model)
    load_checkpoint(model, CHECKPOINT, map_location=args.device)
    model = model.to(args.device).eval()

    # Patch for scores_all (needed for Method 2)
    patch_head_for_full_scores(model.bbox_head)
    print("  done.")

    pipeline = build_test_pipeline(cfg)

    # ── Sample images ─────────────────────────────────────────────────────────
    with open(IMAGE_LIST) as f:
        all_ids = [x.strip() for x in f if x.strip()]

    # Use quickeval pool (mixed VOC + COCO) then sample from it
    pool_ids = select_quickeval_ids(all_ids,
                                    voc_limit=args.pool // 2,
                                    coco_limit=args.pool // 2)
    rng = random.Random(args.seed)
    rng.shuffle(pool_ids)
    selected = pool_ids[:args.n]
    print(f"Selected {len(selected)} images: {selected}")

    # ── Figure setup ──────────────────────────────────────────────────────────
    n_imgs    = len(selected)
    n_methods = 2
    fig_w = 5.5 * n_imgs
    fig_h = 5.5 * n_methods
    fig, axes = plt.subplots(n_methods, n_imgs,
                             figsize=(fig_w, fig_h),
                             squeeze=False)

    method_labels = [
        'Method 1: "object" token  (unknown_thr={:.3f})'.format(args.unknown_thr),
        'Method 2: P_u  (unk_thr={:.2f}, pb_min={:.2f})'.format(
            args.unk_thr, args.pb_min),
    ]

    # ── Row labels ────────────────────────────────────────────────────────────
    for row, label in enumerate(method_labels):
        axes[row][0].annotate(
            label, xy=(0, 0.5), xytext=(-0.12, 0.5),
            xycoords='axes fraction', textcoords='axes fraction',
            fontsize=8.5, fontweight='bold', rotation=90,
            va='center', ha='center',
            annotation_clip=False)

    # ── Run inference per image ───────────────────────────────────────────────
    for col, img_id in enumerate(selected):
        img_path, source = resolve_image_path(img_id)
        img_arr = np.array(Image.open(img_path).convert("RGB"))

        # Method 1 inference (object token)
        pred_m1 = infer_raw(model, pipeline, img_path, texts_m1, args.device)
        kb1, ks1, kn1, ub1, us1 = predict_method1(
            pred_m1, known_classes,
            known_thr=args.known_thr, unknown_thr=args.unknown_thr)

        # Method 2 inference (P_u) — texts differ, need separate infer
        pred_m2 = infer_raw(model, pipeline, img_path, texts_m2, args.device)
        kb2, ks2, kn2, ub2, us2 = predict_method2(
            pred_m2, known_classes,
            unk_thr=args.unk_thr, pb_min=args.pb_min,
            known_keep_thr=args.known_thr, gamma=args.gamma)

        col_title = f"{img_id}  ({source})\nT{task_id}: {n_known} known"

        draw_boxes(axes[0][col], img_arr, kb1, ks1, kn1, ub1, us1,
                   title=col_title if col == 0 else img_id)
        draw_boxes(axes[1][col], img_arr, kb2, ks2, kn2, ub2, us2,
                   title="" if col > 0 else "")

        print(f"  [{img_id}] M1: {len(ub1)} unk  | M2: {len(ub2)} unk")

    fig.suptitle(
        f"YOLO-World OWOD Visualization  —  Task {task_id}  (seed={args.seed})",
        fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0.08, 0, 1, 1])

    out = args.outfile or f"owod_reproduce/vis_task{task_id}_seed{args.seed}_t{args.task}.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
