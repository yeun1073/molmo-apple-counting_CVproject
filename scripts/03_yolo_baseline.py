"""
[5단계] YOLO 검출 기반 카운팅 baseline.

Molmo와 동일한 GT/등급/지표로 평가해 공정 비교.
YOLO는 학습된 모델 -> "Molmo는 zero-shot인데도" 프레이밍의 대조군.

사용:
  python -m scripts.03_yolo_baseline --config configs/config.yaml --weights runs/detect/outputs/yolo_train/minneapple/weights/best.pt
  python -m scripts.03_yolo_baseline --config configs/config.yaml --weights ... --sample 50
"""
import argparse
import json
import os
import yaml
from PIL import Image

from molmo_apple.dataset import load_minneapple, finalize_sample
from molmo_apple.metrics import counting_by_occlusion


def stratified_sample(samples, n, seed=42):
    import random
    from collections import defaultdict
    buckets = defaultdict(list)
    for s in samples:
        buckets[s.occ_level].append(s)
    rng = random.Random(seed)
    result = []
    per_bucket = max(1, n // max(len(buckets), 1))
    for items in buckets.values():
        rng.shuffle(items)
        result.extend(items[:per_bucket])
    taken = {id(s) for s in result}
    rest = [s for s in samples if id(s) not in taken]
    rng.shuffle(rest)
    result.extend(rest[: max(0, n - len(result))])
    return result[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    os.makedirs(cfg["output_dir"], exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(args.weights)

    samples = load_minneapple(cfg["dataset"]["root_dir"], split=args.split)
    prefilled = []
    for s in samples:
        img = Image.open(s.image_path)
        prefilled.append(finalize_sample(s, img.width, img.height))

    if args.sample:
        prefilled = stratified_sample(prefilled, args.sample, seed=args.seed)
        print(f"[샘플링] {len(prefilled)}장")
    print(f"[eval] yolo | split={args.split} | {len(prefilled)}장")

    records = []
    for i, s in enumerate(prefilled, 1):
        res = model(s.image_path, conf=args.conf, verbose=False)[0]
        pred_count = len(res.boxes)
        records.append({"pred": pred_count, "gt": s.gt_count, "occ": s.occ_level})
        print(f"  [{i}/{len(prefilled)}] {os.path.basename(s.image_path):40s} "
              f"pred={pred_count:3d}  gt={s.gt_count:3d}  occ={s.occ_level}")

    count_res = counting_by_occlusion(records)
    result = {
        "method": "yolo",
        "weights": args.weights,
        "split": args.split,
        "n_images": len(records),
        "conf": args.conf,
        "counting": count_res,
    }
    out = os.path.join(cfg["output_dir"], f"result_yolo_{args.split}.json")
    json.dump(result, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
