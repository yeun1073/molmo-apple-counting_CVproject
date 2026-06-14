"""
merge_radius 하이퍼파라미터 탐색 스크립트.

val split에서 타일 추론을 1회 실행하고 결과를 캐시에 저장한다.
이후 merge_radius 값만 변경하며 MAE를 계산하여 최적값을 선정한다.
반드시 val split에서만 실행할 것 (test 누수 방지).

사용:
  python -m scripts.04_tune_merge_radius --config configs/config.yaml
  python -m scripts.04_tune_merge_radius --config configs/config.yaml --skip_inference
"""
import argparse
import json
import os
import yaml
import numpy as np
from PIL import Image

from molmo_apple.model import MolmoPointer
from molmo_apple.prompts import PROMPTS
from molmo_apple.tiling import make_tiles, merge_points
from molmo_apple.dataset import load_minneapple, finalize_sample
from molmo_apple.metrics import counting_metrics


RADII = [10, 15, 20, 30, 40, 50, 75, 100]


def run_inference(samples, model, prompt, cfg, cache_path):
    """타일 추론 실행 후 raw 점 좌표를 캐시에 저장."""
    t = cfg["tiling"]
    cache = {}

    for i, s in enumerate(samples, 1):
        image = Image.open(s.image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        raw_pts = []
        for tile_img, off_x, off_y in make_tiles(image, t["tile_size"], t["overlap"]):
            for p in model.point(tile_img, prompt):
                raw_pts.append({"x": p.x + off_x, "y": p.y + off_y})

        cache[os.path.basename(s.image_path)] = {
            "gt": s.gt_count,
            "occ": s.occ_level,
            "raw_pts": raw_pts,
        }
        print(f"  [{i}/{len(samples)}] {os.path.basename(s.image_path):40s} "
              f"raw_pts={len(raw_pts):4d}  gt={s.gt_count:3d}  occ={s.occ_level}")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    print(f"캐시 저장 -> {cache_path}")
    return cache


def evaluate_radius(cache, radius):
    """캐시된 raw 점에 merge_radius 적용 후 MAE 계산."""
    from molmo_apple.parsing import Point
    records = []
    for name, entry in cache.items():
        pts = [Point(p["x"], p["y"]) for p in entry["raw_pts"]]
        merged = merge_points(pts, radius)
        records.append({"pred": len(merged), "gt": entry["gt"], "occ": entry["occ"]})

    preds = np.array([r["pred"] for r in records])
    gts   = np.array([r["gt"]   for r in records])
    errs  = np.abs(preds - gts)
    return float(errs.mean()), float(np.sqrt((errs**2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip_inference", action="store_true", help="캐시 파일 재사용")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    os.makedirs(cfg["output_dir"], exist_ok=True)
    cache_path = os.path.join(cfg["output_dir"], "tune_cache_val.json")

    if args.skip_inference and os.path.exists(cache_path):
        print(f"캐시 재사용: {cache_path}")
        cache = json.load(open(cache_path, encoding="utf-8"))
    else:
        samples = load_minneapple(cfg["dataset"]["root_dir"], split="val")
        prefilled = []
        for s in samples:
            img = Image.open(s.image_path)
            prefilled.append(finalize_sample(s, img.width, img.height))
        print(f"val 샘플: {len(prefilled)}장")

        model = MolmoPointer(cfg["model"]["model_id"], cfg["model"]["generation"]).load()
        prompt = PROMPTS[cfg["prompt"]["key"]]
        cache = run_inference(prefilled, model, prompt, cfg, cache_path)

    # merge_radius 탐색
    print("\n=== merge_radius 탐색 결과 ===")
    print(f"{'radius':>8}  {'MAE':>8}  {'RMSE':>8}")
    print("-" * 30)
    best_radius, best_mae = RADII[0], float("inf")
    for r in RADII:
        mae, rmse = evaluate_radius(cache, r)
        marker = " <-- 현재 config" if r == cfg["tiling"]["merge_radius"] else ""
        print(f"{r:>8}  {mae:>8.2f}  {rmse:>8.2f}{marker}")
        if mae < best_mae:
            best_mae, best_radius = mae, r

    print(f"\n최적 merge_radius = {best_radius}  (MAE={best_mae:.2f})")
    print(f"configs/config.yaml 의 tiling.merge_radius 를 {best_radius} 으로 변경 권장")


if __name__ == "__main__":
    main()
