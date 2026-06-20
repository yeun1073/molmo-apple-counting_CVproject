"""
전체 평가 파이프라인.

데이터셋에 카운팅을 실행하고 MAE/RMSE(occlusion 등급별 포함) 및
위치 P/R/F1을 산출하여 JSON으로 저장한다.
--method 인자로 raw / tiled / tiled_sam 전환.

사용:
  python -m scripts.02_run_eval --config configs/config.yaml --method raw
  python -m scripts.02_run_eval --config configs/config.yaml --method raw --sample 50
"""
import argparse
import json
import os
import random
import yaml
from collections import defaultdict
from PIL import Image

from molmo_apple.model import MolmoPointer
from molmo_apple.prompts import PROMPTS
from molmo_apple.tiling import tiled_point
from molmo_apple.dataset import load_minneapple, finalize_sample
from molmo_apple.metrics import counting_by_occlusion, localization_prf


def build_point_fn(model, prompt, cfg, method):
    """method에 따라 이미지->점 함수 구성."""
    if method == "raw":
        return lambda img: model.point(img, prompt)

    t = cfg["tiling"]
    def fn(img):
        return tiled_point(
            img,
            point_fn=lambda tile: model.point(tile, prompt),
            tile_size=t["tile_size"],
            overlap=t["overlap"],
            merge_radius=t["merge_radius"],
        )
    return fn


def stratified_sample(samples, n, seed=42):
    """occ 등급별로 균등하게 n장 샘플링."""
    # 먼저 finalize 없이 occ 추정: boxes 수로 임시 분류
    buckets = defaultdict(list)
    for s in samples:
        # 마스크에서 이미 boxes가 채워져 있지 않으면 일단 unknown으로
        buckets[s.occ_level].append(s)

    rng = random.Random(seed)
    result = []
    per_bucket = max(1, n // max(len(buckets), 1))
    for level, items in buckets.items():
        rng.shuffle(items)
        result.extend(items[:per_bucket])

    # 부족하면 나머지에서 채움
    taken = {id(s) for s in result}
    rest = [s for s in samples if id(s) not in taken]
    rng.shuffle(rest)
    result.extend(rest[: max(0, n - len(result))])
    return result[:n]


def load_checkpoint(ckpt_path):
    """이미 처리된 결과 로드 (재시작 지원)."""
    if not os.path.exists(ckpt_path):
        return {}, set()
    records_by_name = {}
    with open(ckpt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                records_by_name[r["name"]] = r
    return records_by_name, set(records_by_name.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", default="tiled", choices=["raw", "tiled"])
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--prompt", default=None, help="프롬프트 키 override (예: P2_occluded). 없으면 config 값 사용")
    ap.add_argument("--limit", type=int, default=None, help="처리할 이미지 수 상한 (디버그용)")
    ap.add_argument("--sample", type=int, default=None, help="층화 샘플 수 (예: 50). None이면 전체")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load_mode", default="4bit", choices=["4bit", "bf16", "auto"])
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    os.makedirs(cfg["output_dir"], exist_ok=True)

    prompt_key = args.prompt if args.prompt else cfg["prompt"]["key"]

    # 체크포인트 파일 (1장씩 저장 -> 중단/재시작 가능)
    tag = f"{args.method}_{prompt_key}_{args.split}"
    ckpt_path = os.path.join(cfg["output_dir"], f"ckpt_{tag}.jsonl")
    done_records, done_names = load_checkpoint(ckpt_path)
    if done_names:
        print(f"[재시작] 이미 완료된 이미지 {len(done_names)}장 건너뜀")

    samples = load_minneapple(cfg["dataset"]["root_dir"], split=args.split)

    # finalize 먼저 해서 occ_level 채우기 (층화 샘플링용)
    prefilled = []
    for s in samples:
        img = Image.open(s.image_path)
        prefilled.append(finalize_sample(s, img.width, img.height))

    if args.sample:
        prefilled = stratified_sample(prefilled, args.sample, seed=args.seed)
        print(f"[샘플링] 층화 샘플 {len(prefilled)}장 선택 (--sample {args.sample})")
    if args.limit:
        prefilled = prefilled[: args.limit]

    # 미완료 이미지만 필터
    todo = [s for s in prefilled if os.path.basename(s.image_path) not in done_names]
    print(f"[eval] {args.method} | split={args.split} | 전체={len(prefilled)}장 | 남은={len(todo)}장")

    model = MolmoPointer(cfg["model"]["model_id"], cfg["model"]["generation"],
                         load_mode=args.load_mode).load()
    prompt = PROMPTS[prompt_key]
    point_fn = build_point_fn(model, prompt, cfg, args.method)

    ckpt_f = open(ckpt_path, "a", encoding="utf-8")
    total = len(prefilled)
    done_count = len(done_names)

    for s in todo:
        done_count += 1
        image = Image.open(s.image_path)
        pts = point_fn(image)
        loc = localization_prf(
            [(p.x, p.y) for p in pts], s.gt_points, cfg["eval"]["loc_dist_thresh"]
        )
        rec = {
            "name": os.path.basename(s.image_path),
            "pred": len(pts), "gt": s.gt_count, "occ": s.occ_level,
            **{f"loc_{k}": v for k, v in loc.items()},
        }
        ckpt_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        ckpt_f.flush()
        done_records[rec["name"]] = rec
        print(f"  [{done_count}/{total}] {rec['name']:40s} "
              f"pred={rec['pred']:3d}  gt={rec['gt']:3d}  occ={rec['occ']}")

    ckpt_f.close()

    # 최종 집계
    all_records = list(done_records.values())
    # 현재 실행에서 선택된 이미지만 집계
    selected_names = {os.path.basename(s.image_path) for s in prefilled}
    all_records = [r for r in all_records if r["name"] in selected_names]

    count_res = counting_by_occlusion(
        [{"pred": r["pred"], "gt": r["gt"], "occ": r["occ"]} for r in all_records]
    )
    avg_f1 = sum(r["loc_f1"] for r in all_records) / max(len(all_records), 1)

    result = {
        "method": args.method,
        "prompt": prompt_key,
        "split": args.split,
        "n_images": len(all_records),
        "counting": count_res,
        "localization_f1_mean": avg_f1,
    }
    out = os.path.join(cfg["output_dir"], f"result_{tag}.json")
    json.dump(result, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
