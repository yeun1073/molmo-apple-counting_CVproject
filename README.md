# Zero-shot Apple Counting with Molmo Pointing

Molmo VLM의 zero-shot pointing 능력으로 과수원 사과를 카운팅하고,
박스 기반 YOLO 대비 밀집·가림(occlusion) 환경에서의 우위를 검증한다.

## Research Questions

- **H1**: Occlusion이 심한 이미지일수록 Molmo(point)가 YOLO(box)보다 누락이 적다.
- **H2**: 타일링 + 중복 제거가 raw Molmo 대비 카운팅 정확도를 유의하게 높인다.

## Key Result

High-occlusion 환경에서 **zero-shot Tiled Molmo (MAE=13.08)**가
**fine-tuned YOLOv8n (MAE=20.75)**을 능가.
→ 상세 수치: [RESULTS.md](RESULTS.md)

## Environment

```
OS           : Windows, Anaconda (conda env: molmo)
GPU          : NVIDIA RTX 3060 12GB
Model        : allenai/Molmo-7B-D-0924 (4-bit quantization)
transformers : 4.45.2
Dataset      : MinneApple (detection split)  train 670장 / test 331장
```

## Project Structure

```
molmo_apple/
├── molmo_apple/                        # 패키지
│   ├── model.py                        # Molmo 로드 + 추론 (4bit 양자화)
│   ├── parsing.py                      # 좌표 파싱 (1세대 0~100 정규화)
│   ├── prompts.py                      # 프롬프트 변형 P1/P2/P3
│   ├── tiling.py                       # ★ 타일링 + 중복 제거 (핵심 기여)
│   ├── dataset.py                      # MinneApple 로딩 + GT 변환 + occ 등급화
│   ├── metrics.py                      # MAE/RMSE (등급별) + 위치 P/R/F1
│   ├── visualize.py                    # 좌표 시각화 도구
│   └── sam_filter.py                   # SAM 거짓양성 필터 (선택 ablation)
├── scripts/
│   ├── 01_smoke_test.py                # 단일 이미지 추론 + 좌표 육안 검증
│   ├── 02_run_eval.py                  # 전체 평가 파이프라인 (체크포인트 지원)
│   ├── 03_yolo_baseline.py             # YOLO 대조군 평가
│   ├── 04_tune_merge_radius.py         # merge_radius 튜닝 (val split 전용)
│   ├── 05_prepare_yolo.py              # MinneApple 마스크 → YOLO 라벨 변환
│   └── 06_plot_results.py              # 결과 시각화 (bar chart / scatter / table)
├── configs/
│   ├── config.yaml                     # 실험 설정 (모델·타일·데이터 경로)
│   └── val_ids.txt                     # val split (train에서 30장 층화 샘플, seed=42)
├── outputs/
│   ├── figures/                        # 생성된 그래프 (png)
│   ├── result_raw_P1_basic_test.json   # raw baseline 결과
│   ├── result_tiled_P1_basic_test.json # tiled 결과 (merge_radius=30)
│   ├── result_yolo_test.json           # YOLO 결과
│   ├── ckpt_raw_P1_basic_test.jsonl    # 이미지별 raw 체크포인트
│   └── ckpt_tiled_P1_basic_test.jsonl  # 이미지별 tiled 체크포인트
├── RESULTS.md                          # 상세 실험 수치 및 분석
└── README.md
```

## Quick Start

```bash
conda activate molmo

# 1. 단일 이미지 동작 확인
python -m scripts.01_smoke_test --image apple.jpg --load_mode 4bit

# 2. 전체 평가 (raw, 50장 층화 샘플)
python -m scripts.02_run_eval --config configs/config.yaml \
    --method raw --split test --sample 50

# 3. 타일링 평가
python -m scripts.02_run_eval --config configs/config.yaml \
    --method tiled --split test --sample 50

# 4. YOLO 라벨 생성 + 학습
python -m scripts.05_prepare_yolo --root C:/datasets/MinneApple
yolo train data=C:/datasets/MinneApple/yolo/dataset.yaml \
    model=yolov8n.pt epochs=50 imgsz=640 batch=8

# 5. YOLO 대조군 평가
python -m scripts.03_yolo_baseline --config configs/config.yaml \
    --weights runs/detect/.../best.pt --sample 50

# 6. merge_radius 튜닝 (val split에서만)
python -m scripts.04_tune_merge_radius --config configs/config.yaml

# 7. 결과 시각화
python -m scripts.06_plot_results --out_dir outputs/figures
```

## Dataset

[MinneApple](https://github.com/nicolaihaeni/MinneApple) — 과수원 사과 검출/분할 데이터셋

```
C:\datasets\MinneApple\
  detection\
    train\  images/ + masks/  (670장, 마스크 있음)
    test\   images/           (331장)
  test_data\
    segmentation\  masks/    (test GT 마스크, 331장)
```

마스크 포맷: uint8 PNG, 픽셀값 0=배경, 1~N=사과 인스턴스 ID

## Occlusion Level Definition

이미지 단위 밀도(사과 수 / 이미지 픽셀 수 × 10⁶, per megapixel) 기준:

| Level | 기준 | test 내 비율 |
|---|---|---|
| low | density < 30 | 34% (17/50) |
| mid | 30 ≤ density < 80 | 42% (21/50) |
| high | density ≥ 80 | 24% (12/50) |

## Implementation Notes

- 모든 Molmo 추론: `load_mode=4bit` (VRAM 12GB 제약)
- `max_new_tokens=2000` 필수 — 500 설정 시 ~29개에서 생성 중단 현상 발생
- 튜닝(tile_size, merge_radius, 프롬프트)은 반드시 **val split**에서만 수행
- 스크립트 실행은 `-m` 모듈 방식 필수 (`python scripts/xxx.py` 는 import 오류)
- 중간 중단 시 체크포인트 자동 재사용 (`ckpt_*.jsonl`)

## Citation

If you use this code, please cite:

```
[작성 예정]
```
