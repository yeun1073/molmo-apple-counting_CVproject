# Experimental Results

## Setup

| Item | Detail |
|---|---|
| Model | allenai/Molmo-7B-D-0924 (4-bit quantization) |
| GPU | NVIDIA RTX 3060 12GB |
| Dataset | MinneApple (detection split) |
| Test set | 331 images (full MinneApple test split) |
| Prompt (P1) | "Point to all the apples in the image." |
| Tile size | 512px, overlap 64px |
| Merge radius | 30px (tuned on val split) |
| YOLO baseline | YOLOv8n, 50 epochs, MinneApple train split |

## Main Results — Counting MAE / RMSE

| Method | Low MAE | Low RMSE | Mid MAE | Mid RMSE | **High MAE** | **High RMSE** | All MAE | All RMSE |
|---|---|---|---|---|---|---|---|---|
| Molmo Raw (zero-shot) | 17.42 | 24.47 | 14.59 | 19.97 | 37.33 | 37.67 | 16.31 | 22.35 |
| **Molmo Tiled (zero-shot)** | 28.98 | 34.01 | 25.57 | 30.52 | **13.08** | **16.60** | 26.20 | 31.29 |
| YOLOv8n (fine-tuned) | **3.70** | **5.03** | **7.49** | **9.55** | 20.75 | 22.27 | **6.77** | **9.22** |

n(low)=105, n(mid)=214, n(high)=12

## Localization F1

| Method | F1 (mean) |
|---|---|
| Molmo Raw | 0.260 |
| Molmo Tiled | **0.406** |

*dist_thresh = 20px*

## merge_radius Tuning (val split, 30 images)

| radius | MAE | RMSE |
|---|---|---|
| 10 | 29.03 | 37.88 |
| 15 | 24.07 | 30.47 |
| 20 | 21.87 | 26.80 |
| **30** | **20.00** | **25.00** |
| 40 | 20.60 | 27.22 |
| 50 | 23.97 | 31.11 |

## YOLO Training (YOLOv8n)

| Metric | Value |
|---|---|
| mAP50 | 0.742 |
| mAP50-95 | 0.402 |
| Precision | 0.769 |
| Recall | 0.653 |
| Training time | 0.119 hours (50 epochs) |

## Key Findings

1. **H1 지지 (occlusion 강인성)**: High-occlusion 환경에서 zero-shot Tiled Molmo (MAE=13.08)가 fine-tuned YOLOv8n (MAE=20.75)을 능가. Raw Molmo는 ~40개 점 생성 한계(saturation)로 인해 밀집 장면에서 MAE=37.33으로 크게 실패.

2. **H2 부분 지지 (타일링 효과)**: 타일링은 high-occ에서 raw 대비 65% MAE 개선. Low/mid에서는 타일 경계 중복 검출로 오히려 악화. Localization F1은 전체적으로 개선(0.260→0.406).

3. **pred=40 포화 현상**: Raw Molmo는 mid 이미지의 64%, high 이미지의 67%에서 pred=40으로 고착. 타일링이 이 구조적 한계를 해결하는 핵심 메커니즘.

## Limitations

- **카운팅 기준 차이**: Molmo는 zero-shot 특성상 지면에 떨어진 사과도 포함하여 카운팅한다. MinneApple GT 라벨은 나무에 달린 사과만 대상으로 하며, YOLO 역시 GT 기준으로 학습되어 동일 기준을 따른다. 이로 인해 특히 낙과가 많은 이미지에서 Molmo의 pred가 체계적으로 과대계상될 수 있으며, YOLO와의 직접 비교 시 이 기준 차이를 명시할 필요가 있다. 향후 연구에서는 "Point to all the apples hanging on the trees" 등 기준을 명시한 프롬프트 변형으로 이를 보정할 수 있다.
