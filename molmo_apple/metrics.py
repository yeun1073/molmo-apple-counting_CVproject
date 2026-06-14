"""
평가 지표.

카운팅(주 지표): MAE, RMSE — occlusion 등급별 및 전체.
위치(보조 지표): 예측점-GT 중심점 헝가리안 매칭 후 픽셀 거리 임계값 기반 P/R/F1.
"""
from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
from scipy.optimize import linear_sum_assignment


def counting_metrics(pred_counts: List[int], gt_counts: List[int]) -> Dict[str, float]:
    p = np.asarray(pred_counts, dtype=float)
    g = np.asarray(gt_counts, dtype=float)
    err = p - g
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "n": len(g),
    }


def counting_by_occlusion(records: List[dict]) -> Dict[str, Dict[str, float]]:
    """occlusion 등급(low/mid/high)별 및 전체 MAE/RMSE를 반환한다."""
    out = {}
    for level in ("low", "mid", "high"):
        sub = [r for r in records if r["occ"] == level]
        if sub:
            out[level] = counting_metrics(
                [r["pred"] for r in sub], [r["gt"] for r in sub]
            )
    out["all"] = counting_metrics([r["pred"] for r in records],
                                  [r["gt"] for r in records])
    return out


def localization_prf(
    pred_points: List[Tuple[float, float]],
    gt_points: List[Tuple[float, float]],
    dist_thresh: float,
) -> Dict[str, float]:
    """헝가리안 매칭 기반 P/R/F1. dist_thresh 픽셀 이내 매칭만 TP로 인정."""
    if not pred_points and not gt_points:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_points or not gt_points:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    P = np.asarray(pred_points)
    G = np.asarray(gt_points)
    cost = np.linalg.norm(P[:, None, :] - G[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(cost)
    tp = sum(1 for r, c in zip(ri, ci) if cost[r, c] <= dist_thresh)

    precision = tp / len(pred_points)
    recall = tp / len(gt_points)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp}
