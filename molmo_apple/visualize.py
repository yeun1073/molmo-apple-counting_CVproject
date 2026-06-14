"""
시각화 유틸리티.

draw_points: 예측 점을 이미지에 오버레이하여 좌표 검증에 사용한다.
compare: 예측 점(빨강)과 GT 중심점(초록)을 같은 이미지에 오버레이한다.
"""
from __future__ import annotations
from typing import List, Tuple
from PIL import Image, ImageDraw

from .parsing import Point


def draw_points(image: Image.Image, points: List[Point], out_path: str,
                color=(255, 0, 0), r: int = 6):
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    for p in points:
        d.ellipse([p.x - r, p.y - r, p.x + r, p.y + r], outline=color, width=3)
    img.save(out_path)
    return out_path


def compare(image: Image.Image,
            pred_points: List[Point],
            gt_points: List[Tuple[float, float]],
            out_path: str):
    """예측(빨강) vs GT(초록) 오버레이."""
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    for (gx, gy) in gt_points:
        d.ellipse([gx - 5, gy - 5, gx + 5, gy + 5], outline=(0, 200, 0), width=3)
    for p in pred_points:
        d.ellipse([p.x - 5, p.y - 5, p.x + 5, p.y + 5], outline=(255, 0, 0), width=3)
    img.save(out_path)
    return out_path
