"""
SAM 기반 거짓양성 필터 (ablation 선택 모듈).

각 예측 점을 SAM 프롬프트로 입력해 마스크를 얻고,
마스크 면적이 사과 크기 범위를 벗어나면 거짓양성으로 제거한다.
config의 sam.enabled=false 시 이 모듈은 사용하지 않는다.
"""
from __future__ import annotations
from typing import List
import numpy as np
from PIL import Image

from .parsing import Point


class SamFilter:
    def __init__(self, sam_predictor, min_area_ratio: float, max_area_ratio: float):
        """
        sam_predictor: SAM2/SAM predictor 객체
        min/max_area_ratio: 전체 이미지 대비 허용 마스크 면적 비율
        """
        self.predictor = sam_predictor
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def filter(self, image: Image.Image, points: List[Point]) -> List[Point]:
        img_arr = np.array(image.convert("RGB"))
        self.predictor.set_image(img_arr)
        total = image.width * image.height
        kept: List[Point] = []
        for p in points:
            masks, scores, _ = self.predictor.predict(
                point_coords=np.array([[p.x, p.y]]),
                point_labels=np.array([1]),
                multimask_output=False,
            )
            area = float(masks[0].sum())
            ratio = area / total
            if self.min_area_ratio <= ratio <= self.max_area_ratio:
                kept.append(p)
        return kept
