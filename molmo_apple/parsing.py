"""
Molmo pointing 출력 텍스트 -> 픽셀 좌표 파싱.

★ 프로젝트에서 가장 먼저 정확히 맞춰야 하는 부분 ★
세대별로 좌표 스케일과 출력 포맷이 다르므로, 실제 출력 샘플을 눈으로 보고
아래 정규식이 맞는지 반드시 확인할 것. 그 후 visualize.py로 육안 검증.

- 1세대 (Molmo-7B-D/O-0924): 좌표 0~100 정규화, <point x="..." y="..."> 류 텍스트
- 2세대 (Molmo2-*): 좌표 1000 스케일, <points ... coords="..."> 태그
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List


@dataclass
class Point:
    x: float  # 픽셀 좌표
    y: float
    label: str | None = None


# ---------------------------------------------------------------------------
# 1세대: 0~100 정규화. Molmo는 <point x="12.3" y="45.6" alt="apple">apple</point>
# 또는 <points x1=.. y1=.. x2=.. y2=.. .../> 형태로 출력하는 경우가 있음.
# 아래는 단일/복수 point 태그를 모두 커버하는 관용 정규식.
# 실제 출력을 확인한 뒤 필요하면 조정.
# ---------------------------------------------------------------------------
_GEN1_XY = re.compile(r'x\d*\s*=\s*"?([0-9.]+)"?\s+y\d*\s*=\s*"?([0-9.]+)"?')


def parse_gen1(text: str, image_w: int, image_h: int) -> List[Point]:
    """1세대 출력 파싱. 좌표는 0~100 정규화 가정 -> 픽셀로 변환."""
    pts: List[Point] = []
    for m in _GEN1_XY.finditer(text):
        xn, yn = float(m.group(1)), float(m.group(2))
        x = xn / 100.0 * image_w
        y = yn / 100.0 * image_h
        if 0 <= x <= image_w and 0 <= y <= image_h:
            pts.append(Point(x, y))
    return pts


# ---------------------------------------------------------------------------
# 2세대: 1000 스케일. 모델 카드의 정규식 로직을 옮긴 것.
# ---------------------------------------------------------------------------
_GEN2_POINTS = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")


def parse_gen2(text: str, image_w: int, image_h: int) -> List[Point]:
    """2세대 출력 파싱. 좌표는 1000 스케일 가정 -> 픽셀로 변환."""
    pts: List[Point] = []
    for m in _GEN2_POINTS.finditer(text):
        _ix, x, y = m.group(1), m.group(2), m.group(3)
        x_px = float(x) / 1000.0 * image_w
        y_px = float(y) / 1000.0 * image_h
        if 0 <= x_px <= image_w and 0 <= y_px <= image_h:
            pts.append(Point(x_px, y_px))
    return pts


def parse_points(text: str, image_w: int, image_h: int, generation: int = 1) -> List[Point]:
    """세대에 맞는 파서 호출. config의 generation 값으로 분기."""
    if generation == 1:
        return parse_gen1(text, image_w, image_h)
    elif generation == 2:
        return parse_gen2(text, image_w, image_h)
    raise ValueError(f"Unknown generation: {generation}")
