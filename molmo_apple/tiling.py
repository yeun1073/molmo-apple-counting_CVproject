"""
타일링 + 중복 제거 파이프라인.

이미지를 겹침(overlap) 격자 타일로 분할하여 타일당 객체 수를 줄인 뒤,
각 타일의 예측 좌표를 원본 이미지 좌표계로 변환하고,
타일 경계 중복 점을 거리 기반 그리디 병합으로 제거한다.
"""
from __future__ import annotations
from typing import List, Callable
from PIL import Image

from .parsing import Point


def make_tiles(image: Image.Image, tile_size: int, overlap: int):
    """이미지를 (tile_image, offset_x, offset_y) 리스트로 분할."""
    W, H = image.width, image.height
    step = tile_size - overlap
    tiles = []
    y = 0
    while y < H:
        x = 0
        while x < W:
            box = (x, y, min(x + tile_size, W), min(y + tile_size, H))
            tiles.append((image.crop(box), x, y))
            if x + tile_size >= W:
                break
            x += step
        if y + tile_size >= H:
            break
        y += step
    return tiles


def merge_points(points: List[Point], radius: float) -> List[Point]:
    """radius 픽셀 이내의 점들을 그리디 방식으로 하나로 병합한다."""
    merged: List[Point] = []
    for p in points:
        dup = False
        for m in merged:
            if (p.x - m.x) ** 2 + (p.y - m.y) ** 2 < radius ** 2:
                dup = True
                break
        if not dup:
            merged.append(p)
    return merged


def tiled_point(
    image: Image.Image,
    point_fn: Callable[[Image.Image], List[Point]],
    tile_size: int = 512,
    overlap: int = 64,
    merge_radius: float = 15.0,
) -> List[Point]:
    """
    타일별 pointing 후 원본 좌표계로 변환하고 중복을 제거한다.

    point_fn: 타일 이미지를 받아 그 타일 좌표계의 Point 리스트를 반환하는 함수.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    all_points: List[Point] = []
    for tile_img, off_x, off_y in make_tiles(image, tile_size, overlap):
        for p in point_fn(tile_img):
            all_points.append(Point(p.x + off_x, p.y + off_y, p.label))
    return merge_points(all_points, merge_radius)
