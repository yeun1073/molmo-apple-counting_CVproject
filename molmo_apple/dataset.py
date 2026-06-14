"""
MinneApple 데이터셋 로딩 및 GT 변환.

인스턴스 마스크에서 박스를 추출하고, 카운팅 GT·위치 GT(박스 중심)·
occlusion 등급을 산출한다.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image


@dataclass
class Sample:
    image_path: str
    boxes: List[Tuple[float, float, float, float]] = field(default_factory=list)  # x1,y1,x2,y2
    gt_count: int = 0
    gt_points: List[Tuple[float, float]] = field(default_factory=list)  # 박스 중심
    occ_level: str = "unknown"  # low / mid / high


def boxes_to_points(boxes):
    """박스 목록을 중심점 GT 좌표 목록으로 변환한다."""
    return [((x1 + x2) / 2.0, (y1 + y2) / 2.0) for (x1, y1, x2, y2) in boxes]


def occlusion_level(boxes, image_area: float):
    """
    단위 면적당 사과 수(per megapixel)로 occlusion 등급 산출.
    임계값: low<30, 30≤mid<80, high≥80 (val split 분포 기준).
    """
    if not boxes or image_area <= 0:
        return "low"
    density = len(boxes) / image_area * 1e6  # per megapixel
    if density < 30:
        return "low"
    elif density < 80:
        return "mid"
    return "high"


def load_minneapple(root_dir: str, split: str = "train") -> List[Sample]:
    """
    MinneApple 인스턴스 마스크 PNG 로딩.

    마스크 포맷: uint8 단채널, 픽셀값 0=배경, 1~N=사과 인스턴스 ID.
    train: detection/train/images + detection/train/masks
    test : detection/test/images  + test_data/segmentation/masks
    """
    root = Path(root_dir)

    if split in ("train", "val"):
        img_dir  = root / "detection" / "train" / "images"
        mask_dir = root / "detection" / "train" / "masks"
    elif split == "test":
        img_dir  = root / "detection" / "test" / "images"
        mask_dir = root / "test_data" / "segmentation" / "masks"
    else:
        raise ValueError(f"지원하지 않는 split: {split!r}  (train / val / test)")

    val_ids: set | None = None
    if split == "val":
        val_file = Path(__file__).parent.parent / "configs" / "val_ids.txt"
        val_ids = set(val_file.read_text(encoding="utf-8").splitlines())

    samples = []
    for img_path in sorted(img_dir.glob("*.png")):
        if val_ids is not None and img_path.name not in val_ids:
            continue
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            continue  # 마스크 없는 이미지 건너뜀

        mask = np.array(Image.open(mask_path))
        boxes: List[Tuple[float, float, float, float]] = []
        for inst_id in np.unique(mask)[1:]:  # 0=배경 제외
            rows, cols = np.where(mask == inst_id)
            x1, y1 = int(cols.min()), int(rows.min())
            x2, y2 = int(cols.max()), int(rows.max())
            boxes.append((x1, y1, x2, y2))

        s = Sample(image_path=str(img_path), boxes=boxes)
        samples.append(s)

    return samples


def finalize_sample(s: Sample, image_w: int, image_h: int) -> Sample:
    """박스로부터 gt_count, gt_points, occ_level을 채운다."""
    s.gt_count = len(s.boxes)
    s.gt_points = boxes_to_points(s.boxes)
    s.occ_level = occlusion_level(s.boxes, image_w * image_h)
    return s
