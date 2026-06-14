"""
MinneApple 인스턴스 마스크 -> YOLO 포맷 변환 + dataset.yaml 생성.

사용:
  python -m scripts.05_prepare_yolo --root C:/datasets/MinneApple
"""
import argparse
import numpy as np
from pathlib import Path
from PIL import Image


def mask_to_yolo_labels(mask_path: Path, img_w: int, img_h: int):
    """마스크 PNG에서 YOLO bounding box 라벨 생성. class=0 (apple)."""
    mask = np.array(Image.open(mask_path))
    lines = []
    for inst_id in np.unique(mask)[1:]:  # 0=배경 제외
        rows, cols = np.where(mask == inst_id)
        x1, y1 = cols.min(), rows.min()
        x2, y2 = cols.max(), rows.max()
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w  = (x2 - x1) / img_w
        h  = (y2 - y1) / img_h
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def convert_split(img_dir: Path, mask_dir: Path, label_dir: Path):
    label_dir.mkdir(parents=True, exist_ok=True)
    img_paths = sorted(img_dir.glob("*.png"))
    print(f"  변환: {img_dir.name} ({len(img_paths)}장) -> {label_dir}")
    for img_path in img_paths:
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            continue
        img = Image.open(img_path)
        lines = mask_to_yolo_labels(mask_path, img.width, img.height)
        label_path = label_dir / (img_path.stem + ".txt")
        label_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="C:/datasets/MinneApple")
    args = ap.parse_args()
    root = Path(args.root)
    yolo_dir = root / "yolo"

    # YOLO는 images/ 와 형제 디렉터리인 labels/ 에서 라벨을 찾음
    # train
    convert_split(
        root / "detection" / "train" / "images",
        root / "detection" / "train" / "masks",
        root / "detection" / "train" / "labels",   # images/ 옆에 위치
    )
    # test (마스크는 test_data/segmentation/masks)
    convert_split(
        root / "detection" / "test" / "images",
        root / "test_data" / "segmentation" / "masks",
        root / "detection" / "test" / "labels",    # images/ 옆에 위치
    )

    # dataset.yaml
    yaml_content = f"""path: {root.as_posix()}
train: detection/train/images
val:   detection/test/images
test:  detection/test/images

nc: 1
names: [apple]
"""
    yaml_path = yolo_dir / "dataset.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"dataset.yaml 저장 -> {yaml_path}")
    print("완료.")


if __name__ == "__main__":
    main()
