"""
단일 이미지 추론 및 좌표 육안 검증.

사용:
  python -m scripts.01_smoke_test --image apple.jpg --load_mode 4bit
  python -m scripts.01_smoke_test --image apple.jpg --show_raw --load_mode 4bit
"""
import argparse
import os
from PIL import Image

from molmo_apple.model import MolmoPointer
from molmo_apple.prompts import PROMPTS, DEFAULT_PROMPT_KEY
from molmo_apple.visualize import draw_points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model_id", default="allenai/Molmo-7B-D-0924")
    ap.add_argument("--generation", type=int, default=1)
    ap.add_argument("--prompt_key", default=DEFAULT_PROMPT_KEY)
    ap.add_argument("--out", default="outputs/check.jpg")
    ap.add_argument("--load_mode", default="4bit", choices=["auto", "bf16", "4bit"],
                    help="12GB GPU는 4bit 권장")
    ap.add_argument("--gpu_mem_gib", type=int, default=11,
                    help="bf16 모드에서 GPU에 올릴 상한(GiB)")
    ap.add_argument("--show_raw", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"모델 로딩 중... (load_mode={args.load_mode})")
    model = MolmoPointer(
        args.model_id, args.generation,
        load_mode=args.load_mode, gpu_mem_gib=args.gpu_mem_gib,
    ).load()
    image = Image.open(args.image)
    prompt = PROMPTS[args.prompt_key]

    if args.show_raw:
        text = model.raw_text(image, prompt)
        print("=" * 50)
        print("모델 원본 출력:")
        print(text)
        print("=" * 50)
        print("위 출력의 좌표 형식이 parsing.py 정규식과 맞는지 확인하세요.")
        return

    points = model.point(image, prompt)
    print(f"detected points: {len(points)}")
    for p in points[:10]:
        print(f"  ({p.x:.1f}, {p.y:.1f})")

    draw_points(image, points, args.out)
    print(f"saved -> {args.out}  (점이 사과 위에 오는지 육안 확인!)")


if __name__ == "__main__":
    main()
