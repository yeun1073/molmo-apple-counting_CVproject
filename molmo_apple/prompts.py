"""
Molmo pointing 프롬프트 변형 모음.

config의 prompt.key로 선택한다. 본 연구의 주 실험은 P1_basic 사용.
P2/P3는 ablation 참고용이며 본 연구에서는 평가하지 않았다.

주의: P1_basic은 지면에 떨어진 사과도 포함하여 카운팅한다.
MinneApple GT 라벨은 나무에 달린 사과만 대상으로 하므로,
결과 해석 시 이 기준 차이를 명시해야 한다.
"""

PROMPTS = {
    "P1_basic": "Point to all the apples in the image.",
    "P2_occluded": "Point to every apple, including ones that are partially hidden by leaves or other apples.",
    "P3_careful": "Point to all apples in the image. Count carefully and do not miss any occluded or small fruit.",
}

DEFAULT_PROMPT_KEY = "P1_basic"
