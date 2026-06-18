"""
실험 결과 시각화.

사용:
  python -m scripts.06_plot_results --out_dir outputs/figures
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


RESULTS = {
    "raw":   "outputs/result_raw_P1_basic_test.json",
    "tiled": "outputs/result_tiled_P1_basic_test.json",
    "yolo":  "outputs/result_yolo_test.json",
}
METHOD_LABELS = {
    "raw":   "Molmo Raw\n(zero-shot)",
    "tiled": "Molmo Tiled\n(zero-shot)",
    "yolo":  "YOLOv8n\n(fine-tuned)",
}
COLORS = {"raw": "#5b9bd5", "tiled": "#ed7d31", "yolo": "#70ad47"}
OCC_LEVELS = ["low", "mid", "high", "all"]
OCC_LABELS = ["Low", "Mid", "High", "All"]


def load_results():
    data = {}
    for method, path in RESULTS.items():
        with open(path, encoding="utf-8") as f:
            data[method] = json.load(f)
    return data


def plot_mae_bar(data, out_dir):
    """occlusion 등급별 MAE 막대그래프."""
    fig, ax = plt.subplots(figsize=(9, 5))

    methods = list(data.keys())
    x = np.arange(len(OCC_LEVELS))
    w = 0.25

    for i, method in enumerate(methods):
        maes = []
        for lvl in OCC_LEVELS:
            counting = data[method]["counting"]
            mae = counting.get(lvl, {}).get("MAE", 0)
            maes.append(mae)
        bars = ax.bar(x + (i - 1) * w, maes, w, label=METHOD_LABELS[method],
                      color=COLORS[method], edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(OCC_LABELS, fontsize=11)
    ax.set_ylabel("MAE (↓ better)", fontsize=11)
    n = data["raw"]["n_images"]
    ax.set_title(f"Counting MAE by Occlusion Level  (N={n} test images)", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.set_ylim(0, 45)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # high occ 영역 강조
    ax.axvspan(1.5, 2.5, color="gold", alpha=0.15, zorder=0)
    ax.text(2, 42, "★ Key finding", ha="center", fontsize=8.5, color="goldenrod", fontstyle="italic")

    out = os.path.join(out_dir, "fig_mae_bar.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"저장: {out}")


def plot_rmse_bar(data, out_dir):
    """RMSE 막대그래프."""
    fig, ax = plt.subplots(figsize=(9, 5))
    methods = list(data.keys())
    x = np.arange(len(OCC_LEVELS))
    w = 0.25

    for i, method in enumerate(methods):
        rmses = []
        for lvl in OCC_LEVELS:
            counting = data[method]["counting"]
            rmse = counting.get(lvl, {}).get("RMSE", 0)
            rmses.append(rmse)
        ax.bar(x + (i - 1) * w, rmses, w, label=METHOD_LABELS[method],
               color=COLORS[method], edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(OCC_LABELS, fontsize=11)
    ax.set_ylabel("RMSE (↓ better)", fontsize=11)
    ax.set_title("Counting RMSE by Occlusion Level", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = os.path.join(out_dir, "fig_rmse_bar.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"저장: {out}")


def plot_summary_table(data, out_dir):
    """결과 요약 테이블 이미지."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    col_labels = ["Method", "Low MAE", "Mid MAE", "High MAE", "All MAE", "All RMSE"]
    rows = []
    for method in ["raw", "tiled", "yolo"]:
        c = data[method]["counting"]
        rows.append([
            METHOD_LABELS[method].replace("\n", " "),
            f"{c.get('low',{}).get('MAE',0):.2f}",
            f"{c.get('mid',{}).get('MAE',0):.2f}",
            f"{c.get('high',{}).get('MAE',0):.2f}",
            f"{c.get('all',{}).get('MAE',0):.2f}",
            f"{c.get('all',{}).get('RMSE',0):.2f}",
        ])

    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 2.0)

    # 헤더 스타일
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2e4057")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # high MAE 열 강조 (col index 3)
    best_high_row = min(range(len(rows)), key=lambda i: float(rows[i][3]))
    for i in range(1, len(rows) + 1):
        tbl[i, 3].set_facecolor("#fff3cd" if i - 1 != best_high_row else "#ffd700")
        if i - 1 == best_high_row:
            tbl[i, 3].set_text_props(fontweight="bold")

    n = data["raw"]["n_images"]
    ax.set_title(f"Table: Counting Performance Comparison  (N={n} test images)",
                 fontsize=12, pad=20, fontweight="bold")
    out = os.path.join(out_dir, "fig_table.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"저장: {out}")


def plot_pred_distribution(out_dir):
    """raw vs tiled pred 분포 (체크포인트 기반)."""
    raw_ckpt  = "outputs/ckpt_raw_P1_basic_test.jsonl"
    tiled_ckpt = "outputs/ckpt_tiled_P1_basic_test.jsonl"
    if not (os.path.exists(raw_ckpt) and os.path.exists(tiled_ckpt)):
        print("체크포인트 없음, 분포 플롯 건너뜀")
        return

    def load_ckpt(path):
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    raw_r  = load_ckpt(raw_ckpt)
    tiled_r = load_ckpt(tiled_ckpt)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

    for ax, records, title, color in zip(
        axes,
        [raw_r, tiled_r],
        ["Molmo Raw", "Molmo Tiled (r=30)"],
        [COLORS["raw"], COLORS["tiled"]],
    ):
        preds = [r["pred"] for r in records]
        gts   = [r["gt"]   for r in records]
        ax.scatter(gts, preds, c=color, alpha=0.65, edgecolors="white", linewidth=0.5, s=60)
        lim = max(max(preds), max(gts)) * 1.05
        ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="pred=gt")
        ax.set_xlabel("GT count", fontsize=10)
        ax.set_ylabel("Predicted count", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    n = len(raw_r)
    fig.suptitle(f"Predicted vs GT Count  (N={n} test images)", fontsize=12, fontweight="bold")
    out = os.path.join(out_dir, "fig_scatter.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"저장: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs/figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data = load_results()
    plot_mae_bar(data, args.out_dir)
    plot_rmse_bar(data, args.out_dir)
    plot_summary_table(data, args.out_dir)
    plot_pred_distribution(args.out_dir)
    print("\n모든 그림 저장 완료 ->", args.out_dir)


if __name__ == "__main__":
    main()
