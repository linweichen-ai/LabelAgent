"""Export a static scientific comparison figure for the GitHub Markdown record."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def plot(summary, output):
    result = json.loads(Path(summary).read_text(encoding="utf-8"))
    method = result["methods"]["wilddet3d-labels"]
    for source in result["methods"].values():
        for threshold in ("0.25", "0.5"):
            all_classes = source["metrics"]["all"][threshold]
            for field in ("matched", "predictions", "references"):
                assert all_classes[field] == sum(stats[threshold][field] for label, stats in source["metrics"].items() if label != "all")
            for stats in source["metrics"].values():
                row = stats[threshold]
                assert row["matched"] + row["unmatched_predictions"] == row["predictions"]
                assert row["matched"] + row["unmatched_references"] == row["references"]
                assert abs(row["precision_against_reference"] - row["matched"] / row["predictions"]) < 1e-12
                assert abs(row["recall_against_reference"] - row["matched"] / row["references"]) < 1e-12
    categories = [("Pallet", "precision_against_reference"), ("Pallet", "recall_against_reference"),
                  ("Pedestrian", "precision_against_reference"), ("Pedestrian", "recall_against_reference")]
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    x = np.arange(len(categories))
    for offset, threshold, color in ((-0.19, "0.25", "#1967D2"), (0.19, "0.5", "#C45A18")):
        values = [method["metrics"][label][threshold][field] for label, field in categories]
        bars = ax.bar(x + offset, values, 0.34, color=color, label=f"3-D IoU >= {threshold}")
        ax.bar_label(bars, labels=[f"{100 * value:.1f}%" for value in values], padding=4, fontsize=10)
    ax.set_xticks(x, ["Pallet\nPrecision", "Pallet\nRecall", "Person\nPrecision", "Person\nRecall"])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Agreement with selected reference")
    fig.suptitle("Archived wilddet3d boxes vs labels reference", x=0.09, y=0.96, ha="left", fontsize=15)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False)
    ax.grid(axis="y", color="#DCE1E6", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.09, 0.03,
        f"Coverage: {method['evaluated_common_frames']}/{result['reference_frames']} frames. Source ignore flags excluded.\n"
        "Reference consistency only; not independent accuracy. Missing prediction frames are not scored.",
        fontsize=9, color="#444444")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.21, top=0.80)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white")
    plt.close(fig)
    return {"aggregate_counts_reconciled": True, "ratios_recomputed": True, "figure": output.name}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.summary, args.output)))
