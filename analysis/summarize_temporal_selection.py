import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results"

CONFIGS = {
    "mean_pooling_4hz": {
        "concat": RESULTS / "meld" / "mean_pooling_4hz" / "concat",
        "mlb": RESULTS / "meld" / "mean_pooling_4hz" / "mlb",
    },
    "conformer_2hz": {
        "concat": RESULTS / "meld" / "temporal_selection" / "conformer_2hz" / "concat",
        "mlb": RESULTS / "meld" / "temporal_selection" / "conformer_2hz" / "mlb",
    },
    "conformer_4hz": {
        "concat": RESULTS / "meld" / "temporal_selection" / "conformer_4hz" / "concat",
        "mlb": RESULTS / "meld" / "temporal_selection" / "conformer_4hz" / "mlb",
    },
    "conformer_6hz": {
        "concat": RESULTS / "meld" / "temporal_selection" / "conformer_6hz" / "concat",
        "mlb": RESULTS / "meld" / "temporal_selection" / "conformer_6hz" / "mlb",
    },
    "conformer_8hz": {
        "concat": RESULTS / "meld" / "temporal_selection" / "conformer_8hz" / "concat",
        "mlb": RESULTS / "meld" / "temporal_selection" / "conformer_8hz" / "mlb",
    },
}


def load_metric(folder, metric):
    files = sorted(folder.glob("*.json"))

    if len(files) != 10:
        raise ValueError(
            f"Expected 10 seed files in {folder}, found {len(files)}."
        )

    values = []

    for file in files:
        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        values.append(data["metrics"][metric])

    return values


summary = {}

for config_name, fusion_paths in CONFIGS.items():

    concat_macro = load_metric(fusion_paths["concat"], "macro_f1")
    mlb_macro = load_metric(fusion_paths["mlb"], "macro_f1")

    concat_weighted = load_metric(fusion_paths["concat"], "weighted_f1")
    mlb_weighted = load_metric(fusion_paths["mlb"], "weighted_f1")

    concat_macro_mean = mean(concat_macro)
    mlb_macro_mean = mean(mlb_macro)

    concat_weighted_mean = mean(concat_weighted)
    mlb_weighted_mean = mean(mlb_weighted)

    # Fusion-agnostic temporal-encoder score:
    # each fusion strategy contributes equally.
    macro_f1_fusion_average = mean(
        [concat_macro_mean, mlb_macro_mean]
    )

    weighted_f1_fusion_average = mean(
        [concat_weighted_mean, mlb_weighted_mean]
    )

    summary[config_name] = {
        "n_seeds_per_fusion": 10,
        "concat": {
            "macro_f1_mean": concat_macro_mean,
            "weighted_f1_mean": concat_weighted_mean,
        },
        "mlb": {
            "macro_f1_mean": mlb_macro_mean,
            "weighted_f1_mean": mlb_weighted_mean,
        },
        "fusion_agnostic_average": {
            "macro_f1": macro_f1_fusion_average,
            "weighted_f1": weighted_f1_fusion_average,
        },
    }


output_path = (
    RESULTS
    / "meld"
    / "temporal_selection"
    / "temporal_selection_summary.json"
)

with output_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)


print("\nTemporal Encoder Selection")
print("Fusion-agnostic average across Concat and MLB")
print("-" * 68)
print(
    f"{'Configuration':<22}"
    f"{'Macro-F1':>14}"
    f"{'Weighted-F1':>16}"
)
print("-" * 68)

for config_name, values in summary.items():

    avg = values["fusion_agnostic_average"]

    print(
        f"{config_name:<22}"
        f"{avg['macro_f1'] * 100:>14.2f}"
        f"{avg['weighted_f1'] * 100:>16.2f}"
    )

print("-" * 68)
print(f"\nSaved summary to: {output_path}")