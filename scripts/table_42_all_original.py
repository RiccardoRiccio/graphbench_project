import os
import json
import pandas as pd

ROOT = "./results_seed_42_all_originals"

models = [
    "PNAOriginal",
    "GPSoriginal",
    "GCNoriginal",
    "GINoriginal",
]

datasets = [
    "electronic_circuits_5_eff",
    "electronic_circuits_5_vout",
    "electronic_circuits_7_eff",
    "electronic_circuits_7_vout",
    "electronic_circuits_10_eff",
    "electronic_circuits_10_vout",
]

# mapping between model name and folder name used in training
model_to_folder = {
    "PNAOriginal": "results_42_pnaoriginal",
    "GPSoriginal": "results_42_gpsoriginal",
    "GCNoriginal": "results_42_gcnoriginal",
    "GINoriginal": "results_42_ginoriginal",
}

table = {}

for model in models:

    table[model] = {}

    model_folder = os.path.join(ROOT, model_to_folder[model])

    for dataset in datasets:

        dataset_folder = os.path.join(model_folder, dataset)

        if not os.path.isdir(dataset_folder):
            table[model][dataset] = None
            continue

        run_dirs = [
            d for d in os.listdir(dataset_folder)
            if os.path.isdir(os.path.join(dataset_folder, d))
        ]

        if len(run_dirs) == 0:
            table[model][dataset] = None
            continue

        run_dirs = sorted(run_dirs)
        latest_run = run_dirs[-1]

        results_json = os.path.join(dataset_folder, latest_run, "results.json")

        if not os.path.isfile(results_json):
            table[model][dataset] = None
            continue

        with open(results_json, "r") as f:
            results = json.load(f)

        table[model][dataset] = results.get("test_rse", None)

# build dataframe
df = pd.DataFrame.from_dict(table, orient="index")
df = df[datasets]

# ---- PRINT TABLE ----

print("\n" + "="*80)
print("TEST RSE RESULTS TABLE")
print("="*80)

print(df.to_string())

print("="*80)

# ---- SAVE CSV ----

output_csv = os.path.join(ROOT, "summary_test_rse_42_all_originals.csv")
df.to_csv(output_csv)

print(f"\nTable saved to: {output_csv}")