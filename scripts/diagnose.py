# # diagnose3.py
# import graphbench
# import torch

# DATASET_NAME = "electronic_circuits_5_eff"
# ROOT = "./graphbench_data"

# loader = graphbench.Loader(root=ROOT, dataset_names=[DATASET_NAME])
# splits = loader.load()[0]

# for split_name, ds in [("train", splits["train"]), ("valid", splits["valid"]), ("test", splits["test"])]:
#     all_y = torch.stack([g.y.squeeze() for g in ds])
#     n_nan = torch.isnan(all_y).sum().item()
#     n_inf = torch.isinf(all_y).sum().item()
#     n_bad = n_nan + n_inf
#     print(f"{split_name}: total={len(ds)}  nan={n_nan}  inf={n_inf}  bad={n_bad} ({100*n_bad/len(ds):.1f}%)")

# # add to diagnose.py
# DATASET_NAME = "electronic_circuits_10_vout"
# loader = graphbench.Loader(root=ROOT, dataset_names=[DATASET_NAME])
# splits = loader.load()[0]

# all_y = torch.stack([g.y.squeeze() for g in splits["train"]])
# print(f"mean: {all_y.mean():.4f}")
# print(f"std:  {all_y.std():.4f}")
# print(f"min:  {all_y.min():.4f}")
# print(f"max:  {all_y.max():.4f}")
# print(f"% < 0.01: {(all_y.abs() < 0.01).float().mean()*100:.1f}%")

# diagnose_datasets.py
import graphbench
import torch

ROOT = "./graphbench_data"

datasets = [
    "electronic_circuits_5_eff",
    "electronic_circuits_5_vout",
    "electronic_circuits_7_eff",
    "electronic_circuits_7_vout",  # the problematic one
    "electronic_circuits_10_eff",
    "electronic_circuits_10_vout",
]

for dataset_name in datasets:
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    try:
        loader = graphbench.Loader(root=ROOT, dataset_names=[dataset_name])
        splits = loader.load()[0]

        for split_name, ds in [("train", splits["train"]), ("valid", splits["valid"]), ("test", splits["test"])]:
            all_y = torch.stack([g.y.squeeze() for g in ds])
            n_nan = torch.isnan(all_y).sum().item()
            n_inf = torch.isinf(all_y).sum().item()
            print(f"  {split_name:<6} | samples: {len(ds):>7} | nan: {n_nan} | inf: {n_inf} | mean: {all_y.mean():.4f} | std: {all_y.std():.4f} | min: {all_y.min():.4f} | max: {all_y.max():.4f}")

    except Exception as e:
        print(f"  ERROR loading dataset: {e}")