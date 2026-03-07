import os
import torch
import graphbench
import pandas as pd

from torch_geometric.loader import DataLoader
from models.baseline_mlp import BaselineMLP

ROOT = "./results"
DATA_ROOT = "./graphbench_data"
SEED = 42
BATCH_SIZE = 512
HIDDEN_DIM = 384
MODEL_NAME = "BaselineMLP"

TRANSFER_TASKS = {
    "electronic_circuits_5_eff": ["electronic_circuits_7_eff", "electronic_circuits_10_eff"],
    "electronic_circuits_5_vout": ["electronic_circuits_7_vout", "electronic_circuits_10_vout"],
}


class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, name="ds"):
        self.base = base_ds
        self.idx = []
        self.bad = 0

        for i in range(len(base_ds)):
            y = getattr(base_ds[i], "y", None)
            if y is None or torch.isnan(y).any() or torch.isinf(y).any():
                self.bad += 1
            else:
                self.idx.append(i)

        print(f"  {name}: kept {len(self.idx)}/{len(base_ds)}, removed {self.bad}")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.base[self.idx[i]]


@torch.no_grad()
def eval_rse(model, loader, evaluator, device):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        batch = batch.to(device)
        y_pred = model(batch).view(-1, 1)
        y_true = batch.y.view(-1, 1).float()
        preds.append(y_pred.cpu())
        trues.append(y_true.cpu())
    y_pred = torch.cat(preds, dim=0)
    y_true = torch.cat(trues, dim=0)
    return evaluator.evaluate(y_pred=y_pred, y_true=y_true)


def latest_run_dir(path):
    if not os.path.isdir(path):
        return None
    subdirs = [
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    ]
    if not subdirs:
        return None
    subdirs = sorted(subdirs)
    return os.path.join(path, subdirs[-1])


def checkpoint_path(source_dataset: str):
    dataset_folder = os.path.join(ROOT, source_dataset)
    run_dir = latest_run_dir(dataset_folder)
    if run_dir is None:
        return None
    ckpt = os.path.join(run_dir, "best_model.pt")
    return ckpt if os.path.isfile(ckpt) else None


def load_source_input_dim(source_dataset: str):
    loader = graphbench.Loader(root=DATA_ROOT, dataset_names=[source_dataset])
    splits = loader.load()[0]

    train_ds = FilteredDataset(splits["train"], f"{source_dataset}-train")

    # reproduce training-time 73k subsampling on 5-component datasets
    if "5_" in source_dataset and len(train_ds) > 73000:
        rng = torch.Generator().manual_seed(SEED)
        indices = torch.randperm(len(train_ds), generator=rng)[:73000].tolist()
        train_ds = torch.utils.data.Subset(train_ds, indices)
        print(f"  subsampled source train to {len(train_ds)} samples")

    sample = train_ds[0]
    x = sample.x.squeeze(1) if sample.x.dim() == 3 else sample.x
    return x.size(-1)


def load_target_test_loader(target_dataset: str, batch_size: int):
    loader = graphbench.Loader(root=DATA_ROOT, dataset_names=[target_dataset])
    splits = loader.load()[0]
    test_ds = FilteredDataset(splits["test"], f"{target_dataset}-test")
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)


def pretty_col(source, target):
    short = {
        "electronic_circuits_5_eff": "5_eff",
        "electronic_circuits_5_vout": "5_vout",
        "electronic_circuits_7_eff": "7_eff",
        "electronic_circuits_7_vout": "7_vout",
        "electronic_circuits_10_eff": "10_eff",
        "electronic_circuits_10_vout": "10_vout",
    }
    return f"{short[source]}→{short[target]}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = graphbench.Evaluator("electroniccircuit")

    table = {MODEL_NAME: {}}

    for source_dataset, target_datasets in TRANSFER_TASKS.items():
        print("\n" + "=" * 100)
        print(f"SOURCE TRAIN DATASET: {source_dataset}")
        print("=" * 100)

        ckpt = checkpoint_path(source_dataset)
        if ckpt is None:
            print(f"  [WARNING] No checkpoint found for {source_dataset}")
            for target_dataset in target_datasets:
                table[MODEL_NAME][pretty_col(source_dataset, target_dataset)] = None
            continue

        input_dim = load_source_input_dim(source_dataset)
        model = BaselineMLP(input_dim=input_dim, hidden_dim=HIDDEN_DIM, out_channels=1).to(device)

        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)

        print(f"  Loaded checkpoint: {ckpt}")

        for target_dataset in target_datasets:
            print(f"  Evaluating on target dataset: {target_dataset}")
            test_loader = load_target_test_loader(target_dataset, BATCH_SIZE)
            test_rse = eval_rse(model, test_loader, evaluator, device)
            col = pretty_col(source_dataset, target_dataset)
            table[MODEL_NAME][col] = test_rse
            print(f"    {col}: {test_rse:.6f}")

    cols = [
        "5_eff→7_eff",
        "5_eff→10_eff",
        "5_vout→7_vout",
        "5_vout→10_vout",
    ]

    df = pd.DataFrame.from_dict(table, orient="index")
    df = df[cols]

    print("\n" + "=" * 100)
    print("GENERALIZATION TEST RSE TABLE (BaselineMLP)")
    print("=" * 100)
    print(df.to_string())
    print("=" * 100)

    csv_path = os.path.join(ROOT, "summary_generalization_test_rse_baseline_mlp.csv")
    df.to_csv(csv_path)
    print(f"\nCSV saved to: {csv_path}")

    latex_table = df.round(4).to_latex(
        index=True,
        caption="Generalization test RSE for BaselineMLP. The model is trained on 5-component circuits and evaluated on larger circuits of the same target type.",
        label="tab:generalization_baseline_mlp",
        column_format="l" + "c" * len(df.columns),
        escape=False,
    )

    print("\n" + "=" * 100)
    print("LATEX TABLE")
    print("=" * 100)
    print(latex_table)
    print("=" * 100)

    tex_path = os.path.join(ROOT, "summary_generalization_test_rse_baseline_mlp.tex")
    with open(tex_path, "w") as f:
        f.write(latex_table)

    print(f"\nLaTeX saved to: {tex_path}")


if __name__ == "__main__":
    main()