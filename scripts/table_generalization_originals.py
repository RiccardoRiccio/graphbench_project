import os
import json
import torch
import graphbench
import pandas as pd

from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

from models.baseline_pna_original import PNAOriginal
from models.baseline_gps_original import GPSoriginal
from models.baseline_gin_original import GINoriginal
from models.baseline_gt_original import GToriginal
from models.baseline_gcn_original import GCNoriginal

try:
    from torch_geometric.transforms import AddRandomWalkPE
    HAS_RWSE = True
except ImportError:
    HAS_RWSE = False
    print("WARNING: AddRandomWalkPE not found in your PyG version. GT/GPS will run without RWSE.")

ROOT = "./results_seed_42_all_originals"
DATA_ROOT = "./graphbench_data"
SEED = 42
RWSE_DIM = 16
BATCH_SIZE = 512
HIDDEN_DIM = 384

MODELS = [
    "PNAOriginal",
    "GPSoriginal",
    "GCNoriginal",
    "GINoriginal",
    "GToriginal",
]

# source -> targets
TRANSFER_TASKS = {
    "electronic_circuits_5_eff": ["electronic_circuits_7_eff", "electronic_circuits_10_eff"],
    "electronic_circuits_5_vout": ["electronic_circuits_7_vout", "electronic_circuits_10_vout"],
}


class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, name="ds", transform=None):
        self.base = base_ds
        self.transform = transform
        self.idx = []
        self.bad = 0
        self.cache = {}

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
        base_i = self.idx[i]
        data = self.base[base_i]

        if self.transform is not None:
            if base_i not in self.cache:
                data = self.transform(data)
                self.cache[base_i] = data.pe.cpu()
            else:
                data.pe = self.cache[base_i]

        return data


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


def compute_deg(dataset):
    max_degree = -1
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg


def build_model(model_key: str, input_dim: int, hidden_dim: int, deg=None):
    if model_key == "GCNoriginal":
        return GCNoriginal(
            in_channels=input_dim,
            hidden_dim=hidden_dim,
            num_layers=4,
            out_dim=1,
            dropout=0.0,
        )

    if model_key == "GINoriginal":
        return GINoriginal(
            in_channels=input_dim,
            hidden_dim=hidden_dim,
            num_layers=4,
            out_dim=1,
            train_eps=False,
        )

    if model_key == "GToriginal":
        pe_dim = RWSE_DIM if HAS_RWSE else 0
        return GToriginal(
            in_channels=input_dim,
            hidden_dim=hidden_dim,
            num_layers=6,
            num_heads=4,
            out_dim=1,
            pe_dim=pe_dim,
            dropout=0.0,
        )

    if model_key == "GPSoriginal":
        if not HAS_RWSE:
            raise RuntimeError("GPSoriginal requires AddRandomWalkPE (data.pe), but HAS_RWSE=False.")
        return GPSoriginal(
            in_dim=input_dim,
            channels=hidden_dim,
            pe_dim=RWSE_DIM,
            num_layers=6,
            attn_type="multihead",
            attn_kwargs={"dropout": 0.0},
        )

    if model_key == "PNAOriginal":
        if deg is None:
            raise ValueError("PNAOriginal requires deg.")
        return PNAOriginal(deg=deg)

    raise ValueError(f"Unknown model_key: {model_key}")


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


def load_source_training_setup(model_key: str, source_dataset: str):
    """
    Load the SOURCE training split in the same way training was done,
    because we need:
      - input_dim
      - deg for PNAOriginal
      - same 73k subsampling for 5-component datasets
    """
    loader = graphbench.Loader(root=DATA_ROOT, dataset_names=[source_dataset])
    splits = loader.load()[0]

    pe_transform = None
    if model_key in ["GToriginal", "GPSoriginal"] and HAS_RWSE:
        pe_transform = AddRandomWalkPE(walk_length=RWSE_DIM, attr_name="pe")

    train_ds = FilteredDataset(splits["train"], f"{source_dataset}-train", transform=pe_transform)

    # replicate training-time 73k subsampling on 5-component datasets
    if "5_" in source_dataset and len(train_ds) > 73000:
        rng = torch.Generator().manual_seed(SEED)
        indices = torch.randperm(len(train_ds), generator=rng)[:73000].tolist()
        train_ds = torch.utils.data.Subset(train_ds, indices)
        print(f"  subsampled source train to {len(train_ds)} samples")

    sample = train_ds[0]
    x = sample.x.squeeze(1) if sample.x.dim() == 3 else sample.x
    input_dim = x.size(-1)

    deg = compute_deg(train_ds) if model_key == "PNAOriginal" else None
    return input_dim, deg


def load_target_test_loader(model_key: str, target_dataset: str, batch_size: int, device):
    loader = graphbench.Loader(root=DATA_ROOT, dataset_names=[target_dataset])
    splits = loader.load()[0]

    pe_transform = None
    if model_key in ["GToriginal", "GPSoriginal"] and HAS_RWSE:
        pe_transform = AddRandomWalkPE(walk_length=RWSE_DIM, attr_name="pe")

    test_ds = FilteredDataset(splits["test"], f"{target_dataset}-test", transform=pe_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return test_loader


def checkpoint_path(model_key: str, source_dataset: str):
    model_folder = os.path.join(ROOT, f"results_{SEED}_{model_key.lower()}", source_dataset)
    run_dir = latest_run_dir(model_folder)
    if run_dir is None:
        return None
    ckpt = os.path.join(run_dir, "best_model.pt")
    return ckpt if os.path.isfile(ckpt) else None


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

    table = {}

    for model_key in MODELS:
        print("\n" + "=" * 100)
        print(f"MODEL: {model_key}")
        print("=" * 100)

        table[model_key] = {}

        for source_dataset, target_datasets in TRANSFER_TASKS.items():
            print(f"\nLoading source checkpoint from: {source_dataset}")

            ckpt = checkpoint_path(model_key, source_dataset)
            if ckpt is None:
                print(f"  [WARNING] No checkpoint found for {model_key} on {source_dataset}")
                for target_dataset in target_datasets:
                    table[model_key][pretty_col(source_dataset, target_dataset)] = None
                continue

            input_dim, deg = load_source_training_setup(model_key, source_dataset)
            model = build_model(model_key=model_key, input_dim=input_dim, hidden_dim=HIDDEN_DIM, deg=deg).to(device)

            state = torch.load(ckpt, map_location=device, weights_only=True)
            model.load_state_dict(state)

            print(f"  Loaded checkpoint: {ckpt}")

            for target_dataset in target_datasets:
                print(f"  Evaluating on target dataset: {target_dataset}")
                test_loader = load_target_test_loader(model_key, target_dataset, BATCH_SIZE, device)
                test_rse = eval_rse(model, test_loader, evaluator, device)
                col = pretty_col(source_dataset, target_dataset)
                table[model_key][col] = test_rse
                print(f"    {col}: {test_rse:.6f}")

    # Build dataframe in fixed column order
    cols = [
        "5_eff→7_eff",
        "5_eff→10_eff",
        "5_vout→7_vout",
        "5_vout→10_vout",
    ]
    df = pd.DataFrame.from_dict(table, orient="index")
    df = df[cols]

    print("\n" + "=" * 100)
    print("GENERALIZATION TEST RSE TABLE")
    print("=" * 100)
    print(df.to_string())
    print("=" * 100)

    # Save CSV
    csv_path = os.path.join(ROOT, "summary_generalization_test_rse.csv")
    df.to_csv(csv_path)
    print(f"\nCSV saved to: {csv_path}")

    # LaTeX
    latex_table = df.round(4).to_latex(
        index=True,
        caption="Generalization test RSE. Each model is trained on 5-component circuits and evaluated on larger circuits of the same target type.",
        label="tab:generalization_results",
        column_format="l" + "c" * len(df.columns),
        escape=False,
    )

    print("\n" + "=" * 100)
    print("LATEX TABLE")
    print("=" * 100)
    print(latex_table)
    print("=" * 100)

    tex_path = os.path.join(ROOT, "summary_generalization_test_rse.tex")
    with open(tex_path, "w") as f:
        f.write(latex_table)
    print(f"\nLaTeX saved to: {tex_path}")


if __name__ == "__main__":
    main()

