import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
import os
import json
import graphbench
from datetime import datetime
import time
from torch_geometric.utils import degree


from models.baseline_gat import GAT
from models.baseline_gcn import GCN
from models.baseline_gin import GIN
from models.baseline_gt import GT
from models.baseline_gtransformer import GTransformer
from models.baseline_gps import GPS
from models.baseline_pna_graphdoc import PNAgraphdoc
from models.baseline_pna_original import PNAOriginal
from models.baseline_pnaresidual import PNAresidual
from models.baseline_pna_decoder import PNAdecoder

import random
import numpy as np
try:
    from torch_geometric.transforms import AddRandomWalkPE
    HAS_RWSE = True
except ImportError:
    HAS_RWSE = False
    print("WARNING: AddRandomWalkPE not found in your PyG version. GT will run without RWSE.")


# ---- RWSE config (GT only) ----
RWSE_DIM = 16

# ---- Reproducibility ----
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)



class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, name="ds", transform=None):
        self.base = base_ds
        self.transform = transform
        self.idx = []
        self.bad = 0
        self.cache = {}  # key: base dataset index -> cached pe tensor on CPU

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
        base_i = self.idx[i]               # stable identity in underlying dataset
        data = self.base[base_i]

        if self.transform is not None:
            if base_i not in self.cache:
                data = self.transform(data)           # adds data.pe
                self.cache[base_i] = data.pe.cpu()    # cache PE on CPU
            else:
                data.pe = self.cache[base_i]          # restored; batch.to(device) will move it

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
    """
    Instantiate a model. Adjust kwargs if your class signatures differ.
    IMPORTANT: make sure each model returns shape [B], e.g. by .squeeze(-1) in forward().
    """
    if model_key == "GAT":
        # GAT(in_channels=9, hidden_dim=384, num_layers=4, heads=4, out_dim=1)
        return GAT(in_channels=input_dim, hidden_dim=hidden_dim, num_layers=4, heads=4, out_dim=1)

    if model_key == "GCN":
        # GCN(in_channels=9, hidden_dim=384, num_layers=4, out_dim=1)
        return GCN(in_channels=input_dim, hidden_dim=hidden_dim, num_layers=4, out_dim=1)

    if model_key == "GIN":
        # GIN(in_channels=9, hidden_dim=384, num_layers=4, out_dim=1)
        return GIN(in_channels=input_dim, hidden_dim=hidden_dim, num_layers=4, out_dim=1)

    if model_key == "GT":
        # GT(in_channels=9, hidden_dim=384, num_layers=6, num_heads=4, out_dim=1, pe_dim=0)
        pe_dim = RWSE_DIM if HAS_RWSE else 0
        return GT(
            in_channels=input_dim,
            hidden_dim=hidden_dim,
            num_layers=6,
            num_heads=4,
            out_dim=1,
            pe_dim=pe_dim,
        )
    if model_key == "GPS":
        if not HAS_RWSE:
            raise RuntimeError("GPS baseline requires AddRandomWalkPE to create data.pe, but HAS_RWSE=False.")
        pe_dim = RWSE_DIM
        return GPS(
            in_dim=input_dim,          # 9 for circuits
            channels=hidden_dim,       # 384
            pe_dim=pe_dim,             # 16
            num_layers=6,              # choose 6 (or 10 like tutorial, but 6 is fine)
            attn_type="multihead",
            attn_kwargs={"dropout": 0.0},
        )
    
    
    if model_key == "GTransformer":
        # GT(in_channels=9, hidden_dim=384, num_layers=6, num_heads=4, out_dim=1, pe_dim=0)
        pe_dim = RWSE_DIM if HAS_RWSE else 0
        return GTransformer(
            in_channels=input_dim,
            hidden_dim=hidden_dim,
            num_layers=6,
            num_heads=4,
            out_dim=1,
            pe_dim=pe_dim,
        )

    if model_key == "PNAgraphdoc":
        if deg is None:
            raise ValueError("PNA requires deg. Pass deg=compute_deg(train_ds).")
        return PNAgraphdoc(
            in_channels=input_dim,   # 9
            hidden_dim=hidden_dim,   # 384
            num_layers=4,
            deg=deg,
            out_dim=1,
        )
    if model_key == "PNAresidual":
        if deg is None:
            raise ValueError("PNA requires deg. Pass deg=compute_deg(train_ds).")
        return PNAresidual(
            in_channels=input_dim,   # 9
            hidden_dim=hidden_dim,   # 384
            num_layers=4,
            deg=deg,
            out_dim=1,
        )
    if model_key == "PNAOriginal":
        if deg is None:
            raise ValueError("PNAOriginal requires deg.")
        return PNAOriginal(deg=deg)
    
    if model_key == "PNAdecoder":
        if deg is None:
            raise ValueError("PNAdecoder requires deg.")
        return PNAdecoder(deg=deg)
    
    
    
    
    raise ValueError(f"Unknown model_key: {model_key}")
def run(model_key: str, dataset_name: str):
    start_time = time.time()
    # ---- Config ----
   
    root          = "./graphbench_data"
    epochs        = 30        # change to 700 for full run
    batch_size    = 512
    lr            = 1e-3
    hidden_dim    = 384
    model_name = model_key
    # save_dir      = f"./results/{dataset_name}"
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    # master_folder = f"./results_seed_{SEED}"
    results_root = f"./results_{SEED}_{model_key.lower()}"
    save_dir = f"{results_root}/{dataset_name}/{model_key}_{timestamp}_seed{SEED}"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- Load + filter ----
    Loader = graphbench.Loader(root=root, dataset_names=[dataset_name])
    splits = Loader.load()[0]

    pe_transform = None
    if model_key in ["GT", "GPS", "GTransformer"] and HAS_RWSE:
        pe_transform = AddRandomWalkPE(walk_length=RWSE_DIM, attr_name="pe")

    train_ds = FilteredDataset(splits["train"], "train", transform=pe_transform)
    train_bad = train_ds.bad
    # FILTER ONLY 73K FOR THE 5_EFF DATASET
    if "5_" in dataset_name and len(train_ds) > 73000:
        rng = torch.Generator().manual_seed(SEED)
        indices = torch.randperm(len(train_ds), generator=rng)[:73000].tolist()
        train_ds = torch.utils.data.Subset(train_ds, indices)
        print(f"  subsampled train to {len(train_ds)} samples")

    valid_ds = FilteredDataset(splits["valid"], "valid", transform=pe_transform)
    test_ds  = FilteredDataset(splits["test"],  "test",  transform=pe_transform)
    print(f"Split sizes — train: {len(train_ds)}  valid: {len(valid_ds)}  test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # ---- Model ----
    sample    = train_ds[0]
    x         = sample.x.squeeze(1) if sample.x.dim() == 3 else sample.x
    input_dim = x.size(-1)

    deg = compute_deg(train_ds) if model_key in ["PNAgraphdoc", "PNAOriginal", "PNAresidual", "PNAdecoder"] else None
    model = build_model(model_key=model_key, input_dim=input_dim, hidden_dim=hidden_dim, deg=deg).to(device)
    opt   = Adam(model.parameters(), lr=lr)
    evaluator = graphbench.Evaluator("electroniccircuit")

    # ---- Train ----
    best_val_rse = float("inf")
    log = []   # list of dicts, one per epoch

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_graphs = 0.0, 0

        for batch in train_loader:
            batch  = batch.to(device)
            y_pred = model(batch).float()
            y_true = batch.y.view(-1).float()
            loss   = F.mse_loss(y_pred, y_true)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss   += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs

        train_mse = total_loss / max(total_graphs, 1)
        val_rse   = eval_rse(model, valid_loader, evaluator, device)

        # ---- log ----
        log.append({"epoch": epoch, "train_mse": train_mse, "val_rse": val_rse})
        print(f"Epoch {epoch:03d} | train_mse: {train_mse:.6f} | val_RSE: {val_rse:.6f}")

        # ---- save best model ----
        if val_rse < best_val_rse:
            best_val_rse = val_rse
            torch.save(model.state_dict(), f"{save_dir}/best_model.pt")
            print(f"  → new best saved (val_RSE={best_val_rse:.6f})")

    # ---- save last model + logs ----
    torch.save(model.state_dict(), f"{save_dir}/last_model.pt")
    duration = time.time() - start_time 
    print(f"Training done: {duration/60:.2f} min ({duration:.1f} sec)")
    with open(f"{save_dir}/log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nLogs saved to {save_dir}/log.json")

    # ---- Test using best model ----
    model.load_state_dict(torch.load(f"{save_dir}/best_model.pt", weights_only=True))
    test_rse = eval_rse(model, test_loader, evaluator, device)
    print(f"Test RSE (best model): {test_rse:.6f}")

    
    results = {
        "dataset":       dataset_name,
        "model":         model_name,
        "seed":          SEED,
        "epochs":        epochs,
        "batch_size":    batch_size,
        "lr":            lr,
        "hidden_dim":    hidden_dim,
        "best_val_rse":  best_val_rse,
        "test_rse":      test_rse,
        "train_sizes": {
            "original":    len(splits["train"]),              # 234093
            "after_filter": len(splits["train"]) - train_bad, # 234084
            "used":        len(train_ds),                     # 73000 (or 234084 if not subsampled)
        },
        "removed_nan": {
            "train": train_bad,
            "valid": valid_ds.bad,
            "test":  test_ds.bad,
        },

        "duration_min":  round(duration / 60, 2),
        "pe_dim": (RWSE_DIM if (model_key in ["GT", "GPS", "GTransformer"] and HAS_RWSE) else 0),
    }
    with open(f"{save_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {save_dir}/results.json")



def main():
   
     # ---- Config ----
    datasets = [
        "electronic_circuits_5_eff",
        # "electronic_circuits_5_vout",
        "electronic_circuits_7_eff",
        # "electronic_circuits_7_vout",
        # "electronic_circuits_10_eff",
        # "electronic_circuits_10_vout",
    ]
    
    models = ["PNAOriginal"]

    for model_key in models:
        for dataset_name in datasets:
            print(f"\n{'='*80}")
            print(f"Training model={model_key} on dataset={dataset_name}")
            print(f"{'='*80}")
            run(model_key=model_key, dataset_name=dataset_name)


    
if __name__ == "__main__":
    main()