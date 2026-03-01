import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
import os
import json
import graphbench
from datetime import datetime
import time

from models.baseline_mlp import BaselineMLP


# ---- Reproducibility ----
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

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

def run(dataset_name):
    start_time = time.time()
    # ---- Config ----
   
    root          = "./graphbench_data"
    epochs        = 2        # change to 700 for full run
    batch_size    = 512
    lr            = 1e-3
    hidden_dim    = 384
    model_name = "BaselineMLP"
    # save_dir      = f"./results/{dataset_name}"
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir   = f"./results/{dataset_name}/{model_name}_{timestamp}_seed{SEED}"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- Load + filter ----
    Loader = graphbench.Loader(root=root, dataset_names=[dataset_name])
    splits = Loader.load()[0]

    train_ds = FilteredDataset(splits["train"], "train")
    train_bad = train_ds.bad 
    # FILTER ONLY 73K FOR THE 5_EFF DATASET
    if "5_" in dataset_name and len(train_ds) > 73000:
        rng = torch.Generator().manual_seed(SEED)
        indices = torch.randperm(len(train_ds), generator=rng)[:73000].tolist()
        train_ds = torch.utils.data.Subset(train_ds, indices)
        print(f"  subsampled train to {len(train_ds)} samples")

    valid_ds = FilteredDataset(splits["valid"], "valid")
    test_ds  = FilteredDataset(splits["test"],  "test")

    print(f"Split sizes — train: {len(train_ds)}  valid: {len(valid_ds)}  test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # ---- Model ----
    sample    = train_ds[0]
    x         = sample.x.squeeze(1) if sample.x.dim() == 3 else sample.x
    input_dim = x.size(-1)

    model = BaselineMLP(input_dim=input_dim, hidden_dim=hidden_dim, out_channels=1).to(device)
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
    }
    with open(f"{save_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {save_dir}/results.json")



def main():
     # ---- Config ----
    datasets = [
        "electronic_circuits_5_eff",
        "electronic_circuits_5_vout",
        "electronic_circuits_7_eff",
        "electronic_circuits_10_eff",
        "electronic_circuits_10_vout",
    ]
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Training on: {dataset_name}")
        print(f"{'='*60}")
        run(dataset_name)

    
if __name__ == "__main__":
    main()