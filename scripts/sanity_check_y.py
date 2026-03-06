import graphbench
import torch
from collections import Counter
import math

# ========= CONFIG =========
root = "./graphbench_data"

DATASETS = [
    "electronic_circuits_5_eff",
    "electronic_circuits_5_vout",
    "electronic_circuits_7_eff",
    "electronic_circuits_7_vout",
    "electronic_circuits_10_eff",
    "electronic_circuits_10_vout",
]

# How many graphs to scan per split for the heavy checks:
MAX_TRAIN_GRAPHS = 5000
MAX_VALID_GRAPHS = 2000
MAX_TEST_GRAPHS  = 2000

# If you want full-split exact stats, set these to None (can be slower)
# MAX_TRAIN_GRAPHS = None
# MAX_VALID_GRAPHS = None
# MAX_TEST_GRAPHS  = None


# ========= HELPERS =========
def safe_x(g):
    x = getattr(g, "x", None)
    if x is None:
        return None
    if x.dim() == 3 and x.size(1) == 1:
        x = x.squeeze(1)
    return x


def summarize_list(vals, name):
    vals_t = torch.tensor(vals, dtype=torch.float)
    print(f"{name}:")
    print(
        f"  mean={vals_t.mean().item():.6f} std={vals_t.std(unbiased=False).item():.6f} "
        f"min={vals_t.min().item():.6f} max={vals_t.max().item():.6f}"
    )


def graph_stats(ds, split_name, max_graphs=None):
    num_graphs = len(ds) if max_graphs is None else min(len(ds), max_graphs)

    num_nodes_list = []
    num_edges_list = []
    densities = []
    y_scalar_list = []

    has_edge_attr = 0
    has_self_loops = 0
    undirected_count = 0
    duplicate_edge_graphs = 0

    x_shapes = Counter()
    x_unique_values = set()
    x_row_sums = Counter()
    node_type_counts = Counter()
    extra_fields = Counter()

    for i in range(num_graphs):
        g = ds[i]

        # record available fields
        for key in g.keys():
            if key not in ["x", "edge_index", "edge_attr", "y", "batch", "ptr"]:
                extra_fields[key] += 1

        x = safe_x(g)
        edge_index = g.edge_index
        y = g.y.view(-1).float()

        n = g.num_nodes
        e = edge_index.size(1)

        num_nodes_list.append(n)
        num_edges_list.append(e)

        # IMPORTANT: do NOT blindly y.mean() if multi-target.
        # For circuits, y is usually scalar, but we handle both.
        if y.numel() == 1:
            y_scalar_list.append(y.item())
        else:
            y_scalar_list.append(y[0].item())  # store first target dim for summary

        # density for directed graph interpretation
        if n > 1:
            densities.append(e / (n * (n - 1)))
        else:
            densities.append(0.0)

        # edge_attr presence
        if getattr(g, "edge_attr", None) is not None:
            has_edge_attr += 1

        # self-loops
        src, dst = edge_index
        if (src == dst).any().item():
            has_self_loops += 1

        # duplicate edges
        edge_tuples = list(zip(src.tolist(), dst.tolist()))
        if len(edge_tuples) != len(set(edge_tuples)):
            duplicate_edge_graphs += 1

        # undirectedness check: every (u,v) has (v,u)
        edge_set = set(edge_tuples)
        is_undirected = all((v, u) in edge_set for (u, v) in edge_set)
        if is_undirected:
            undirected_count += 1

        # node feature checks
        if x is not None:
            x_shapes[tuple(x.shape)] += 1
            x_unique_values.update(torch.unique(x).cpu().tolist())

            # row sums
            row_sums = x.sum(dim=-1)
            for val in row_sums.cpu().tolist():
                x_row_sums[float(val)] += 1

            # if one-hot, extract type id
            if x.dim() == 2:
                for row in x:
                    if torch.all((row == 0) | (row == 1)) and abs(row.sum().item() - 1.0) < 1e-6:
                        node_type = int(row.argmax().item())
                        node_type_counts[node_type] += 1

    print(f"\n=== {split_name} split summary (checked {num_graphs} graphs) ===")
    summarize_list(num_nodes_list, "num_nodes")
    summarize_list(num_edges_list, "num_edges")
    summarize_list(densities, "density")
    summarize_list(y_scalar_list, "target y (scalar view)")

    print("\nEdge structure:")
    print(f"  graphs with edge_attr present: {has_edge_attr}/{num_graphs}")
    print(f"  graphs with self-loops:        {has_self_loops}/{num_graphs}")
    print(f"  graphs fully undirected:       {undirected_count}/{num_graphs}")
    print(f"  graphs with duplicate edges:   {duplicate_edge_graphs}/{num_graphs}")

    print("\nNode features:")
    print("  x shapes (top 10):", x_shapes.most_common(10))
    # unique values can be large; keep it but you can comment out if too spammy
    print("  unique values in x:", sorted(x_unique_values))
    print("  row sums in x (top 10):", x_row_sums.most_common(10))
    print("  node type counts:", dict(sorted(node_type_counts.items())))

    print("\nExtra Data fields:")
    print(" ", dict(extra_fields))


def corrcoef(a, b):
    a = torch.tensor(a, dtype=torch.float)
    b = torch.tensor(b, dtype=torch.float)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.pow(2).sum().sqrt() * b.pow(2).sum().sqrt()).item()
    if denom == 0:
        return float("nan")
    return (a * b).sum().item() / denom


def simple_target_correlations(ds, split_name, max_graphs=5000):
    ys = []
    ns = []
    es = []
    type_count_vectors = []

    num_graphs = min(len(ds), max_graphs)

    for i in range(num_graphs):
        g = ds[i]
        x = safe_x(g)
        y = g.y.view(-1).float()
        y_scalar = y.item() if y.numel() == 1 else y[0].item()

        ys.append(y_scalar)
        ns.append(g.num_nodes)
        es.append(g.edge_index.size(1))

        if x is not None and x.dim() == 2 and x.size(-1) == 9:
            counts = x.sum(dim=0).cpu().tolist()  # counts per one-hot dim
            type_count_vectors.append(counts)

    print(f"\n=== {split_name} simple target correlations (first {num_graphs} graphs) ===")
    print("corr(y, num_nodes):", corrcoef(ys, ns))
    print("corr(y, num_edges):", corrcoef(ys, es))

    if type_count_vectors:
        type_count_vectors = torch.tensor(type_count_vectors, dtype=torch.float)
        ys_t = torch.tensor(ys, dtype=torch.float)
        for j in range(type_count_vectors.size(1)):
            print(f"corr(y, count_type_{j}):", corrcoef(ys_t.tolist(), type_count_vectors[:, j].tolist()))


# ---- NEW: y distribution stats per split (mean/std/var + percentiles) ----
def y_split_stats(ds, split_name, max_graphs=None):
    n = len(ds) if max_graphs is None else min(len(ds), max_graphs)
    ys = []
    for i in range(n):
        g = ds[i]
        y = g.y.detach().view(-1).float().cpu()  # scalar or multi-dim
        ys.append(y)
    Y = torch.stack(ys, dim=0)  # [n, y_dim]

    print(f"\n=== y stats: {split_name} (n={n}, y_dim={Y.size(1)}) ===")
    print("mean:", Y.mean(dim=0).tolist())
    print("std :", Y.std(dim=0, unbiased=False).tolist())
    print("var :", Y.var(dim=0, unbiased=False).tolist())
    print("min :", Y.min(dim=0).values.tolist())
    print("max :", Y.max(dim=0).values.tolist())

    qs = torch.tensor([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    for j in range(Y.size(1)):
        qv = torch.quantile(Y[:, j], qs)
        print(f"percentiles y[{j}] (0,1,5,25,50,75,95,99,100%):", qv.tolist())


# ---- NEW: mean-predictor baseline MSE (the “RSE≈1” anchor) ----
def mean_predictor_baseline(train_ds, eval_ds, train_name="train", eval_name="eval", max_train=None, max_eval=None):
    n_train = len(train_ds) if max_train is None else min(len(train_ds), max_train)
    ys_train = [train_ds[i].y.view(-1).float().cpu() for i in range(n_train)]
    Ytr = torch.stack(ys_train, dim=0)
    mu = Ytr.mean(dim=0)  # [y_dim]

    n_eval = len(eval_ds) if max_eval is None else min(len(eval_ds), max_eval)
    ys_eval = [eval_ds[i].y.view(-1).float().cpu() for i in range(n_eval)]
    Yev = torch.stack(ys_eval, dim=0)

    mse_per_dim = ((Yev - mu) ** 2).mean(dim=0)
    mse_all = mse_per_dim.mean().item()

    print(f"\n=== Mean predictor baseline (train->{eval_name}) ===")
    print(f"train mean (from {train_name}, n={n_train}):", mu.tolist())
    print(f"MSE on {eval_name} (per dim, n={n_eval}):", mse_per_dim.tolist())
    print(f"overall MSE on {eval_name}:", mse_all)


# ---- OPTIONAL: quick check if y is standardized ----
def check_y_standardized(ds, split_name, max_graphs=2000):
    n = min(len(ds), max_graphs)
    Y = torch.stack([ds[i].y.view(-1).float().cpu() for i in range(n)], dim=0)
    mean = Y.mean(dim=0)
    std = Y.std(dim=0, unbiased=False)
    print(f"\n=== y standardization check: {split_name} (n={n}) ===")
    print("mean:", mean.tolist())
    print("std :", std.tolist())
    print("If mean≈0 and std≈1, targets are standardized.")


def inspect_one_sample(ds, split_name):
    g = ds[0]
    x = safe_x(g)

    print(f"\n=== One detailed sample ({split_name}[0]) ===")
    print(g)
    print("keys:", g.keys())
    print("x shape:", None if x is None else tuple(x.shape))
    print("edge_index shape:", tuple(g.edge_index.shape))
    print("edge_attr:", None if getattr(g, "edge_attr", None) is None else tuple(g.edge_attr.shape))
    print("y:", g.y)

    if x is not None:
        print("first 5 node rows:\n", x[:5])
        if x.dim() == 2:
            print("row sums (first 20):", x.sum(dim=-1)[:20])
            if x.size(-1) == 9:
                print("argmax node types (first 20):", x.argmax(dim=-1)[:20])

    print("first 20 edges:\n", g.edge_index[:, :20])


# ========= MAIN LOOP OVER DATASETS =========
for dataset_name in DATASETS:
    print("\n" + "=" * 100)
    print("DATASET:", dataset_name)
    print("=" * 100)

    Loader = graphbench.Loader(root=root, dataset_names=[dataset_name])
    splits = Loader.load()[0]

    train_ds = splits["train"]
    valid_ds = splits["valid"]
    test_ds  = splits["test"]

    print("Split sizes:")
    print("  train:", len(train_ds))
    print("  valid:", len(valid_ds))
    print("  test :", len(test_ds))

    # One sample (train[0]) to see fields
    inspect_one_sample(train_ds, "train")

    # Structural summaries
    graph_stats(train_ds, "train", max_graphs=MAX_TRAIN_GRAPHS)
    graph_stats(valid_ds, "valid", max_graphs=MAX_VALID_GRAPHS)

    # Target distribution stats (this is the key)
    y_split_stats(train_ds, "train", max_graphs=MAX_TRAIN_GRAPHS)
    y_split_stats(valid_ds, "valid", max_graphs=MAX_VALID_GRAPHS)
    y_split_stats(test_ds,  "test",  max_graphs=MAX_TEST_GRAPHS)

    # Mean-predictor baseline MSE: interpret RSE≈1 vs your reported train_mse numbers
    mean_predictor_baseline(
        train_ds, valid_ds,
        train_name="train",
        eval_name="valid",
        max_train=MAX_TRAIN_GRAPHS,
        max_eval=MAX_VALID_GRAPHS,
    )

    # Correlations (optional, but you already had it)
    simple_target_correlations(train_ds, "train", max_graphs=min(MAX_TRAIN_GRAPHS or len(train_ds), 5000))

    # Optional standardized check
    check_y_standardized(train_ds, "train", max_graphs=min(2000, len(train_ds)))
    check_y_standardized(valid_ds, "valid", max_graphs=min(2000, len(valid_ds)))