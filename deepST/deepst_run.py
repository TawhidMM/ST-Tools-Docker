# run_deepst.py
import os
import scanpy as sc
import matplotlib.pyplot as plt
import deepstkit as dt
import numpy as np

# ────────────────────────────── CHANGE THESE ──────────────────────────────
H5AD_PATH       = "/data/starmap.h5ad"          # ← your file name
OUTPUT_FOLDER   = "/data/deepst_results"
N_DOMAINS       = 7
USE_MORPHOLOGY  = False
# ──────────────────────────────────────────────────────────────────────────

SEED = 0
dt.utils_func.seed_torch(SEED)

print("Loading data...")
adata = sc.read_h5ad(H5AD_PATH)
print(adata)

if 'spatial' not in adata.obsm:
    raise ValueError("No 'spatial' coordinates in .h5ad → cannot continue")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("\nAugmentation...")
adata = dt.augment_adata(
    adata,
    use_morphological=USE_MORPHOLOGY,
    spatial_type="KDTree",
    spatial_k=15,
    neighbour_k=4,
    adjacent_weight=0.25
)

print("Building graph...")
graph_dict = dt.adj.graph(
    adata.obsm["spatial"],
    distType="KDTree",
    k=10
).main()

print("Initializing DeepST runner...")
runner = dt.main.run(
    save_path=OUTPUT_FOLDER,
    task="Identify_Domain",
    pre_epochs=6,
    epochs=5,
    use_gpu=True
)

print("Preprocessing data...")
processed_data = runner._data_process(adata, pca_n_comps=200)

print("Training DeepST model...")
embeddings = runner._fit(
    data=processed_data,
    graph_dict=graph_dict
)

adata.obsm["DeepST_embed"] = embeddings

print("Clustering...")
adata = runner._get_cluster_data(
    adata,
    n_domains=N_DOMAINS,
    priori=True,
    shape="square"
)

print("Plotting...")
fig, ax = plt.subplots(figsize=(8, 6))
sc.pl.spatial(
    adata,
    color="DeepST_refine_domain",
    frameon=False,
    spot_size=50,
    ax=ax,
    show=False
)
plt.title(f"DeepST domains (n={N_DOMAINS})")
plt.savefig(os.path.join(OUTPUT_FOLDER, "domains.pdf"), dpi=300, bbox_inches='tight')
plt.close()


print("\nFinished.")
print("Results in:", OUTPUT_FOLDER)
print("• domains.pdf")
print("• umap.pdf")
print("• adata_final.h5ad")



"""


docker run --rm \
  -v ./mount:/data \
  -v ./deepst_run.py:/app/run_deepst.py \
  --gpus all \
  --entrypoint "" \
  deepst:latest \
  python /app/run_deepst.py



"""
