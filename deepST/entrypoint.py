import sys
import os
from pathlib import Path
import scanpy as sc
import pandas as pd
import deepstkit as dt
import json
import shutil


CONFIG_PATH = Path(sys.argv[1])
with open(CONFIG_PATH, "r") as f:
        config = json.load(f)


WORKSPACE = Path("/workspace")

# ========== Configuration ==========
DATA_DIR = WORKSPACE / "input"
SAMPLE_ID = "" 
RESULTS_DIR = WORKSPACE / "outputs"


# ========== Initialize Analysis ==========
# Set random seed and initialize DeepST
dt.utils_func.seed_torch(seed=config["seed"])

# Create DeepST instance with analysis parameters
deepst = dt.main.run(
    save_path=RESULTS_DIR,
    task="Identify_Domain",
    pre_epochs=config["pre_epochs"],
    epochs=config["epochs"],
    use_gpu=True
)

# ========== Data Loading & Preprocessing ==========
# (Optional) Load spatial transcriptomics data (Visium platform)
# e.g. adata = anndata.read_h5ad("*.h5ad"), this data including .obsm['spatial']
adata = deepst._get_adata(
    platform=config["platform"],
    data_path=DATA_DIR,
    data_name=SAMPLE_ID,
    verbose=False
)

if config["use_morphological"]:
    adata = deepst._get_image_crop(
            adata, 
            data_name=SAMPLE_ID,
            cnn_type=config["cnn_type"],
            pca_n_comps=config["img_pca_n_comps"])

    IMAGE_CROP_PATH = RESULTS_DIR / "Image_crop"
    if IMAGE_CROP_PATH.exists():
        shutil.rmtree(IMAGE_CROP_PATH)

# ========== Feature Engineering ==========
# Data augmentation (skip morphological if no H&E)
adata = deepst._get_augment(
    adata,
    spatial_type=config["spatial_type"],
    neighbour_k=config["neighbour_k"],
    spatial_k=config["spatial_k"],
    n_components=config["n_components"],
    use_morphological = config["use_morphological"]  # Set True if using H&E features
)

# Construct spatial neighborhood graph
graph_dict = deepst._get_graph(
    adata.obsm["spatial"],
    k=config["k_graph"],
    distType=config["distType"],        # Spatial relationship modeling
    rad_cutoff=config["rad_cutoff"]    # Radius cutoff for 'Radius' method
)

# Dimensionality reduction
data = deepst._data_process(
    adata,
    pca_n_comps=config["pca_n_comps"]
)

# ========== Model Training ==========
# Train DeepST model and obtain embeddings
deepst_embed = deepst._fit(
    data=data,
    graph_dict=graph_dict,
    conv_type=config["conv_type"],          
    linear_encoder_hidden=config["linear_encoder_hidden"], 
    linear_decoder_hidden=config["linear_decoder_hidden"],
    conv_hidden=config["conv_hidden"],
    p_drop=config["p_drop"],
    dec_cluster_n=config["dec_cluster_n"],
    kl_weight=config["kl_weight"],
    mse_weight=config["mse_weight"],
    bce_kld_weight=config["bce_kld_weight"],
    domain_weight=config["domain_weight"],
)
adata.obsm["DeepST_embed"] = deepst_embed

# ========== Spatial Domain Detection ==========
# Cluster spots into spatial domains
adata = deepst._get_cluster_data(
    adata,
    n_domains=config["n_domains"],
    priori=True
)



RESULTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame({
    "barcode": adata.obs_names,
    "domain": adata.obs["DeepST_refine_domain"].astype(int)
})

df.to_csv(RESULTS_DIR / "predictions.csv", index=False)

embed_df = pd.DataFrame(
    deepst_embed,
    index=adata.obs_names,
    columns=[f"DeepST_dim_{i+1}" for i in range(deepst_embed.shape[1])]
)

embed_df.to_csv(RESULTS_DIR / "embeddings.csv", index=True)