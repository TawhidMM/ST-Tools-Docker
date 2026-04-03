import sys
import os
from pathlib import Path

sys.path.append('/opt/STAIG')

import sys
import os
import yaml
import random
import warnings

warnings.filterwarnings("ignore")

import torch
import numpy as np
import scanpy as sc
import pandas as pd

from staig.adata_processing import LoadSingle10xAdata
from staig.staig import STAIG


# -----------------------------
# Paths
# -----------------------------
WORKSPACE = Path("/workspace")
CONFIG_PATH = Path(sys.argv[1])


# -----------------------------
# Load config (YAML only)
# -----------------------------
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

INPUT_PATH = WORKSPACE / config["input_path"]
OUTPUT_PATH = WORKSPACE / config["output_path"]

# -----------------------------
# Determinism
# -----------------------------
seed = int(config.get("seed", 42))

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)



class Args:
    label = True
args = Args()    
# -----------------------------
# Load data (Space Ranger dir)
# -----------------------------
adata = LoadSingle10xAdata(
    path=INPUT_PATH,
    n_neighbors=config["num_neigh"],
    n_top_genes=config["num_gene"],
    image_emb=True,
    label=args.label,
).run()


# -----------------------------
# Train + evaluate
# -----------------------------
staig = STAIG(
    args=args,
    config=config,
    single=False
)

staig.adata = adata
staig.train()
staig.eva()
staig.cluster(label=args.label)



df = pd.DataFrame({
    "barcode": adata.obs_names,
    "domain": adata.obs["domain"].astype(int)
})
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH / "predictions.csv", index=False)

embeddings = adata.obsm['emb']

embed_df = pd.DataFrame(
    embeddings,
    index=adata.obs_names,
    columns=[f"Staig_dim_{i+1}" for i in range(embeddings.shape[1])]
)

embed_df.to_csv(OUTPUT_PATH / "embeddings.csv", index=True)