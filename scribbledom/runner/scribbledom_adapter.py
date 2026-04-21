#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd
import shutil


def create_symlink(source, target):
    """
    Makes 'target' point to 'source'. 
    If 'target' is a directory, it must be removed first.
    """
    target = Path(target)
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target) # Remove the tool's default directory
    
    target.parent.mkdir(parents=True, exist_ok=True)
    
    target.symlink_to(source)
    print(f"[ADAPTER] Linked Tool Internal: {target} -> Backend: {source}")


def prepare_inputs(config, tool_root: Path):
    mounted_input_dir = Path("/input")
    
    tool_input_dir = (
        tool_root / 
        config["space_ranger_output_directory"] / 
        config["dataset"] / 
        config["samples"][0]
    )
    
    tool_input_dir.mkdir(parents=True, exist_ok=True)

    h5_candidates = list(mounted_input_dir.glob("*.h5")) + list(mounted_input_dir.glob("*.hdf5"))
    if not h5_candidates:
        raise FileNotFoundError("No .h5 or .hdf5 file found in /input")
    
    provided_h5_file = h5_candidates[0]
    tool_expected_name = f"{config['samples'][0]}_filtered_feature_bc_matrix.h5"

    # Create the renamed .h5 link
    create_symlink(provided_h5_file, tool_input_dir / tool_expected_name)

    # 4. everything else (Efficiently)
    for item in mounted_input_dir.iterdir():
        if item == provided_h5_file:
            continue

        create_symlink(source=item, target=tool_input_dir / item.name)



def prepare_outputs(config, workspace_dir: Path, tool_root: Path):
    """Links tool internal output folders to the backend's mounted output volume."""
    workspace_output_dir = workspace_dir / "outputs"
    workspace_output_dir.mkdir(parents=True, exist_ok=True)

    scribbledom_output_dir = (
        tool_root / 
        config["final_output_folder"] / 
        config["dataset"] / 
        config["samples"][0] /
        config["schema"]
    )

    create_symlink(workspace_output_dir, scribbledom_output_dir)

    print("[ADAPTER] Output symlinks established.")



def prepare_scribble(config, tool_root: Path):        
    """Converts backend JSON annotations to the specific CSV format the tool needs."""
    if config.get("schema") != "expert":
        return

    annotation_file = Path('/annotation/annotations.json')
    if not annotation_file.exists():
        print("[ADAPTER] Skipping scribble: annotation.json not found.")
        return

    with open(annotation_file) as f:
        data = json.load(f)
    
    df = pd.DataFrame([
        {"barcode": i["barcode"], "label": i["label_id"]} 
        for i in data.get("labels", [])
    ])
    
    prep_folder_name = config.get("preprocessed_data_folder", "preprocessed_data")
    save_path = tool_root / prep_folder_name / config["dataset"] / config["samples"][0]
    save_path.mkdir(parents=True, exist_ok=True)
    
    output_file = save_path / "manual_scribble.csv"
    df.to_csv(output_file, index=False)

    print(f"[ADAPTER] Scribble CSV generated at {output_file}")


def rename_output_files(workspace_dir: Path):
    """Renames tool output files to match backend expectations."""
   
    workspace_output_dir = workspace_dir / "outputs"

    tool_prediction_file = workspace_output_dir / "final_barcode_labels.csv"
    tool_embeddings_file = workspace_output_dir / "final_barcode_embeddings.csv"

    expected_prediction_file = workspace_output_dir / "predictions.csv"
    expected_embeddings_file = workspace_output_dir / "embeddings.csv"

    if not tool_prediction_file.exists() or not tool_embeddings_file.exists():
        raise FileNotFoundError("Expected output files not found in the tool's output directory.")
    
    tool_prediction_file.rename(expected_prediction_file)
    tool_embeddings_file.rename(expected_embeddings_file)