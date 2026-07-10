#!/usr/bin/env python3
import sys
import subprocess
import json
from pathlib import Path
import  scribbledom_adapter




def run_command(command, cwd=None):
    print(f"\n[RUNNING] {' '.join(command)}\n")
    subprocess.run(command, cwd=cwd, check=True)



def main():
   
    config_file = Path('/workspace/config/config.json')

    with open(config_file) as f:
        config = json.load(f)

    # Job root is usually two levels up from the config file in the mounted volume
    workspace_dir = config_file.parent.parent.resolve()
    scribbledom_root = Path("/opt/ScribbleDom")

    # -----------------------------------------------------
    # 1. Run Adaptation (The Hook)
    # -----------------------------------------------------
    scribbledom_adapter.prepare_inputs(config, scribbledom_root)
    scribbledom_adapter.prepare_config(config, config_file)
    scribbledom_adapter.prepare_outputs(config, workspace_dir, scribbledom_root)
    scribbledom_adapter.prepare_scribble(config, scribbledom_root)

    # -----------------------------------------------------
    # 2. Execution Logic
    # -----------------------------------------------------
    try:
        commands = [
            ["Rscript", "get_genex_data_from_10x_h5.R", str(config_file)],
            ["python3", "visium_data_to_matrix_representation_converter.py", "--params", str(config_file)]
        ]

        
        if config["schema"] == "expert":
            commands.append(["python3", "scribble_dom.py", "--params", str(config_file)])
        else:
            commands.append(["python3", "autoscribble_dom.py", "--params", str(config_file)])

        commands.extend([
            ["python3", "best_model_estimator.py", "--params", str(config_file)],
            ["python3", "show_results.py", "--params", str(config_file)]
        ])

        for cmd in commands:
            run_command(cmd, cwd=scribbledom_root)

        print("\n[SUCCESS] Pipeline completed.\n")

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed: {e}\n")
        sys.exit(1)


    # -----------------------------------------------------
    # 3. Rename outputs according to backeend contract
    # -----------------------------------------------------
    scribbledom_adapter.rename_output_files(workspace_dir)

if __name__ == "__main__":

    print("\n[INFO] Starting ScribbleDom pipeline...\n")

    main()