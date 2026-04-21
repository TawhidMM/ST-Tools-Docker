import csv
import json
import uuid
from pathlib import Path

def convert_csv_to_scribble_json(csv_path, output_json_path):
    labels = []
    
    # Open the CSV. Assuming the format provided: ,cluster.init
    # and subsequent lines: BARCODE,LABEL
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        # Skip header
        next(reader)
        
        for row in reader:
            if len(row) < 2:
                continue
            
            barcode = row[0].strip()
            # The CSV shows values after the comma, we map those to label_id
            label_id = row[1].strip() if row[1].strip() else None

            if label_id != "label_1":
                print("hola")
            
            labels.append({
                "barcode": barcode,
                "label_id": label_id,
                "label_name": f"Region_{label_id}" # Random label name
            })

    # Construct the final schema
    data = {
        "annotation_id": str(uuid.uuid4()),
        "dataset_id": "test_dataset_2026",
        "labels": labels
    }

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

    print(f"Successfully converted {csv_path} to {output_json_path}")

# Execute
convert_csv_to_scribble_json('/mnt/Drive E/Class Notes/L-4 T-2/ScribbleDom/preprocessed_data/Human_DLPFC/151507/manual_scribble.csv', 'annotation.json')