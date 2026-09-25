import os
import json
import hashlib
import numpy as np
import pandas as pd

DATA_SEED = 20260925
LABEL_SEED = 20260926
N_PATIENTS = 200
SLICES_PER_PATIENT = 8

def compute_sha256(filepath):
    sha = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha.update(chunk)
    return sha.hexdigest()

def generate_dataset():
    os.makedirs("data/slices", exist_ok=True)
    
    # Label generation
    label_rng = np.random.default_rng(LABEL_SEED)
    labels = np.array([0]*100 + [1]*100)
    label_rng.shuffle(labels)
    
    records = []
    manifest = []
    
    for p_idx in range(N_PATIENTS):
        patient_id = f"NSCLC_P{p_idx:03d}"
        study_id = f"{patient_id}_ST00"
        patient_label = labels[p_idx]
        
        # Patient stream rule: SeedSequence([data_seed, patient_index])
        p_seq = np.random.SeedSequence([DATA_SEED, p_idx])
        p_rng = np.random.default_rng(p_seq)
        
        # Persistent patient anatomy shortcut
        patient_texture = p_rng.normal(0.5, 0.1, size=(2, 64, 64)).astype(np.float32)
        
        for s_idx in range(SLICES_PER_PATIENT):
            slice_id = f"{patient_id}_S{s_idx:02d}"
            array_path = f"data/slices/{slice_id}.npy"
            
            # Add slice variation
            noise = p_rng.normal(0, 0.05, size=(2, 64, 64)).astype(np.float32)
            slice_array = patient_texture + noise
            
            # Apply synthetic signal if label is 1
            if patient_label == 1:
                slice_array[:, 30:34, 30:34] += 0.2
                
            # Clip to finite values in [0,1]
            slice_array = np.clip(slice_array, 0.0, 1.0)
            np.save(array_path, slice_array, allow_pickle=False)
            
            records.append({
                "slice_id": slice_id,
                "patient_id": patient_id,
                "study_id": study_id,
                "slice_index": s_idx,
                "array_path": array_path,
                "label": int(patient_label),
                "generator_version": "1.0.0"
            })
            
            manifest.append({
                "array_path": array_path,
                "sha256": compute_sha256(array_path)
            })

    # Sort and save metadata
    df = pd.DataFrame(records)
    df.sort_values(by=["patient_id", "slice_index"], inplace=True)
    df.to_csv("data/metadata.csv", index=False)
    
    # Sort and save image manifest
    manifest.sort(key=lambda x: x["array_path"])
    with open("data/image_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Generated {len(df)} samples across {N_PATIENTS} patients.")

if __name__ == "__main__":
    generate_dataset()