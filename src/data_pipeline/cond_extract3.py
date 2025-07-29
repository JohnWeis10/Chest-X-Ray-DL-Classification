import os
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

INPUT_DIR = "/content/drive/MyDrive/CheXpert/output/clean_train_data"
OUTPUT_DIR = "/content/drive/MyDrive/CheXpert/output/conditional_train"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Column indices for relevant conditions
LUNG_OPACITY_IDX = 3
CONSOLIDATION_IDX = 6
ENLARGED_CARDIOMEDIASTINUM_IDX = 1

for file in sorted(os.listdir(INPUT_DIR)):
    if not file.endswith(".npz"):
        continue
    data = np.load(os.path.join(INPUT_DIR, file))
    images = data["images"]
    labels = data["labels"]

    # Boolean mask where all 3 target labels == 1
    mask = (
        (labels[:, LUNG_OPACITY_IDX] == 1) &
        (labels[:, CONSOLIDATION_IDX] == 1) &
        (labels[:, ENLARGED_CARDIOMEDIASTINUM_IDX] == 1)
    )

    filtered_images = images[mask]
    filtered_labels = labels[mask]

    if len(filtered_images) > 0:
        out_path = os.path.join(OUTPUT_DIR, f"cond_{file}")
        np.savez_compressed(out_path, images=filtered_images, labels=filtered_labels)
        print(f"Saved {len(filtered_images)} conditional samples to {out_path}")