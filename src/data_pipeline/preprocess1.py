import argparse
import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Constants
IMAGE_SIZE = 224
RESIZE_SIZE = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
DISEASE_LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
    'Fracture', 'Support Devices'
]

def load_and_preprocess_image(path, template):
    #resizing
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    # 256x256
    image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #maybe implementing autoencoding

    #template matching
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    x, y = max_loc
    cropped = image[y:y+IMAGE_SIZE, x:x+IMAGE_SIZE]

    if cropped.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        cropped = cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE))

    # normalize based on ImageNet training set
    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = ((rgb - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)
    return normalized


def apply_label_smoothing(labels, a=0.55, b=0.85):
    # 'U+Ones+LSR' label smoothing for -1 labels. Describes as the best results in the paper
    smoothed = []
    for row in labels:
        new_row = []
        for val in row:
            if val == -1:
                new_row.append(np.random.uniform(a, b))
            else:
                new_row.append(val)
        smoothed.append(new_row)
    return np.array(smoothed, dtype=np.float32)

def main(args):
    df = pd.read_csv(args.labels_csv)
    # check that dataset is reading correctly
    print("Available columns in CSV:", df.columns.tolist())

    # filter for valid scan type
    df = df[df['AP/PA'] == 'AP']
    # filter out rows with missing image paths
    df.dropna(subset=['Path'], inplace=True)

    # parse image paths
    image_rel_paths = df['Path'].tolist()
    # set valid labels
    labels = df[DISEASE_LABELS].fillna(0).astype(np.float32).values
    # confirm image paths look correct
    print("First image path example:", os.path.join(args.image_root, df['Path'].iloc[0]))

    # Load my template
    template = cv2.imread(args.template_image, cv2.IMREAD_GRAYSCALE)
    #check valid template
    if template is None or template.shape != (224, 224):
        raise ValueError("Invalid template. Must be 224x224 grayscale.")

    # prepare output folders
    os.makedirs(args.preprocessed_train, exist_ok=True)
    os.makedirs(args.preprocessed_test, exist_ok=True)
    os.makedirs(args.preprocessed_val, exist_ok=True)
    #how many images to process at once
    BATCH_SIZE = 1000
    images = []
    filtered_labels = []
    batch_index = 0

    # save one batch of data to train/test/val .npz files
    def save_batch(imgs, lbls, batch_idx):
        imgs = np.array(imgs, dtype=np.float32)
        lbls = apply_label_smoothing(lbls)

        # First split into train_val and test (80% / 20%)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            imgs, lbls, test_size=0.2, random_state=42
        )

        # Then split train_val into train and val (87.5% / 12.5% of original, which becomes 70/10 overall)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.125, random_state=42
        )

        np.savez_compressed(os.path.join(args.preprocessed_train, f"train_batch_{batch_idx}.npz"),
                            images=X_train, labels=y_train)

        np.savez_compressed(os.path.join(args.preprocessed_test, f"test_batch_{batch_idx}.npz"),
                            images=X_test, labels=y_test)

        np.savez_compressed(os.path.join(args.preprocessed_val, f"val_batch_{batch_idx}.npz"),
                            images=X_val, labels=y_val)

    for i, rel_path in enumerate(image_rel_paths):
        full_path = os.path.join(args.image_root, rel_path)
        try:
            #image preprocessing
            img = load_and_preprocess_image(full_path, template)
            # add image to current batch
            images.append(img)
            # add label to current batch
            filtered_labels.append(labels[i])
        except Exception as e:
            # log skipped image
            print(f"Skipping {full_path}: {e}")

        if len(images) >= BATCH_SIZE:
            # write batch to disk
            save_batch(images, filtered_labels, batch_index)
            print(f"Saved batch {batch_index} with {len(images)} images.")
            # reset for next batch
            images, filtered_labels = [], []
            batch_index += 1

    # save final batch (if it has any images)
    if images:
        save_batch(images, filtered_labels, batch_index)
        print(f"Saved final batch {batch_index} with {len(images)} images.")

if __name__ == "__main__":
    import types
    args = types.SimpleNamespace(
        labels_csv="../../CheXpert-v1.0/train_cheXbert(subset).csv",
        image_root="../../",
        template_image="../../CheXpert-v1.0/Template.jpg",
        preprocessed_train="output/train",
        preprocessed_test="output/test",
        preprocessed_val="output/val"
    )
    main(args)

def preprocess_image_for_inference(image_path: str, template: np.ndarray) -> np.ndarray:
    """
    Preprocess a single uploaded image for inference:
    - Resize to 256x256
    - Template match to crop 224x224 region
    - Normalize using ImageNet stats
    Returns:
        np.ndarray shape (224, 224, 3)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image at path: {image_path}")

    # Resize and grayscale for template matching 256x256
    image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Template match to find best crop
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    x, y = max_loc
    cropped = image[y:y+IMAGE_SIZE, x:x+IMAGE_SIZE]

    # If crop goes out of bounds, pad and recrop
    if cropped.shape[0] < IMAGE_SIZE or cropped.shape[1] < IMAGE_SIZE:
        padded = cv2.copyMakeBorder(image, 0, IMAGE_SIZE, 0, IMAGE_SIZE, cv2.BORDER_CONSTANT, value=0)
        cropped = padded[y:y+IMAGE_SIZE, x:x+IMAGE_SIZE]

    if cropped.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        cropped = cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE))

    # Normalize to ImageNet stats
    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = ((rgb - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)

    return normalized