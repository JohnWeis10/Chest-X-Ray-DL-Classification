from tensorflow.keras.models import load_model
import os

from google.colab import drive
drive.mount('/content/drive')

model_paths = {
    "densenet121": "/content/drive/MyDrive/CheXpert/models/densenet121.final.keras",
    "densenet169": "/content/drive/MyDrive/CheXpert/models/densenet169.final.keras",
    "densenet201": "/content/drive/MyDrive/CheXpert/models/densenet201.final.keras",
    "inceptionresnetv2": "/content/drive/MyDrive/CheXpert/models/inceptionresnetv2.final.keras",
    "xception": "/content/drive/MyDrive/CheXpert/models/xception.final.keras",
    "nasnetlarge": "/content/drive/MyDrive/CheXpert/models/nasnetlarge.final.keras"
}

models = {name: load_model(path) for name, path in model_paths.items()}

# Load all test data once
def load_all_test_images(npz_dir):
    all_images = []
    all_labels = []
    for file in sorted(os.listdir(npz_dir)):
        if file.endswith(".npz"):
            data = np.load(os.path.join(npz_dir, file))
            all_images.append(data["images"])
            all_labels.append(data["labels"])
    return np.concatenate(all_images), np.concatenate(all_labels)

test_dir = "/content/drive/MyDrive/CheXpert/output/test"
X_test, y_test = load_all_test_images(test_dir)

import numpy as np

def ensemble_predict(models, images):
    preds = []
    for name, model in models.items():
        p = model.predict(images, verbose=0)
        preds.append(p)
    avg_preds = np.mean(preds, axis=0)
    return avg_preds

from sklearn.metrics import roc_auc_score

ensemble_preds = ensemble_predict(models, X_test)
y_test_bin = (y_test >= 0.5).astype(int)

for i in range(14):
    try:
        auc = roc_auc_score(y_test_bin[:, i], ensemble_preds[:, i])
        print(f"Class {i} AUC: {auc:.3f}")
    except ValueError as e:
        print(f"Class {i} AUC: ERROR - {str(e)}")

mean_auc = roc_auc_score(y_test_bin, ensemble_preds, average="macro")
print(f"\nMean AUC across 14 classes: {mean_auc:.3f}")

# ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
chexpert_eval_indices = [8, 2, 6, 5, 10]
y_test_bin = (y_test >= 0.5).astype(int)

auc_scores = []
for i in chexpert_eval_indices:
    try:
        auc = roc_auc_score(y_test_bin[:, i], ensemble_preds[:, i])
        print(f"Class {i} AUC: {auc:.3f}")
        auc_scores.append(auc)
    except ValueError as e:
        print(f"Class {i} AUC: ERROR - {str(e)}")

mean_auc = np.mean(auc_scores)
print(f"\nMean AUC across CheXpert 5 classes: {mean_auc:.3f}")