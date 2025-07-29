# Mount Google Drive and Check GPU
from google.colab import drive
import tensorflow as tf

# Mount your Google Drive
drive.mount('/content/drive')

# Confirm GPU
print("Available GPU(s):", tf.config.list_physical_devices('GPU'))

# Imports
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from sklearn.covariance import EmpiricalCovariance
from scipy.spatial.distance import mahalanobis
import numpy as np
import os

# Paths and Configs
TRAIN_DIR = "/content/drive/MyDrive/CheXpert/output/train"
CLEAN_SAVE_PATH = "/content/drive/MyDrive/CheXpert/output/clean_train_data"
BATCH_SIZE = 32
EPOCHS = 5
NUM_CLASSES = 14

# ResNet50 model
def build_multilabel_resnet(input_shape=(224, 224, 3), num_classes=14):
    base_model = ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

# .npz data loader
def npz_generator(npz_dir):
    for file in sorted(os.listdir(npz_dir)):
        if file.endswith(".npz"):
            path = os.path.join(npz_dir, file)
            data = np.load(path)
            for img, label in zip(data["images"], data["labels"]):
                yield img, label

def load_dataset(npz_dir, batch_size=32, shuffle_buffer=2048):
    ds = tf.data.Dataset.from_generator(
        lambda: npz_generator(npz_dir),
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(14,), dtype=tf.float32)
        )
    )
    ds = ds.shuffle(buffer_size=shuffle_buffer)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# Train simple resnet50 model
def train_model():
    ds = load_dataset(TRAIN_DIR, BATCH_SIZE)
    total_batches = tf.data.experimental.cardinality(ds).numpy()
    train_ds = ds.take(int(0.9 * total_batches))
    val_ds = ds.skip(int(0.9 * total_batches))

    model = build_multilabel_resnet()
    model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
    return model

def streamed_frodo_filter_memory_safe(feature_model, train_dir, save_path, percentile=99, batch_size=32):

    os.makedirs(CLEAN_SAVE_PATH, exist_ok=True)
    feature_list = []
    file_map = []

    # Extract features per file (keeps RAM low)
    for file in sorted(os.listdir(train_dir)):
        if not file.endswith(".npz"):
            continue
        data = np.load(os.path.join(train_dir, file))
        images = data["images"]
        feats = feature_model.predict(images, batch_size=batch_size)
        feature_list.append(feats)
        file_map.append((file, feats.shape[0]))

    all_features = np.concatenate(feature_list, axis=0)
    print(f"Extracted features shape: {all_features.shape}")

    # Fit Mahalanobis model
    cov_model = EmpiricalCovariance()
    cov_model.fit(all_features)
    cov_inv = np.linalg.inv(cov_model.covariance_)

    # Compute distances
    distances = []
    for feats in feature_list:
        for f in feats:
            dist = mahalanobis(f, cov_model.location_, cov_inv)
            distances.append(dist)

    threshold = np.percentile(distances, percentile)
    print(f"FRODO threshold ({percentile}th percentile): {threshold:.4f}")

    # Reprocess and save filtered samples per file
    dist_index = 0
    for file, count in file_map:
        data = np.load(os.path.join(train_dir, file))
        images = data["images"]
        labels = data["labels"]
        feats = all_features[dist_index:dist_index + count]

        keep_imgs = []
        keep_lbls = []

        for img, lbl, f in zip(images, labels, feats):
            dist = mahalanobis(f, cov_model.location_, cov_inv)
            if dist <= threshold:
                keep_imgs.append(img)
                keep_lbls.append(lbl)

        dist_index += count

        if keep_imgs:
            out_file = os.path.join(CLEAN_SAVE_PATH, f"clean_{file}")
            np.savez_compressed(out_file, images=np.array(keep_imgs), labels=np.array(keep_lbls))
            print(f"Saved {len(keep_imgs)} clean samples to {out_file}")

model = train_model()
# model_save_path = "/content/drive/MyDrive/CheXpert/models/resnet50_multilabel.h5"
# model.save(model_save_path)

# model = load_model("/content/drive/MyDrive/CheXpert/models/resnet50_multilabel.h5")
x = model.get_layer("conv3_block4_out").output
x = GlobalAveragePooling2D()(x)
feature_model = Model(inputs=model.input, outputs=x)

streamed_frodo_filter_memory_safe(feature_model, TRAIN_DIR, CLEAN_SAVE_PATH)