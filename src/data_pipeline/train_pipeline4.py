import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

def npz_generator(npz_path):
    if npz_path.endswith(".npz"):
        data = np.load(npz_path)
        for img, label in zip(data["images"], data["labels"]):
            yield img, label
    else:
        for file in sorted(os.listdir(npz_path)):
            if file.endswith(".npz"):
                data = np.load(os.path.join(npz_path, file))
                for img, label in zip(data["images"], data["labels"]):
                    yield img, label

def load_dataset(npz_path, batch_size=32, shuffle_buffer=2048, repeat=False):
    ds = tf.data.Dataset.from_generator(
        lambda: npz_generator(npz_path),
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(14,), dtype=tf.float32)
        )
    )
    ds = ds.shuffle(buffer_size=shuffle_buffer)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def train_two_phase_model(
    model_name: str,
    base_model_builder,
    conditional_dir: str,
    full_dir: str,
    val_dir: str,
    model_save_dir: str,
    phase1_epochs: int = 5,
    phase2_epochs: int = 5,
    batch_size: int = 32
):
    import os

    os.makedirs(model_save_dir, exist_ok=True)
    phase1_path = os.path.join(model_save_dir, f"{model_name}.phase1.weights.h5")
    final_model_path = os.path.join(model_save_dir, f"{model_name}.final.keras")

    # Load validation set once
    val_ds = load_dataset(val_dir, batch_size=batch_size)

    # --- Phase 1 ---
    print(f"===== Phase 1: Conditional training for {model_name} =====")
    model = base_model_builder()
    train_ds = load_dataset(conditional_dir, batch_size=batch_size, repeat=True)
    steps_per_epoch = 10

    model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["AUC"])
    model.fit(train_ds, validation_data=val_ds, epochs=5, steps_per_epoch=steps_per_epoch)


    # model.fit(train_ds, validation_data=val_ds, epochs=phase1_epochs)
    model.save_weights(phase1_path)

    # --- Phase 2 ---
    print(f"\n===== Phase 2: Fine-tuning on full dataset for {model_name} =====")
    model = base_model_builder()
    model.load_weights(phase1_path)
    for layer in model.layers[:-1]:
        layer.trainable = False

    train_ds = load_dataset(full_dir, batch_size=batch_size)

    model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["AUC"])
    model.fit(train_ds, validation_data=val_ds, epochs=phase2_epochs)
    model.save(final_model_path)

    print(f"\n Finished training {model_name}, saved to {final_model_path}")
    return model

def build_densenet121_model():
    base_model = DenseNet121(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    out = Dense(14, activation="sigmoid")(x)
    return Model(inputs=base_model.input, outputs=out)

train_two_phase_model(
    model_name="densenet121",
    base_model_builder=build_densenet121_model,
    conditional_dir="/content/drive/MyDrive/CheXpert/output/conditional_train_all.npz",
    full_dir="/content/drive/MyDrive/CheXpert/output/train",
    val_dir="/content/drive/MyDrive/CheXpert/output/val",
    model_save_dir="/content/drive/MyDrive/CheXpert/models",
    phase1_epochs=5,
    phase2_epochs=5,
    batch_size=32
)

from tensorflow.keras.applications import DenseNet169

def build_densenet169_model():
    base_model = DenseNet169(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    out = Dense(14, activation="sigmoid")(x)
    return Model(inputs=base_model.input, outputs=out)

train_two_phase_model(
    model_name="densenet169",
    base_model_builder=build_densenet169_model,
    conditional_dir="/content/drive/MyDrive/CheXpert/output/conditional_train_all.npz",
    full_dir="/content/drive/MyDrive/CheXpert/output/train",
    val_dir="/content/drive/MyDrive/CheXpert/output/val",
    model_save_dir="/content/drive/MyDrive/CheXpert/models",
    phase1_epochs=5,
    phase2_epochs=5,
    batch_size=32
)

from tensorflow.keras.applications import DenseNet201

def build_densenet201_model():
    base_model = DenseNet201(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    out = Dense(14, activation="sigmoid")(x)
    return Model(inputs=base_model.input, outputs=out)

train_two_phase_model(
    model_name="densenet201",
    base_model_builder=build_densenet201_model,
    conditional_dir="/content/drive/MyDrive/CheXpert/output/conditional_train_all.npz",
    full_dir="/content/drive/MyDrive/CheXpert/output/train",
    val_dir="/content/drive/MyDrive/CheXpert/output/val",
    model_save_dir="/content/drive/MyDrive/CheXpert/models",
    phase1_epochs=5,
    phase2_epochs=5,
    batch_size=32
)

from tensorflow.keras.applications import InceptionResNetV2

def build_inceptionresnetv2_model():
    base_model = InceptionResNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    out = Dense(14, activation="sigmoid")(x)
    return Model(inputs=base_model.input, outputs=out)

train_two_phase_model(
    model_name="inceptionresnetv2",
    base_model_builder=build_inceptionresnetv2_model,
    conditional_dir="/content/drive/MyDrive/CheXpert/output/conditional_train_all.npz",
    full_dir="/content/drive/MyDrive/CheXpert/output/train",
    val_dir="/content/drive/MyDrive/CheXpert/output/val",
    model_save_dir="/content/drive/MyDrive/CheXpert/models",
    phase1_epochs=5,
    phase2_epochs=5,
    batch_size=32
)

from tensorflow.keras.applications import Xception

def build_xception_model():
    base_model = Xception(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    out = Dense(14, activation="sigmoid")(x)
    return Model(inputs=base_model.input, outputs=out)

train_two_phase_model(
    model_name="xception",
    base_model_builder=build_xception_model,
    conditional_dir="/content/drive/MyDrive/CheXpert/output/conditional_train_all.npz",
    full_dir="/content/drive/MyDrive/CheXpert/output/train",
    val_dir="/content/drive/MyDrive/CheXpert/output/val",
    model_save_dir="/content/drive/MyDrive/CheXpert/models",
    phase1_epochs=5,
    phase2_epochs=5,
    batch_size=32
)

from tensorflow.keras.applications import NASNetLarge

def build_nasnetlarge_model():
    base_model = NASNetLarge(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    out = Dense(14, activation="sigmoid")(x)
    return Model(inputs=base_model.input, outputs=out)

train_two_phase_model(
    model_name="nasnetlarge",
    base_model_builder=build_nasnetlarge_model,
    conditional_dir="/content/drive/MyDrive/CheXpert/output/conditional_train_all.npz",
    full_dir="/content/drive/MyDrive/CheXpert/output/train",
    val_dir="/content/drive/MyDrive/CheXpert/output/val",
    model_save_dir="/content/drive/MyDrive/CheXpert/models",
    phase1_epochs=5,
    phase2_epochs=5,
    batch_size=32
)