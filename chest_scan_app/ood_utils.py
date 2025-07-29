import os
import numpy as np
from scipy.spatial.distance import mahalanobis
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# Get absolute path to this script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OOD_DIR = os.path.join(BASE_DIR, "ood")

mean_vec = np.load(os.path.join(OOD_DIR, "mean_vec.npy"))
cov_inv = np.load(os.path.join(OOD_DIR, "cov_inv.npy"))
threshold = np.load(os.path.join(OOD_DIR, "distance_threshold.npy"))

def build_feature_model():
    base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    x = base_model.get_layer("conv3_block4_out").output
    x = GlobalAveragePooling2D()(x)
    return Model(inputs=base_model.input, outputs=x)

feature_model = build_feature_model()

def is_in_distribution(image: np.ndarray) -> bool:
    feat = feature_model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
    dist = mahalanobis(feat, mean_vec, cov_inv)
    return dist <= threshold