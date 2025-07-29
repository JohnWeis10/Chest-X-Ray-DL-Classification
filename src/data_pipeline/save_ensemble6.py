from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Average

from google.colab import drive
drive.mount('/content/drive')

def load_and_rename_model(path, name):
    original = load_model(path)
    new_input = Input(shape=(224, 224, 3), name=f"{name}_input")
    new_output = original(new_input)
    return Model(inputs=new_input, outputs=new_output, name=name)

m1 = load_and_rename_model("/content/drive/MyDrive/CheXpert/models/densenet121.final.keras", "densenet121")
m2 = load_and_rename_model("/content/drive/MyDrive/CheXpert/models/densenet169.final.keras", "densenet169")
m3 = load_and_rename_model("/content/drive/MyDrive/CheXpert/models/densenet201.final.keras", "densenet201")
m4 = load_and_rename_model("/content/drive/MyDrive/CheXpert/models/inceptionresnetv2.final.keras", "inceptionresnetv2")
m5 = load_and_rename_model("/content/drive/MyDrive/CheXpert/models/xception.final.keras", "xception")
m6 = load_and_rename_model("/content/drive/MyDrive/CheXpert/models/nasnetlarge.final.keras", "nasnetlarge")

# Sanity check: all models must accept same input shape
input_tensor = Input(shape=(224, 224, 3), name="ensemble_input")

# Feed the same input into all models
outputs = [m(input_tensor) for m in [m1, m2, m3, m4, m5, m6]]

# Average their predictions
avg_output = Average()(outputs)

# Build the ensemble model
ensemble_model = Model(inputs=input_tensor, outputs=avg_output)
ensemble_model.summary()

#Save Model
ensemble_model.save("/content/drive/MyDrive/CheXpert/models/ensemble_model.keras")