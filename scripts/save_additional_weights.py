import numpy as np
from tensorflow.keras.models import load_model

model_path = '../models/Pneumonia_ROC_0975_cutoff_024.keras'
model = load_model(model_path)

# model.summary()

# Save the weights of the additional layers
for i, layer in enumerate(model.layers[-4:]):  # Get the last 4 layers (Flatten, Dense, Dropout, Dense)
    weights = layer.get_weights()
    for j, weight in enumerate(weights):
        np.save(f'../models/additional_layer_{i}_weight_{j}.npy', weight)

print('Weights saved successfully!')