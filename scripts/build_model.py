import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers, Model


def build_model(weights_folder):
    image_height, image_width = 256, 256

    # Load the pre-trained InceptionV3 model
    pre_trained_model = InceptionV3(input_shape=(image_height, image_width, 3),
                                    weights='imagenet',
                                    include_top=False)

    # Freeze the weights of the pre-trained layers
    for layer in pre_trained_model.layers:
        layer.trainable = False

    last_layer = pre_trained_model.get_layer('mixed7')
    last_output = last_layer.output

    # Create the additional layers
    x = layers.Flatten()(last_output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    # Create the model
    model = Model(pre_trained_model.input, x)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Load the saved weights of the additional layers
    additional_layers_weights = []
    for i in range(4):  # Get the last 4 layers (Flatten, Dense, Dropout, Dense)
        layer_weights = []
        for j in range(len(model.layers[-4:][i].get_weights())):
            weight = np.load(f'{weights_folder}/additional_layer_{i}_weight_{j}.npy')
            layer_weights.append(weight)
        additional_layers_weights.append(layer_weights)

    # Set the weights of the additional layers
    model.layers[-4].set_weights(additional_layers_weights[0])
    model.layers[-3].set_weights(additional_layers_weights[1])
    model.layers[-2].set_weights(additional_layers_weights[2])
    model.layers[-1].set_weights(additional_layers_weights[3])

    return model


if __name__ == '__main__':
    weights_folder = '../app/model/weights'
    model = build_model(weights_folder=weights_folder)
    model.summary()
    print('Model built successfully!')
    # Save the model to the models folder
    model.save('../models/Pneumonia_ROC_0975_cutoff_024.keras')
    print('Model saved successfully!')
