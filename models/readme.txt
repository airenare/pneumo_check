The final model file is 1.89 GB, so it is not uploaded here.
To build the model, you can run the /scripts/build_model.py script.
It will build a model from InceptionV3, add new layers and weights, and save the model to the /models directory.
The same logic is implemented in the application (app/main.py), but the model is cached and not saved to the disk.