from keras.preprocessing.image import load_img
import numpy as np

def get_features(images):
    features = [None for _ in images]

    for i, image in enumerate(images):
        feature = load_img(image).resize((64, 64))
        feature = np.array(feature)
        features[i] = feature
    
    features = np.array(features)
    features = features.reshape(len(features), 64, 64, 1)