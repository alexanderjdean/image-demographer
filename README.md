# image-demographer

This is a project that uses the [UTKFace](https://www.kaggle.com/datasets/jangedoo/utkface-new) dataset from Kaggle, a dataset that consists of over $20,000$ face images with annotations of age, gender, and ethnicity. The dataset is pulled using Kaggle's API in ```script.py```.

The model built is a Convolutional Neural Network (CNN) in ```build.py``` that predicts the age and gender of an individual from the given categories defined in the original dataset. Only binary gender is predicted with a rough degree of accuracy (85-90%), with age being considerably more difficult to predict.