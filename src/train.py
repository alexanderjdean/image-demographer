import os
import pandas as pd
from PIL import Image
from build import get_features

images = []
age_labels = []
gender_labels = []

for file_name in os.listdir("data/"):
    age = file_name.split('_')[0]
    gender = file_name.split('_')[1]

    images.append("data/" + file_name)
    age_labels.append(age)
    gender_labels.append(gender)

dataset = pd.DataFrame()
dataset['image'], dataset['age'], dataset['gender'] = images, age_labels, gender_labels
genders = {0 : "Male", 1 : "Female"}

X = get_features(dataset['image'])
X = X / 255.0
