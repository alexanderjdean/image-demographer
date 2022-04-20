import os
import pandas as pd
import numpy as np
from build import get_features, build_model, IMAGE_SIZE, log_training

images = []
age_labels = []
gender_labels = []

for file_name in os.listdir("data/"):
    age = file_name.split('_')[0]
    gender = file_name.split('_')[1]

    images.append("data/" + file_name)
    age_labels.append(int(age))
    gender_labels.append(int(gender))

dataset = pd.DataFrame()
dataset['image'], dataset['age'], dataset['gender'] = images, age_labels, gender_labels
genders = {0 : "Male", 1 : "Female"}

X = get_features(dataset['image'])
X = X / 255
Y_gender = np.array(dataset['gender'])
Y_age = np.array(dataset['age'])

model = build_model((IMAGE_SIZE, IMAGE_SIZE, 1))
history = model.fit(x=X, y=[Y_gender, Y_age], batch_size=32, epochs=30, validation_split=0.2, 
            callbacks=[log_training()])

print("The model evaluated the gender of an individual with an accuracy of " + 
    str(history.history['gender_out_accuracy'][-1] * 100) + "%")