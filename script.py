import os
import shutil
import subprocess
import random
import numpy as np

def partition_files(files, current_path, test_set=False):
    for file_name in files:
        age = file_name.partition('_')[0]

        if not test_set: new_path = "data/training_set/" + age
        else: new_path = "data/test_set/" + age
        
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            
        shutil.move(current_path + file_name, new_path)

subprocess.call(['sh', './shells/get_data.sh'])
directory = os.fsencode("data/")
files = os.listdir("data/")
os.makedirs("data/training_set")
os.makedirs("data/test_set")

print("Creating the training set of images and test set of images...")

TRAINING_NUM = int(np.ceil(len(files) * 0.8))
TEST_NUM =  len(files) - TRAINING_NUM

print("There will be " + str(TRAINING_NUM) + " images in the training set...")
print("There will be " + str(TEST_NUM) + " images in the test set...")

files_moved = 0

for file_name in random.sample(files, len(files)):
    if files_moved < TRAINING_NUM:
        shutil.move("data/" + file_name, "data/training_set/")
    else: 
        shutil.move("data/" + file_name, "data/test_set/")

    files_moved += 1


print("Partitioning images by age of individual in the training set...")
partition_files(os.listdir("data/training_set"), "data/training_set/", test_set=False)

print("Partitioning images by age of individual in the test set...")
partition_files(os.listdir("data/test_set"), "data/test_set/", test_set=True)

print("Done! :)")