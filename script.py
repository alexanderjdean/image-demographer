import os
import shutil
import subprocess

subprocess.call(['sh', './shells/get_data.sh'])
directory = os.fsencode("data/")
files = os.listdir("data/")

print("Partitioning images by age of individual...")

for file_name in files:
    age = file_name.partition('_')[0]
    new_path = "data/" + age
    
    if not os.path.exists(new_path):
        os.makedirs(new_path)
        
    shutil.move("data/" + file_name, new_path)

print("Done! :)")