echo "Starting now. Downloading the data set..."
kaggle datasets download -d jangedoo/utkface-new
unzip utkface-new.zip

echo "Removing unnecessary folders and files..."
rm -r utkface-new.zip
rm -r utkface_aligned_cropped
rm -r crop_part1

echo "Renaming the data file..."
mv UTKFace/ data/

echo "Training and evaluating the model..."
python src/main.py

echo "Removing all training data..."
rm -r data/