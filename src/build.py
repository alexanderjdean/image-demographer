from keras.preprocessing.image import load_img
from keras.models import Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input
import tensorflow as tf
import datetime
import numpy as np

IMAGE_SIZE = 128

def get_features(images):
    features = []

    for image in images:
        feature = load_img(image, color_mode="grayscale")
        feature = feature.resize((IMAGE_SIZE, IMAGE_SIZE))
        feature = np.array(feature)
        features.append(feature)
    
    features = np.array(features)
    features = features.reshape(len(features), IMAGE_SIZE, IMAGE_SIZE, 1)
    return features

def build_model(input_shape):
    inputs = Input((input_shape))

    conv_1 = Conv2D(32, kernel_size=(3, 3), activation='relu') (inputs)
    maxp_1 = MaxPooling2D(pool_size=(2, 2)) (conv_1)
    
    conv_2 = Conv2D(64, kernel_size=(3, 3), activation='relu') (maxp_1)
    maxp_2 = MaxPooling2D(pool_size=(2, 2)) (conv_2)

    conv_3 = Conv2D(128, kernel_size=(3, 3), activation='relu') (maxp_2)
    maxp_3 = MaxPooling2D(pool_size=(2, 2)) (conv_3)

    conv_4 = Conv2D(256, kernel_size=(3, 3), activation='relu') (maxp_3)
    maxp_4 = MaxPooling2D(pool_size=(2, 2)) (conv_4)

    flatten = Flatten() (maxp_4)

    dense_1 = Dense(256, activation='relu') (flatten)
    dense_2 = Dense(256, activation='relu') (flatten)

    dropout_1 = Dropout(0.3) (dense_1)
    dropout_2 = Dropout(0.3) (dense_2)

    output_1 = Dense(1, activation='sigmoid', name='gender_out') (dropout_1)
    output_2 = Dense(1, activation='relu', name='age_out') (dropout_2)

    model = Model(inputs=[inputs], outputs=[output_1, output_2])
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer='adam', metrics=['accuracy'])
    return model

class log_training(tf.keras.callbacks.Callback):
  def on_train_batch_begin(self, batch, logs=None):
    print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_train_batch_end(self, batch, logs=None):
    print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

  def on_test_batch_begin(self, batch, logs=None):
    print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_test_batch_end(self, batch, logs=None):
    print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))