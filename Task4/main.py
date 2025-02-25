import zipfile
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras import Sequential, losses
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, LeakyReLU as LR, Reshape, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Lambda, GlobalAveragePooling2D
preFeat_archive = 'pretrain_features.csv.zip'
preFeat_dir = 'pretrain_features.csv'
preLab_archive = 'pretrain_labels.csv.zip'
preLab_dir = 'pretrain_labels.csv'
trainFeat_archive = 'train_features.csv.zip'
treainFeat_dir = 'train_features.csv'
trainLab_archive = 'train_labels.csv.zip'
trainLab_dir = 'train_labels.csv'
testFeat_archive = 'test_features.csv.zip'
testFeat_dir = 'test_features.csv'
sample_dir = 'sample.csv'
#unzip file to target location and return that location
def unzip(archive_dir, target_dir):
    with zipfile.ZipFile(archive_dir) as zip_file:
        if os.path.exists(target_dir) or os.path.isfile(target_dir):
            print('Unzipped file already exists.')
        else:
            zip_file.extractall()
            print(archive_dir,'successfully unziped!')
    return target_dir

df_preFeat = pd.read_csv(unzip(preFeat_archive,preFeat_dir))
df_preLab = pd.read_csv(unzip(preLab_archive,preLab_dir))
df_trainFeat = pd.read_csv(unzip(trainFeat_archive,treainFeat_dir))
df_trainLab = pd.read_csv(unzip(trainLab_archive,trainLab_dir))
df_testFeat = pd.read_csv(unzip(testFeat_archive,testFeat_dir))
df_sample = pd.read_csv(sample_dir)


df_preFeat.shape
df_preLab.shape
df_trainFeat.shape
df_trainLab.shape
df_testFeat.shape
df_sample.shape
X_pre = df_preFeat.drop(['Id', 'smiles'], axis=1).values
y_pre = df_preLab.drop(['Id'], axis=1).values
X_train = df_trainFeat.drop(['Id', 'smiles'], axis=1).values
y_train = df_trainLab.drop(['Id'], axis=1).values
X_test = df_testFeat.drop(['Id', 'smiles'], axis=1).values
indexs = df_testFeat['Id'].values
base_model = Sequential()
base_model.add(Dense(500, input_shape=(X_pre.shape[1],), activation='relu'))
base_model.add(Dense(200, activation='relu'))
base_model.add(Dense(50, activation='relu'))
base_model.add(Dense(5, activation='relu'))
base_model.add(Dense(1))

base_model.summary()
optimizer = tf.keras.optimizers.RMSprop(0.0001)
base_model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
base_model.fit(X_pre, y_pre, epochs=10, validation_split=0.2)
base_model.trainable = False
add_model = Sequential()
add_model.add(Dense(20, activation='relu', input_shape=(base_model.output_shape[1:])))
add_model.add(Dense(1))
model2 = Model(base_model.input, add_model(base_model.output))
model2.summary()
model2.compile(optimizer=optimizer,
              loss='mse',
              metrics=['mae'])
model2.fit(X_train, y_train, epochs=2000, validation_split=0.2)
y_t = model2.predict(X_test)
print(y_t[0:10])
header = df_sample.columns
predictions = np.c_[indexs, y_t]
dr = pd.DataFrame(data=predictions, columns=header)
dr.to_csv('output.csv', index=False, float_format='%.3f')