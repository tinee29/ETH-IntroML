import glob
import zipfile
import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Lambda

img_archive = 'food.zip'
img_dir = 'food'
res_img_dir = './resized_food/'
features_file = 'features.pckl'
train_triplets = 'train_triplets.txt'
test_triplets = 'test_triplets.txt'
submission_file = 'output.txt'
IMG_SIZE = [299,299]
input_shape = (299,299,3)
def preprocess_file(archive_dir, target_dir):
    with zipfile.ZipFile(img_archive) as zip_file:
        if os.path.exists(img_dir) or os.path.isfile(img_dir):
            print('Unzipped file already exists and renamed.')
        else:
            zip_file.extractall()
            print(archive_dir,'successfully unziped!')
            print('Renaming image files...')
            curr_dir = os.getcwd()
            os.chdir(target_dir)
            for index, oldfile in enumerate(glob.glob("*.jpg"), start=0):
                newfile = '{}.jpg'.format(index)
                os.rename (oldfile,newfile)
            os.chdir(curr_dir)
            print('files successfully renamed!')
            
    
def get_gen(dir_name, batch_size):
    curr_idx = 0
    size = 10000
    while True:
        batch = []
        while len(batch) < batch_size:
            img_name= dir_name + '/' + str(int(curr_idx)) + ".jpg"
            img = load_img(img_name)
            img = img_to_array(img)
            img = tf.image.resize_with_pad(img,IMG_SIZE[0],IMG_SIZE[1],antialias=True)
            img = tf.keras.applications.inception_resnet_v2.preprocess_input(img)
            batch.append(img)
            curr_idx = (curr_idx + 1) % size

        batch = np.array(batch)
        labels = np.zeros(batch_size)

        try:
            yield batch, labels
        except StopIteration:
            return
def extract_features():
    
    img_gen = get_gen(img_dir,1)
    print('Extracting features...')
    # resnet feature extraction
    
    resnet_inception = tf.keras.applications.InceptionResNetV2(pooling='avg',include_top=False)
    # restnet takes care of features extraction
    resnet_inception.trainable = False

    # Declare input
    x = x_in = Input(shape=input_shape)
    x = resnet_inception(x)

    # Get the whole model
    model = Model(inputs=x_in, outputs=x)
    
    x_feat = model.predict(img_gen,steps=10000)
    print('Features successfully extracted!')
    return x_feat
def buildTripletTensor(features, triplets_file, gen_labels=False):
    if(gen_labels):
        print("Generating training features tensor...")
    else:
        print("Generating test features tensor...")
    # Import pandas
    triplets_df = pd.read_csv(triplets_file, delim_whitespace=True, header=None, names=["A", "B", "C"])
    # Features tensor
    train_tensors = []
    # Labels
    labels = []
    # Number of triplets in the file
    num_triplets = len(triplets_df)

    for i in range(num_triplets):
        # Get triplet
        triplet = triplets_df.iloc[i]
        A, B, C = triplet['A'], triplet['B'], triplet['C']
        # Get features
        tensor_a = features[A]
        tensor_b = features[B]
        tensor_c = features[C]
        # Concatenete
        triplet_tensor = np.concatenate((tensor_a, tensor_b, tensor_c), axis=-1)
        if(gen_labels):
            reverse_triplet_tensor = np.concatenate((tensor_a, tensor_c, tensor_b), axis=-1)
            # Add to train tensors
            train_tensors.append(triplet_tensor)
            labels.append(1)
            train_tensors.append(reverse_triplet_tensor)
            labels.append(0)
        else:
            train_tensors.append(triplet_tensor)
        print('Triplets generated: {}/{}'.format(i+1,num_triplets),end="\r")


    train_tensors = np.array(train_tensors)
    if(gen_labels):
        labels = np.array(labels)
        print("Training feature tensors generated!")
        return train_tensors, labels
    else:
        print("Testing feature tensors generated!")
        return train_tensors

def model_train_and_predict(x_train,y,x_test):

    print("Building model...")
    # Build model to process features
    x = x_in = Input(x_train.shape[1:])
    x = Activation('relu')(x)
    x = Dropout(0.7)(x)
    x = Dense(1152)(x)
    x = Activation('relu')(x)
    x = Dense(288)(x)
    x = Activation('relu')(x)
    x = Dense(72)(x)
    x = Activation('relu')(x)
    x = Dense(18)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=x_in, outputs=x)
    print("Compiling model...")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train model
    print("Training model...")
    model.fit(x_train, y, epochs=10)
    print("Training completed!")

    # Predict
    print("Making inference...")
    y_test = model.predict(x_test)
    return y_test
res_img = preprocess_file(img_archive,img_dir)
features = extract_features()
train_tensor, lables = buildTripletTensor(features, train_triplets, gen_labels=True)
test_tensor = buildTripletTensor(features, test_triplets, gen_labels=False)
y_t = model_train_and_predict(train_tensor,lables,test_tensor)
# Create submission file
print("Genrating submission file...")
y_test_thresh = np.where(y_t < 0.5, 0, 1)
np.savetxt(submission_file, y_test_thresh, fmt='%d')
print("Submission file generated! Done.")