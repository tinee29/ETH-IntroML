import zipfile
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

# File paths for input data
preFeat_archive = 'pretrain_features.csv.zip'
preFeat_dir = 'pretrain_features.csv'
preLab_archive = 'pretrain_labels.csv.zip'
preLab_dir = 'pretrain_labels.csv'
trainFeat_archive = 'train_features.csv.zip'
trainFeat_dir = 'train_features.csv'
trainLab_archive = 'train_labels.csv.zip'
trainLab_dir = 'train_labels.csv'
testFeat_archive = 'test_features.csv.zip'
testFeat_dir = 'test_features.csv'
sample_dir = 'sample.csv'

# Function to unzip files
def unzip(archive_dir, target_dir):
    """
    Unzips the provided archive and returns the target directory.
    If the file already exists, it skips the extraction.
    """
    with zipfile.ZipFile(archive_dir) as zip_file:
        if os.path.exists(target_dir) or os.path.isfile(target_dir):
            print('Unzipped file already exists.')
        else:
            zip_file.extractall()
            print(f'{archive_dir} successfully unzipped!')
    return target_dir

# Unzip and load data
df_preFeat = pd.read_csv(unzip(preFeat_archive, preFeat_dir))  # Pre-trained features
df_preLab = pd.read_csv(unzip(preLab_archive, preLab_dir))      # Pre-trained labels
df_trainFeat = pd.read_csv(unzip(trainFeat_archive, trainFeat_dir))  # Training features
df_trainLab = pd.read_csv(unzip(trainLab_archive, trainLab_dir))      # Training labels
df_testFeat = pd.read_csv(unzip(testFeat_archive, testFeat_dir))      # Test features
df_sample = pd.read_csv(sample_dir)  # Sample submission file

# Prepare data for training and testing
X_pre = df_preFeat.drop(['Id', 'smiles'], axis=1).values  # Pre-trained features (input)
y_pre = df_preLab.drop(['Id'], axis=1).values             # Pre-trained labels (output)
X_train = df_trainFeat.drop(['Id', 'smiles'], axis=1).values  # Training features (input)
y_train = df_trainLab.drop(['Id'], axis=1).values             # Training labels (output)
X_test = df_testFeat.drop(['Id', 'smiles'], axis=1).values    # Test features (input)
indexs = df_testFeat['Id'].values  # IDs for test molecules

# Build the base model
base_model = Sequential([
    Dense(500, input_shape=(X_pre.shape[1],)),  # Input layer
    Dense(200, activation='relu'),             # Hidden layer 1
    Dense(50, activation='relu'),              # Hidden layer 2
    Dense(5, activation='relu'),               # Hidden layer 3
    Dense(1)                                   # Output layer
])

# Compile the base model
optimizer = tf.keras.optimizers.RMSprop(0.0001)
base_model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

# Train the base model on pre-trained data
print("Training the base model...")
base_model.fit(X_pre, y_pre, epochs=10, validation_split=0.2)

# Freeze the base model for fine-tuning
base_model.trainable = False

# Add a new model on top of the frozen base model
add_model = Sequential([
    Dense(20, activation='relu', input_shape=(base_model.output_shape[1:])),  # Additional layer
    Dense(1)  # Output layer
])

# Combine the base model and the additional model
model2 = Model(base_model.input, add_model(base_model.output))

# Compile the combined model
model2.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Fine-tune the model on training data
print("Fine-tuning the model...")
model2.fit(X_train, y_train, epochs=2000, validation_split=0.2)

# Predict properties for test molecules
print("Making predictions...")
y_t = model2.predict(X_test)

# Save predictions to a submission file
predictions = np.c_[indexs, y_t]  # Combine IDs with predictions
dr = pd.DataFrame(data=predictions, columns=df_sample.columns)  # Create DataFrame
dr.to_csv('output.csv', index=False, float_format='%.3f')  # Save to CSV
print("Predictions saved to output.csv!")
