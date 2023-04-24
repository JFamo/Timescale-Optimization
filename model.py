from keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, Reshape, Concatenate
from keras.models import Model
from data_utils import load_all_positional_data
import numpy as np
from sklearn.model_selection import train_test_split

# Model vars
input_shape = (3, 34) # Input shape for timescale 3
encoding_dim = 1024 # Encoding dimension
kernel_size = 3 # Kernel size, always 3/5
filters = 1024 # Number of conv filters

# Input layer shape
input_layer = Input(shape=input_shape)

# Encode each input timestep
encoded1 = Dense(encoding_dim, activation='relu')(input_layer[:, 0])
encoded2 = Dense(encoding_dim, activation='relu')(input_layer[:, 1])
encoded3 = Dense(encoding_dim, activation='relu')(input_layer[:, 2])

# Concatenated input for conv layer
concatenated = Concatenate(axis=1)([encoded1, encoded2, encoded3])
concatenated = Reshape((3, encoding_dim))(concatenated)

# Conv layer, 1024 filters at some size
conv = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(concatenated)

# max pooling down to single 1024 feature for decoding
pooled = GlobalMaxPooling1D()(conv)

# Decoder into some timesteps
decoded = Dense(3 * 34, activation='relu')(pooled)

# Output layer, reshape decoded state down to 3 timescales
output_layer = Reshape((3, 34))(decoded)

# Function to split training data into groups of n
def split_into_n_time(poses, n):
    # Ensure shape[0] is divisible by n
    while poses.shape[0] % n != 0:
        poses = poses[:-1, :]
    
    # Split and return
    n_split_poses = np.vsplit(poses, poses.shape[0] / n)

    # Flatten each array
    #for i in range(len(n_split_poses)):
    # n_split_poses[i] = n_split_poses[i].flatten()

    # Return
    return n_split_poses

def train_model(data_path):
    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Load data
    poses = load_all_positional_data(data_path)

    # Print loaded data shape
    print("Loaded poses with " + str(poses.shape))

    # Split poses into groups of 3
    poses = split_into_n_time(poses, 3)

    # Choose training and testing data
    poses_train = poses[::2]
    poses_test = poses[1::2]

    # Handle inconsistent split
    if len(poses_train) > len(poses_test):
        poses_train = poses_train[:-1]

    # Do random 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(poses_train, poses_test, test_size=0.2)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # DEBUG
    # X_train = X_train[:100]
    # y_train = y_train[:100]
    # X_test = X_test[:100]
    # y_test = y_test[:100]

    # Flatten Ys
    #for i in range(len(y_train)):
    #  y_train[i] = y_train[i].flatten()

    #for i in range(len(y_test)):
    #  y_test[i] = y_test[i].flatten()

    # DEBUG
    print("Beginning training with Training:" + str(len(X_train)) + "," + str(len(y_train)) + ", Testing:" + str(len(X_test)) + "," + str(len(y_test)))

    # DEBUG
    print("Train X size " + str(X_train[0].shape))
    print("Train Y size " + str(y_train[0].shape))
    print("Test X size " + str(X_test[0].shape))
    print("Test y size " + str(y_test[0].shape))

    # Train
    model.fit(X_train, y_train, epochs=10, shuffle=True, validation_data=(X_test, y_test))
