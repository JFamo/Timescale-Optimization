from keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, Reshape, Concatenate
from keras.models import Model
from data_utils import load_all_positional_data
import numpy as np
from sklearn.model_selection import train_test_split

# Class for a model handling three timesteps
class Three_Timescale_Model():
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

# # Class for a model handling four timesteps
# class Four_Timescale_Model():
#     input_shape = (4, 34) # Input shape for timescale 4
#     encoding_dim = 1024 # Encoding dimension
#     kernel_size = 3 # Kernel size, always 3/5
#     filters = 1024 # Number of conv filters

#     # Input layer shape
#     input_layer = Input(shape=input_shape)

#     # Define encoder and decoder
#     encoder = Dense(encoding_dim, activation='relu')
#     decoder_four = Dense(4 * 34, activation='relu')
    
#     # Define utility layers
#     concater = Concatenate(axis=1)
#     reshaper = Reshape((3, encoding_dim))
#     reshaper_two = Reshape((2, encoding_dim))
#     pooling = GlobalMaxPooling1D()

#     # Define conv filter layers
#     conv_layer_1 = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')
#     conv_layer_2 = Conv1D(filters=filters, kernel_size=2, activation='relu')

#     # Encode each input timestep
#     encoded1 = encoder(input_layer[:, 0])
#     encoded2 = encoder(input_layer[:, 1])
#     encoded3 = encoder(input_layer[:, 2])
#     encoded4 = encoder(input_layer[:, 3])

#     # Concatenated inputs for conv layer
#     concatenated1 = concater([encoded1, encoded2, encoded3])
#     concatenated1 = reshaper(concatenated1)

#     concatenated2 = concater([encoded2, encoded3, encoded4])
#     concatenated2 = reshaper(concatenated2)

#     # Conv layer, 1024 filters at some size
#     conv1 = conv_layer_1(concatenated1)
#     conv2 = conv_layer_1(concatenated2)

#     # max pooling down to single 1024 feature for decoding
#     pooled1 = pooling(conv1)
#     pooled2 = pooling(conv2)

#     concatenated3 = concater([pooled1, pooled2])
#     concatenated3 = reshaper_two(concatenated3)

#     # Last convolution on those outputs in second layer
#     conv3 = conv_layer_2(concatenated3)
#     pooled3 = pooling(conv3)

#     # Decoder into some timesteps
#     decoded = decoder_four(pooled3)

#     # Output layer, reshape decoded state down to 4 timescales
#     output_layer = Reshape((4, 34))(decoded)

# Class for a model handling five timesteps
# class Five_Timescale_Model():
#     input_shape = (5, 34) # Input shape for timescale 5
#     encoding_dim = 1024 # Encoding dimension
#     kernel_size = 3 # Kernel size, always 3/5
#     filters = 1024 # Number of conv filters

#     # Input layer shape
#     input_layer = Input(shape=input_shape)

#     # Define encoder and decoder
#     encoder = Dense(encoding_dim, activation='relu')
#     decoder_three = Dense(3 * 34, activation='relu')
#     decoder_five = Dense(5 * 34, activation='relu')
    
#     # Define utility layers
#     concater = Concatenate(axis=1)
#     reshaper = Reshape((3, encoding_dim))
#     pooling = GlobalMaxPooling1D()

#     # Define conv filter layers
#     conv_layer_1 = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')
#     conv_layer_2 = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')

#     # Encode each input timestep
#     encoded1 = encoder(input_layer[:, 0])
#     encoded2 = encoder(input_layer[:, 1])
#     encoded3 = encoder(input_layer[:, 2])
#     encoded4 = encoder(input_layer[:, 3])
#     encoded5 = encoder(input_layer[:, 4])

#     # Concatenated inputs for conv layer
#     concatenated1 = concater([encoded1, encoded2, encoded3])
#     concatenated1 = reshaper(concatenated1)

#     concatenated2 = concater([encoded2, encoded3, encoded4])
#     concatenated2 = reshaper(concatenated2)

#     concatenated3 = concater([encoded3, encoded4, encoded5])
#     concatenated3 = reshaper(concatenated3)

#     # Conv layer, 1024 filters at some size
#     conv1 = conv_layer_1(concatenated1)
#     conv2 = conv_layer_1(concatenated2)
#     conv3 = conv_layer_1(concatenated3)

#     # max pooling down to single 1024 feature for decoding
#     pooled1 = pooling(conv1)
#     pooled2 = pooling(conv2)
#     pooled3 = pooling(conv3)

#     concatenated4 = concater([pooled1, pooled2, pooled3])
#     concatenated4 = reshaper(concatenated4)

#     # Last convolution on those outputs in second layer
#     conv4 = conv_layer_2(concatenated4)
#     pooled4 = pooling(conv4)

#     # Decoder into some timesteps
#     decoded = decoder_five(pooled4)

#     # Output layer, reshape decoded state down to 5 timescales
#     output_layer = Reshape((5, 34))(decoded)

# # Function to split training data into groups of n
# def split_into_n_time(poses, n):
#     # Ensure shape[0] is divisible by n
#     while poses.shape[0] % n != 0:
#         poses = poses[:-1, :]
    
#     # Split and return
#     n_split_poses = np.vsplit(poses, poses.shape[0] / n)

#     # Flatten each array
#     #for i in range(len(n_split_poses)):
#     # n_split_poses[i] = n_split_poses[i].flatten()

#     # Return
#     return n_split_poses

def train_model(data_path, save_model=True, save_path='models/model_ts3_0', n=3):
    # Define the model
    if n == 5:
        ref_model = Five_Timescale_Model()
        #pass
    if n == 4:
        #ref_model = Four_Timescale_Model()
        pass
    else:
        ref_model = Three_Timescale_Model()
    model = Model(inputs=ref_model.input_layer, outputs=ref_model.output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Load data
    poses = load_all_positional_data(data_path)

    # Print loaded data shape
    print("Loaded poses with " + str(poses.shape))

    # Split poses into groups of n
    poses = split_into_n_time(poses, n)

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
    model.fit(X_train, y_train, epochs=1, shuffle=True, validation_data=(X_test, y_test))

    # Save trained model
    if save_model:
        model.save(save_path)
