import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from data_utils import load_all_positional_data
from sklearn.model_selection import train_test_split

hidden_dim = 1024 
feature_dim = 34

class Autoencoder(Model):
  def __init__(self, hidden_dim, feature_dim):
    super(Autoencoder, self).__init__()

    self.feature_dim = feature_dim
    self.hidden_dim = hidden_dim   
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(feature_dim,)),
      layers.Dense(hidden_dim, activation=None),
      layers.Dense(hidden_dim, activation=None),
    ])

    self.conv = tf.keras.Sequential([
      layers.Conv1D(filters=1024, kernel_size=3),
      layers.GlobalMaxPooling1D() # Fit dimension over time
    ])

    self.decoder = tf.keras.Sequential([
      layers.Dense(hidden_dim, activation=None),
      layers.Dense(feature_dim * 3, activation=None), # Output length is 3 for first timescale
    ])

  def call(self, x):
    encoded_stack = []

    print("X has " + str(tf.shape(x)))

    # Input x has (3,34), encode each
    for row in x:

      print("Row has " + str(tf.shape(row)))

      encoded = self.encoder(row)
      encoded_stack.append(encoded)

    # Stack encodings into (3,1024)
    encoded_data = tf.stack(encoded_stack, axis=0)

    print("Stack has " + str(tf.shape(encoded_data)))

    # Run convolution on stack
    convoluted_data = self.conv(encoded_data)

    # Decode convolution to expected dimensions
    decoded = self.decoder(convoluted_data)

    # Split into 3 separate frames
    decoded_split = tf.split(decoded, 3, axis=1)

    return decoded_split

# Custom loss function with predictions and truths of flat (34*3,)
def timescale_loss(y_true, y_pred):
  loss_sum = 0
  for i in range(34):
    loss_sum += ((y_pred[i] - y_true[i]) ** 2)
  return loss_sum

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

def train_autoencoding(data_path):
  # Compile autoencoder
  autoencoder = Autoencoder(hidden_dim, feature_dim)
  autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

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

  # DEBUG
  X_train = X_train[:100]
  y_train = y_train[:100]
  X_test = X_test[:100]
  y_test = y_test[:100]

  X_train = np.array(X_train)
  X_test = np.array(X_test)
  y_train = np.array(y_train)
  y_test = np.array(y_test)

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
  autoencoder.fit(X_train, y_train, epochs=10, shuffle=True, validation_data=(X_test, y_test))
