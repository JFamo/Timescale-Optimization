from matplotlib import pyplot as plt
#from gluoncv import model_zoo, data, utils
#from gluoncv.data.transforms.pose import detector_to_alpha_pose, heatmap_to_coord_alpha_pose
from keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, Reshape, Concatenate
from keras.models import Model
from data_utils import load_all_positional_data, load_all_keyframe_data, load_gt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from PIL import Image
from model import split_into_n_time
import random
import keras.models

def load_model(load_path):
    model = keras.models.load_model(load_path)
    return model

# Function to get some test frame image
def get_test_frame_image(frames_path, scene, clip, frame):
    # Load image from test data
    file_name = os.path.join(frames_path, '{:02d}'.format(scene) + "_" + '{:04d}'.format(clip), '{:03d}'.format(frame) + ".jpg")
    img = Image.open(file_name)
    return np.array(img)

# Function to visualize model predicted keypose via gluoncv
def show_prediction_plot(frames_path, scene, clip, frame, predictions):

    # Get image as nparray
    img_array = get_test_frame_image(frames_path, scene, clip, frame)

    # Fake confidence
    confidence = np.ones((predictions.shape[0], predictions.shape[1], 1))

    # Class ids
    class_IDs = np.array(range(predictions.shape[0]))

    # Ignore boxes
    bounding_boxes = np.array([])
    scores = np.array([])

    # # Plot
    # ax = utils.viz.plot_keypoints(img_array, pred_coords, confidence, class_IDs, bounding_boxs, scores, box_thresh=0.5, keypoint_thresh=0.2)
    # plt.show()

# Function to run testing on some data
def run_test(model_path, frames_path, n=3, threshold=0.5):
    # Load data
    poses = load_all_positional_data(frames_path)

    # Load kfs
    keyframes = load_all_keyframe_data(frames_path)

    # Print loaded data shape
    print("Loaded poses with " + str(poses.shape))

    # Split poses into groups of n
    poses = split_into_n_time(poses, n)
    keyframes = split_into_n_time(keyframes, n)

    # Load model
    model = load_model(model_path)

    # Make predictions
    results = model.predict(np.array(poses))
    
    # Count
    right_count = 0
    total_count = 0

    # Iterate and check mse
    for i in range(len(poses)): # Groups of 3 poses
        for d in range(len(keyframes[i])):
            kf = keyframes[i][d]
            #print("Keyframe had " + str(kf[0]) + ", " + str(kf[1]) + ", " + str(kf[2]))
            #print("loaded GT " + str(load_gt(kf[0],kf[1],kf[2])))
            #print("-> " + str(mean_squared_error(poses[i][d], results[i][d])))

            total_count += 1
            trueVal = int(load_gt(kf[0],kf[1],kf[2]))

            # Anomaly on mse > 30
            if mean_squared_error(poses[i][d], results[i][d]) > threshold*100:
                # Consider anomaly
                if trueVal == 1:
                    right_count += 1
            else:
                if trueVal == 0:
                    right_count += 1


    # DEBUG
    print(str(right_count / total_count))