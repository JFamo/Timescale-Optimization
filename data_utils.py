import numpy as np
import os
from tqdm import tqdm
import json

# Function to process a single person
def process_person_json(clip_data, person_index, scene_id, clip_id):

    # Load json for a single person
    person_data = clip_data[str(person_index)]
    sing_pose_np = []
    sing_scores_np = []
    sing_keyframes_np = []

    # Iterate keyframes for person
    single_person_dict_keys = sorted(person_data.keys())
    for key in single_person_dict_keys:

        # Save keypoints to nparray
        curr_pose_np = np.array(person_data[key]['keypoints']).reshape(-1, 3) # 17 parts, x,y,c = 51 fields
        sing_pose_np.append(curr_pose_np)
        sing_scores_np.append(person_data[key]['scores'])
        sing_keyframes_np.append(np.array([scene_id, clip_id, key]))

    # Stack results from each keyframe
    sing_pose_np = np.stack(sing_pose_np, axis=0)
    sing_scores_np = np.stack(sing_scores_np, axis=0)
    sing_keyframes_np = np.stack(sing_keyframes_np, axis=0)

    return sing_pose_np, sing_scores_np, sing_keyframes_np


# Function to process json data from a single scene & clip
def process_clip_json(json_data, results_root_path, scene_id, clip_id):

    # Iterate people detected in clip
    for person_index in sorted(json_data.keys(), key=lambda x: int(x)):

        # Process each person into np arrays
        person_pose_np, person_scores_np, person_keyframes_np = process_person_json(json_data, person_index, scene_id, clip_id)

        # Save person data as array
        np.save(os.path.join(results_root_path, scene_id + "_" + clip_id + "_" + str(person_index) + "_pose"), person_pose_np)
        np.save(os.path.join(results_root_path, scene_id + "_" + clip_id + "_" + str(person_index) + "_score"), person_scores_np)
        np.save(os.path.join(results_root_path, scene_id + "_" + clip_id + "_" + str(person_index) + "_keyframe"), person_keyframes_np)


# Function to load each tracked_person.json file
def load_dataset(data_root_path, results_root_path):

    # Read relevant files from data dir
    dir_list = os.listdir(data_root_path)
    json_list = sorted([json for json in dir_list if json.endswith('tracked_person.json')])

    # Iterate each file, pull out scene and clip, construct path
    for person_data_file in tqdm(json_list):
        scene_id, clip_id = person_data_file.split('_')[:2]
        clip_json_path = os.path.join(data_root_path, person_data_file)

        with open(clip_json_path, 'r') as f:
            clip_dict = json.load(f)
            process_clip_json(clip_dict, results_root_path, scene_id, clip_id)


# Function to load a single person per clip per scene
def load_person_in_clip(results_root_path, scene_id, clip_id, person_id):
    scores = np.load(os.path.join(results_root_path, scene_id + "_" + clip_id + "_" + str(person_id) + "_score"))
    poses = np.load(os.path.join(results_root_path, scene_id + "_" + clip_id + "_" + str(person_id) + "_pose"))

    return poses, scores

# Function to load all people's positional data
def load_all_positional_data(results_root_path):
    # Read relevant files from data dir
    dir_list = os.listdir(results_root_path)
    pose_file_list = sorted([npfile for npfile in dir_list if npfile.endswith('_pose.npy')])

    all_poses = []

    # Iterate each file, pull out scene and clip, construct path
    for pose_data_file in tqdm(pose_file_list):
        scene_id, clip_id, person_id = pose_data_file.split('_')[:3]
        pose_data_file_full = os.path.join(results_root_path, pose_data_file)

        pose = np.load(pose_data_file_full)

        # Select only positional data
        pose = pose[:, :, 0:2]
        pose = pose.reshape(-1, 34)

        all_poses.append(pose)

        # DEBUG
        # print(str(pose_data_file) + " " + str(pose.shape))

    # Concat all poses
    return np.concatenate(all_poses, axis=0)

# Function to load all people's kf data
def load_all_keyframe_data(results_root_path):
    # Read relevant files from data dir
    dir_list = os.listdir(results_root_path)
    pose_file_list = sorted([npfile for npfile in dir_list if npfile.endswith('_keyframe.npy')])

    all_poses = []

    # Iterate each file, pull out scene and clip, construct path
    for pose_data_file in tqdm(pose_file_list):
        scene_id, clip_id, person_id = pose_data_file.split('_')[:3]
        pose_data_file_full = os.path.join(results_root_path, pose_data_file)

        pose = np.load(pose_data_file_full)
        all_poses.append(pose)

    # Concat all poses
    return np.concatenate(all_poses, axis=0)

# Function to load GT
def load_gt(scene, clip, keyframe):
    path_str = "./data/gt/" + scene + "_" + clip + ".npy"
    scene = np.load(path_str)
    return scene[int(keyframe)]