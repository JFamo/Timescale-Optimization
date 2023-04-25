import argparse
from data_utils import load_dataset
from model import train_model
from test import run_test

# Parse input args
parser = argparse.ArgumentParser()
parser.add_argument("--do_load_data", help="Whether to load data", action="store_true")
parser.add_argument("--data_path", help="Path to the training/testing data to load")
parser.add_argument("--numpy_path", help="Path to save the processed data to")

parser.add_argument("--train_encoders", help="Whether to train encoder/decoder", action="store_true")
parser.add_argument("--encoding_data_path", help="Path to train encodings from")

parser.add_argument("--train_model", help="Whether to train model", action="store_true")
parser.add_argument("--model_data_path", help="Path to train model from")
parser.add_argument("--save_model", help="Whether to save model", action="store_true")
parser.add_argument("--model_save_path", help="Path to save model to")
parser.add_argument("--model_config", help="Timescale config to use, as a number")

parser.add_argument("--test_model", help="Whether to test model", action="store_true")
parser.add_argument("--model_load_path", help="Path to load model from")
parser.add_argument("--test_data_path", help="Path to load test data from")
parser.add_argument("--test_frames", help="Number of frames to run test across (based on model)")
args = parser.parse_args()

# Handle processing data
if args.do_load_data:
    load_dataset(args.data_path, args.numpy_path)

# Handle train encoder/decoder
if args.train_encoders:
   # train_autoencoding(args.encoding_data_path)
   pass

# Handle train model
if args.train_model:
    train_model(args.model_data_path, args.save_model, args.model_save_path, int(args.model_config))

# Handle test model
if args.test_model:
    run_test(args.model_load_path, args.test_data_path, int(args.test_frames))