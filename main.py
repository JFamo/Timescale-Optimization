import argparse
from data_utils import load_dataset
from encoding import train_autoencoding
from model import train_model

# Parse input args
parser = argparse.ArgumentParser()
parser.add_argument("--do_load_data", help="Whether to load data", action="store_true")
parser.add_argument("--data_path", help="Path to the training/testing data to load")
parser.add_argument("--numpy_path", help="Path to save the processed data to")

parser.add_argument("--train_encoders", help="Whether to train encoder/decoder", action="store_true")
parser.add_argument("--encoding_data_path", help="Path to train encodings from")

parser.add_argument("--train_model", help="Whether to train model", action="store_true")
parser.add_argument("--model_data_path", help="Path to train model from")
args = parser.parse_args()

# Handle processing data
if args.do_load_data:
    load_dataset(args.data_path, args.numpy_path)

# Handle train encoder/decoder
if args.train_encoders:
    train_autoencoding(args.encoding_data_path)

# Handle train model
if args.train_model:
    train_model(args.model_data_path)