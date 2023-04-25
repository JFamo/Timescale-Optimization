# Timescale-Optimization
Optimizing timescale-driven architectures for human abnormal behavior detection

# Setup
This repository is configured using pipenv for dependency management

Install via `pip install pipenv`

Install dependencies via `pipenv install`

Then use `pipenv run <file.py>` to execute files

## Loading data
Download the data files and place them in the root of the repository

To process json data into numpy form, as used by the program, use the args as follows:

`pipenv run python main.py --do_load_data --data_path="data/train" --numpy_path="numpy/train"`
`pipenv run python main.py --do_load_data --data_path="data/test" --numpy_path="numpy/test"`

Where `--do_load_data` signals to process json data, `--data_path` indicates the relative path of the json data, `--numpy_path` indicates the path to save data to

## Training the Model
To train the model, ensure the data is loaded to some path. Then, use:

`pipenv run python main.py --train_model --save_model --model_data_path=numpy/train`

The save path will default to the `models` folder which should be created.

# Testing the model
To test the model, note the path where the model is saved and the data path of testing data. Then, use:

`pipenv run python main.py --test_model --model_load_path="models/model_ts3_0" --test_data_path="numpy/test" --test_frames=3`

This will run evaluation across all frames in the test set.