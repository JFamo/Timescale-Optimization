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

Where `--do_load_data` signals to process json data, `--data_path` indicates the relative path of the json data, `--numpy_path` indicates the path to save data to