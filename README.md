# Project 5 Recognition with deep learning
This repository contains the code and dataset of project 5.

## Usage

### Setup prerequests
#### Using python venv
- First create a virtual environment by running `python3 -m venv .venv` in the root folder.
- Then activate it by running `.venv/bin/activate`.
  - In windows maybe you should run `.venv\Scripts\activate`.
- Then install following dependencies:
  `pip install jupyter matplotlib scipy opencv-python pytorch torchvision pyyaml`

#### Using conda
- Run `conda env create -f proj5/proj5_env_<OS>.yml` to create a new environment named "cs6476_proj5" with all the necessary dependencies.
- Then activate it by running `conda activate cs6476_proj5`.

### Run jupyter notebook
All the codes are compeleted in this repository. Open `proj5/proj5.ipynb` in **jupyter notebook** and run all cells to train the model. And you would see the result in each part of the experiment.

