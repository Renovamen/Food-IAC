# Scripts for Building Dataset

This folder contains scripts for building the dataset, including:

- [`crawler.py`](crawler.py): A crawler for crawling images along with their captions and scores from [dpchallenge.com](https://www.dpchallenge.com/). Images will be saved in folder `data/images` and other information will be recorded in `data/raw.json`.

- [`clean.py`](clean.py): Perform basic cleaning on raw data (`data/raw.json`) and filter captions based on their 'informativeness score' (see our paper for more information). Output `data/clean.json`.

- [`split.py`](split.py): Split the data into train set, validation set and test set (6:1:1), and create tokens for each caption. Output files will be saved under `data/final`.

- [`profile.py`](profile.py): Provide a simple profile of the established dataset.
