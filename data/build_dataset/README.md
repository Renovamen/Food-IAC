# Scripts for Building Dataset

This folder contains the scripts for building dataset.

## Overview

- [`crawler.py`](crawler.py): A crawler for crawling images along with their captions and scores from [dpchallenge.com](https://www.dpchallenge.com/). Images will be saved in folder `data/images` and other information will be recorded in `data/raw.json`.

- [`clean.py`](clean.py): Perform basic cleaning on raw data (`data/raw.json`) and filter captions based on their 'informativeness score' (see our paper for more information). Output `data/clean.json`.

- [`lda.py`](lda.py): Train a LDA model for topic classification. Models will be saved to `lda_path`.

- [`classify.py`](classify.py): Classify captions into 6 aspects based on LDA model trained by [`lda.py`](lda.py). Output files will be saved to `output_path`.

- [`split.py`](split.py): Split the data into train set, validation set and test set (6:1:1), and create tokens for each caption. Output files will be saved to `data/final`.

- [`profile.py`](profile.py): Provide a simple profile of the established dataset.


&nbsp;

## Requirements

Make sure you are using Python3.5+, then:

```bash
pip install -r requirements.txt
```
