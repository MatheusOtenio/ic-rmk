# Student Dropount Prediction Thesis
This repository contains all scripts and analyses for my final course project, where I aim to predict student dropout rates using machine learning. Note that all Excel, CSV and original Datasets files are ignored for upload to this repository.

## Folder Structure

### `data-generator`
- **Description**:  Contains scripts that generate various versions of the original dataset.
- **Objective**: Generate various versions of the original dataset.

### `exploratory-analysis`
- **Description**: Contains notebooks for exploring patterns and behaviors in student data.
- **Objective**: To analyze the data and uncover interesting insights.

### `full-experiments`
- **Description**: Contains scripts that generate various versions of the original dataset and run experiments on all datasets (first thing that was done in this research).
- **Improvements Needed**: Optimize the scripts to avoid running unnecessary experiments, especially when a dataset lacks proper class balance.

### `periods-experiments`
- **Description**: Contains a script that runs specifics datasets predictions by periods.
- **Objective**: Get good results for ours analysis.

### `searching-good-results`
- **Description**: Contains notebooks for reviewing and analyzing the results obtained from our experiments.
- **Objective**: After running all experiments, we have over 500,000 results. To sift through the results, visualize the performance of predictive models, and identify the best-performing ones.
