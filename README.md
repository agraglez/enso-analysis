# ENSO Analysis

## Project Overview
The purpose of this project is to create a predictive model to determine anomaly values on El Niño/Southern Oscillation (ENSO). Some interesting questions we will answer in this project are:
* How variables relate to each other?
* Which ML model is best to forecast climate variations?

## Python Packages Used
To work on the project the following packages were used:
* **General Purpose:** `warnings`, `random` and `sys`.
* **Data Manipulation:** `pandas` and `numpy`.
* **Data Visualization:** `matplotlib` and `seaborn`.
* **Statistical analysis:** `statsmodel`.
* **Machine Learning:** `sklearn`, `tensorflow`, `pyod` and `skforecast`.

## Data
For the analysis two different datasets were used:
* El Nino dataset from the UC Irvine ML repository which can be found in the following [link](https://archive.ics.uci.edu/dataset/122/el+nino).
* El Nino-Southern Oscillation (ENSO) Data from Kaagle which can be found in the following [link](https://www.kaggle.com/datasets/shabanamir/enso-data).

## Code Structure
The structure of the code is shown below:
```
|-- data
    |-- elnino
    |-- elnino.col
    |-- ENSO.csv
    |-- tao-all2.col
    |-- tao-all2.dat
|-- notebooks
    |-- enso-analysis.ipynb
|-- utils
    |-- my_utils
README.md
```

## Results and evaluation
The relevant results are presented on the jupyter notebook of the project; some important results and evaluation methods of the analysis are mentioned below:
* For the supervised models the metric used was RMSE. 
* Unsupervided methods were used to detect anomalies.
* Several models were use to forecast ENSO anomalies with neural networks being the best approach.
* The best RMSE was approximately 0.2 on the test set.