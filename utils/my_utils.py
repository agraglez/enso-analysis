# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 11:55:47 2023

@author: grago
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_importance(importance, names, model_type):
    '''
    Plot feature importance of the model.

    Parameters
    ----------
    importance : Numpy ndarray of shape (n_features,)
        Numpy array containint feature importance.
    names : list 
        List of names of features.
    model_type : str
        Name of the model.

    Returns
    -------
    None.

    '''
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names,
            'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(10, 8))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + 'Feature Importance')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Importance')


def series_to_nn(data, step=1):
    '''
    Create arrays of values from timeseries.

    Parameters
    ----------
    data : Numpy ndarray of shape (n,1)
        Timeseries of interest.
    step : Int, optional
        Number of lag observations. The default is 1.

    Returns
    -------
    Numpy ndarrays of shape (n-step-1,1) and (n-step-1,) 
        
    '''
    X = []
    Y = []
    for i in range(len(data)-step-1):
        X.append(data[i:(i+step), 0])
        Y.append(data[i + step, 0])
    return np.array(X), np.array(Y)
