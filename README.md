# ML_Project

## Introduction 

The main objective of this project, is to be able to build a machine learning model, that can predict the popularity of a song on Spotify, based on various features of the song.

We began this project by finding a dataset collected from the Spotify Web API, that contains contains 169k songs from the year 1921 to year 2020.

The dataset contained 19 columns of different features of the song, including one column for "Popularity". This column contains continuous data, and scores the songs in a range of 0-100, based on their popularity.

Therefore, we defined the "Popularity" as our target variable, and for our projet, we tested different Tree-Based Regression algorithms, to find an optimal model that will be most likely to predict the popularity of a song.

## Business Need:
We identified a few use cases in which our machine learning model could add value to businesses/ individuals:

- This model could be useful for Spotify, when they want to understand which new songs on the platform could be popular, and how they can push these songs to Spotify users
- This model could help new and upcoming artists, who can put the data of their unreleased songs to predict the popularity of it
- Record labels/ record companies could benefit from this model as well - by being able to predict the popularity of a song, it can help them reduce the risk when - investing in new songs by their artists, or even when selecting new artists
- It can help artists and labels understand which features of a song are going to impact the level of popularity

## Setup 
Set up Python and install the necessary packages as described below. 
 - Python version 3.9

### for ml
- from sklearn.linear_model import LinearRegression
### base modules
- import os
- import sys
- import copy

### for manipulating data
- import numpy as np
- import pandas as pd
- import math
- import dill
- import scipy

### for Machine Learning
- from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, BaggingRegressor
- from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
- from sklearn import metrics
- from sklearn.preprocessing import LabelEncoder
- from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
- from sklearn.calibration import CalibratedClassifierCV
- from sklearn.inspection import permutation_importance
- from scipy.cluster import hierarchy
- from sklearn.metrics import mean_squared_error

### for visualization
- from IPython.display import display
- from matplotlib import pyplot as plt

## Final Model:
In this project, we tested several parameters, using Grid Search and understanding our feature importance/ correlation to improve our final Random Forest Regressor model.

We concluded that our final model, "best_model2" is the best model for predicting the popularity of our song. This model does not overfit, and the R^2 score on the testing set (0.9093) means our model is 91% likely to accurately predict the popularity of our song.

The OOB score (0.7686) is not significantly lower than the R^2 on our training set (0.7752) which is another good indication that our model is not overfitting.

The RMSE on both training and valid sets remain in the range between 0-10, which suggests our model is accurate, however we noticed that the RMSE on the training set (10.3381) is still slightly higher than the RMSE on the test set (6.3370) suggesting that our model could still be slightly overfitting.

## How we would further improve our project:

- Using Boosting algorithms such as CatBoost and XBoost, to find a more accurate model
- Understanding feature importance, permutation importance and feature correlation in more depth, in order to see how we can use the information we got to further understand our dataset and find out how to improve our model
- Since we realised a lot of our features were not relevant enough to predict the popularity of a song, we could potentially add new features to our dataset, that provide more details of a song, that could be more relavant than our existing features
- We would perform the Grid Search with more values. Since the Grid Search takes a long time to run, if we had more time for this project, we would be able to get better optimal combination of parameters
- Testing out more paramemters with the Bagging model - our bagging model gave us decent results, however need more time and more processing power
- Spend more time understanding overfitting, to be able to properly identify when our model is overfitting, and how to properly overcome this problem
- Test our best model on a new dataset of songs, to see how accurately it performs

## Acknowledgments

Group members: Gaël CAVECCHIA, Nathan DESTREZ, Trisha KUMAR
