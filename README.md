
## Used Car Price Prediction
We have used a used car price dataset, visualized it, and checked it for price prediction purposes.

## Process

## VIF and multicollinearity
Multicollinearity occurs when two or more independent variables in a data frame have a high correlation with one another in a regression model.
VIF score of an independent variable represents how well the variable is explained by other independent variables.


#Logistic Regression
It is a supervised machine learning algorithm mainly used for classification tasks where the goal is to predict the probability that an instance of belonging to a given class or not. It is a kind of statistical algorithm, which analyze the relationship between a set of independent variables and the dependent binary variables. It is a powerful tool for decision-making.

##Decision Tree
A decision tree is a non-parametric supervised learning algorithm, which is utilized for both classification and regression tasks. It has a hierarchical, tree structure, which consists of a root node, branches, internal nodes and leaf nodes.

## Linear Regression
Linear regression is a type of statistical analysis used to predict the relationship between two variables. It assumes a linear relationship between the independent variable and the dependent variable, and aims to find the best-fitting line that describes the relationship.

##Gradient boosting Regression
It calculates the difference between the current prediction and the known correct target value. This difference is called residual. After that Gradient boosting Regression trains a weak model that maps features to that residual

##XGB Regresser
XGBoost is an efficient implementation of gradient boosting that can be used for regression predictive modeling. How to evaluate an XGBoost regression model using the best practice technique of repeated k-fold cross-validation. How to fit a final model and use it to make a prediction on new data.



## Libraries and Usage

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',None)
plt.style.use('fivethirtyeight')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

```






## Accuracy






## Run Locally

Clone the project

```bash
  git clone https://link-to-project
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  npm install
```

Start the server

```bash
  npm run start
```


## Used By
The project is used by various automobile companies to check and scan for the market and to deal with various stakeholders, and is even helpful for customers in every strata.
## Appendix

A very crucial project in the realm of data science and sentiment analyss using visualization techniques as well as machine learning modelling.

## Acknowledgements

The project is taken from
https://www.kaggle.com/code/chloe912/used-cars-prediction
## Tech Stack

**Client:** Python, Machine Learning, Rgression analysis, ML modelling, working on a csv file, XGB regresser, gradient boost regresser



## Feedback

If you have any feedback, please reach out to us at chawlapc.619@gmail.com

