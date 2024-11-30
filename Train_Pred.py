import pandas as pd
import numpy as np
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

# extracting information from csv file 
a = open('USA_Housing.csv','r')
b = csv.reader(a, delimiter=',')
data_X = pd.read_csv('USA_Housing.csv')

data_Y = data_X[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population', 'Price']]

# seperating dependent and independent variables
X = data_Y.iloc[0:,0:5]
y = data_Y['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 9)
lr = LinearRegression()
lr.fit(X_train , y_train)
y_train_prediction = lr.predict(X_train)
y_test_prediction = lr.predict(X_test)
plt.scatter(y_train, y_train_prediction)
plt.xlabel('actual price')
plt.ylabel('predicted price')
plt.title('best fit line')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
plt.show()

# evaluation
def training_evaluation(actual,predicted):
       mse = mean_squared_error(actual, predicted)
       mae = mean_absolute_error(actual, predicted)
       rmse = np.sqrt(mean_squared_error(actual, predicted))
       r2 = r2_score(actual, predicted)
       print(f'mse:{mse}')
       print(f'mae:{mae}')
       print(f'rmse:{rmse}')
       print(f'r2:{r2}')

training_evaluation(y_train,y_train_prediction)