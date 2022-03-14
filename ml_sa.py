# -*- coding: utf-8 -*-

"""
Created on February 2022

@author: Hosein Khanalizadeh

"""

# افزودن پکیج ها
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split , cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans


# خواندن فایل داده ها
data = pd.read_csv('data.csv' , sep=';')

# اطلاعاتی در مورد داده ها
data.describe()

# تنظیم داده ها و اضافه کردن ستون هایی مرتبط برای اجرای بهتر الگوریتم
data['alcohol_per_pH'] = data['alcohol'] / data['pH']
data['volatile acidity_per_pH'] = data['volatile acidity'] / data['pH']
data['chlorides_per_fixed acidity'] = data['chlorides'] / data['fixed acidity']
data['sulphates_per_total sulfur dioxide'] = data['sulphates'] / data['total sulfur dioxide']
df_label = data['quality'].values
temp = data.drop(['quality'] , axis=1)

# نرمال کردن داده ها در بازه 0 و 1
df = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))

# تقسیم داده ها به بخش آموزش و تست
X_train, X_test, y_train , y_test = train_test_split(df, df_label, test_size=0.2, random_state=2)

# مدل رگرسیون خطی (Linear Regression)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
print('predictions : ', lin_reg.predict(X_train))
print('label       : ', np.array(y_train))
print(30 * '-')

# مدل درخت تصمیم (Decision Tree Regressor)
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)
print('predictions : ', tree_reg.predict(X_train))
print('label       : ', np.array(y_train))
print(30 * '-')

# الگوریتم جنگل تصادفی (Random Forest Regressor)
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, y_train)
print('predictions : ', lin_reg.predict(X_train))
print('label       : ', np.array(y_train))
print(30 * '-')

# الگوریتم ماشین بردار (SVM)
svm = SVC(random_state=1)
svm.fit(X_train, y_train)
print('predictions : ', svm.predict(X_train))
print('label       : ', np.array(y_train))
print(30 * '-')

# الگوریتم نزدیک ترین همسایه (KNN)
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
print('predictions : ', svm.predict(X_train))
print('label       : ', np.array(y_train))
print(30 * '-')

# ارزیابی خطا با روش Mean Squared Error (MSE)
scores = cross_val_score(lin_reg , X_train , y_train , scoring='neg_mean_squared_error' , cv=6)
lin_reg_trs = np.sqrt(-scores)
scores = cross_val_score(tree_reg , X_train , y_train , scoring='neg_mean_squared_error' , cv=6)
tree_reg_trs = np.sqrt(-scores)
scores = cross_val_score(forest_reg , X_train , y_train , scoring='neg_mean_squared_error' , cv=6)
forest_reg_trs = np.sqrt(-scores)
scores = cross_val_score(svm, X_train , y_train , scoring='neg_mean_squared_error' , cv=6)
svm_trs = np.sqrt(-scores)
scores = cross_val_score(knn, X_train , y_train , scoring='neg_mean_squared_error' , cv=6)
knn_trs = np.sqrt(-scores)

def display_scores(scores , model_name):
	print(model_name)
	print('scores             : ' , scores)
	print('mean               : ' , scores.mean())
	print('standard deviation : ' , scores.std())
	print(30 * '-')

display_scores(lin_reg_trs , 'Linear Regression')
display_scores(tree_reg_trs , 'Decision Tree Regression')
display_scores(forest_reg_trs , 'Random Forest Regression')
display_scores(svm_trs , 'SVM')
display_scores(knn_trs , 'KNN')
