'''
Build a decision tree classifier on Pima
The build a random forest classifier on Pima
'''

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

def load_data():
	'''
	Load and scale data
	'''
	df = pd.read_csv('diabetes.csv')
	print('Any N/A exist:',df.isnull().values.any())
	# check for unique elements for case of dummy variables
	names = list(df.columns[:-1])
	scaler = MinMaxScaler()
	df.iloc[:,:-1] = scaler.fit_transform(df.iloc[:,:-1])
	print(df.head())
	return df,names

def train_test_split_data(df):
	'''
	Split the data into training and test sets
	'''
	X = df.iloc[:,:-1].to_numpy()
	y = df.iloc[:,-1].to_numpy()
	xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.25,random_state=42)
	return xtrain,xtest,ytrain,ytest

def decision_tree(xtrain,xtest,ytrain,ytest,names,exp):
	'''
	Build and fit a decision tree model, then use it to predict category and report accuracy score
	'''
	dt = DecisionTreeClassifier(max_features='auto',max_depth=5,min_samples_leaf=2,random_state=exp)
	dtree = dt.fit(xtrain,ytrain)
	ypred = dt.predict(xtest)
	acc = accuracy_score(ytest,ypred)
	print(acc,' Accuracy Score Decision tree')
	# Rules of the decision tree
	dt_rules = export_text(dtree,feature_names=names,show_weights=True)
	#print(dt_rules)
	return acc

def random_forest_tree(xtrain,xtest,ytrain,ytest,exp,size):
	'''
	Build and fit a random forest model, then use it to predict category and report accuracy score
	'''
	rf = RandomForestClassifier(n_estimators=size,max_depth=15,min_samples_leaf=2,max_leaf_nodes=20,n_jobs=-1,random_state=exp)
	rf.fit(xtrain,ytrain)
	ypred = rf.predict(xtest)
	acc = accuracy_score(ytest,ypred)
	print(acc,' Accuracy Score Random Forest')
	return acc

def extra_classifier_tree(xtrain,xtest,ytrain,ytest,exp,size):
	'''
	Build and fit a extra tree model, then use it to predict category and report accuracy score
	'''
	et = ExtraTreesClassifier(n_estimators=size,max_depth=15,min_samples_leaf=2,max_leaf_nodes=20,n_jobs=-1,random_state=exp)
	et.fit(xtrain,ytrain)
	ypred = et.predict(xtest)
	acc = accuracy_score(ytest,ypred)
	print(acc,' Accuracy Score Extra Trees')
	return acc

def neural_network(xtrain,xtest,ytrain,ytest,exp):
	'''
	Build and fit a neural network model, then use it to predict category and report accuracy score
	'''
	nn = MLPClassifier(hidden_layer_sizes=(30,30),learning_rate_init=0.001,solver='adam',max_iter=500,random_state=exp)
	nn.fit(xtrain,ytrain)
	ypred = nn.predict(xtest)
	acc = accuracy_score(ytest,ypred)
	print(acc,' Accuracy Score Neural Network')
	return acc

def performance_five_tests(xtrain,xtest,ytrain,ytest,names):
	'''
	Report the mean classification accuracy performance for each of the models across 5 experiments
	'''
	exp_num = 5 
	acc_dt = np.empty(exp_num)
	acc_rf = np.empty(exp_num)
	acc_et = np.empty(exp_num)
	acc_nn = np.empty(exp_num)
	size = 50
	for exp in range(exp_num):
		acc_dt[exp] = decision_tree(xtrain,xtest,ytrain,ytest,names,exp)
		acc_rf[exp] = random_forest_tree(xtrain,xtest,ytrain,ytest,exp,size)
		acc_et[exp] = extra_classifier_tree(xtrain,xtest,ytrain,ytest,exp,size)
		acc_nn[exp] = neural_network(xtrain,xtest,ytrain,ytest,exp)
	print(acc_dt.mean(), ' Decision Tree mean')
	print(acc_rf.mean(), ' Decision Random Forest mean')
	print(acc_et.mean(), ' Extra Tree Classifer mean')
	print(acc_nn.mean(), ' Neural Network Classifier mean')

def num_trees_performance_tests(xtrain,xtest,ytrain,ytest):
	'''
	Random forest and extra tree classifer for different ensemble sizes
	The mean and standard deviation were reported for 10 experiments for each ensemble size
	'''
	num_trees = list(range(10,101,20))
	exp_num = 10 
	ensemble_mean_rf = []
	ensemble_mean_et = []
	for size in num_trees:
		acc_rf = np.empty(exp_num)
		acc_et = np.empty(exp_num)
		for exp in range(exp_num):
			acc_rf[exp] = random_forest_tree(xtrain,xtest,ytrain,ytest,exp,size)
			acc_et[exp] = extra_classifier_tree(xtrain,xtest,ytrain,ytest,exp,size)
		ensemble_mean_rf.append(acc_rf.mean())
		ensemble_mean_et.append(acc_et.mean())
	print('For number of trees 10 to 100 by 20')
	print('Random Forest',ensemble_mean_rf)
	print('Extra Trees', ensemble_mean_et)

def main():
	df,names = load_data()
	xtrain,xtest,ytrain,ytest = train_test_split_data(df)
	# Explore mean performance 5 tests
	performance_five_tests(xtrain,xtest,ytrain,ytest,names)
	# Explore different number of trees in ensemble
	num_trees_performance_tests(xtrain,xtest,ytrain,ytest)

if __name__ == '__main__':
	main()