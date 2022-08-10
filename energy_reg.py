'''
A Regression problem using energy dataset
'''
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor

# goal we will just predict the y1 values with the regressor
def load_data():
	'''
	Load and scale the data
	'''
	df = pd.read_excel('energy.xlsx')
	print('Are there any N/A values:',df.isnull().values.any())
	scaler = MinMaxScaler()
	df.iloc[:,:-2] = scaler.fit_transform(df.iloc[:,:-2])
	return df

def train_test_split_data(df):
	'''
	split the data into a training and test set
	'''
	X = df.iloc[:,:-2].to_numpy()
	y = df.iloc[:,-2].to_numpy()
	xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.25,random_state=0)
	return xtrain,xtest,ytrain,ytest

def decision_tree(xtrain,xtest,ytrain,ytest,exp):
	'''
	A build, fit and predict a decision tree model, with min samples to form a leaf being 2
	'''
	dt = DecisionTreeRegressor(max_features='auto',min_samples_leaf=2,max_depth=10,random_state=exp)
	dt.fit(xtrain,ytrain)
	ypred = dt.predict(xtest)
	rmse = np.sqrt(mean_squared_error(ytest,ypred))
	print('RMSE Decision tree:', rmse)
	return rmse

def random_forest_tree(xtrain,xtest,ytrain,ytest,exp,size):
	'''
	A random forest that takes in an input size, to determine the amount of trees in the ensemble
	'''
	rf = RandomForestRegressor(n_estimators=size,max_depth=10,min_samples_leaf=2,max_leaf_nodes=None,random_state=exp)
	rf.fit(xtrain,ytrain)
	ypred = rf.predict(xtest)
	rmse = np.sqrt(mean_squared_error(ytest,ypred))
	print('RMSE Random forest:', rmse)
	return rmse

def extra_tree(xtrain,xtest,ytrain,ytest,exp,size):
	'''
	A gradient boosted tree, that takes in a size to determine number of trees in the ensemble
	'''
	et = ExtraTreesRegressor(n_estimators=size,max_depth=10,min_samples_leaf=2,max_leaf_nodes=None,random_state=exp)
	et.fit(xtrain,ytrain)
	ypred = et.predict(xtest)
	rmse = np.sqrt(mean_squared_error(ytest,ypred))
	print('RMSE Extra Tree', rmse)
	return rmse

def neural_network(xtrain,xtest,ytrain,ytest,exp):
	'''
	A neural network model with two hidden layers, adam optimizer and learning rate of 0.001
	'''
	nn = MLPRegressor(hidden_layer_sizes=(30,30),learning_rate_init=0.001,solver='adam',max_iter=1000,random_state=exp)
	nn.fit(xtrain,ytrain)
	ypred = nn.predict(xtest)
	rmse = np.sqrt(mean_squared_error(ytest,ypred))
	print('RMSE Neural Network', rmse)
	return rmse

def ten_experinments(xtrain,xtest,ytrain,ytest):
	'''
	Run 10 experiments for each of the model types
	'''
	exp_num = 10
	size = 100
	rmse_dt = np.empty(exp_num)
	rmse_rf = np.empty(exp_num)
	rmse_et = np.empty(exp_num)
	rmse_nn = np.empty(exp_num)
	for exp in range(exp_num):
		rmse_dt[exp] = decision_tree(xtrain,xtest,ytrain,ytest,exp)
		rmse_rf[exp] = random_forest_tree(xtrain,xtest,ytrain,ytest,exp,size)
		rmse_et[exp] = extra_tree(xtrain,xtest,ytrain,ytest,exp,size)
		rmse_nn[exp] = neural_network(xtrain,xtest,ytrain,ytest,exp)
	print('Mean RMSE Decision tree',rmse_dt.mean())
	print('Mean RMSE Random Forest',rmse_rf.mean())
	print('Mean RMSE Extra Tree',rmse_et.mean())
	print('Mean RMSE Neural Network',rmse_nn.mean())

def different_ensemble_sizes(xtrain,xtest,ytrain,ytest):
	'''
	Run 10 experiments for each ensemble size and report mean RMSE for model types
	'''
	exp_num = 10
	ensemble_size = list(range(10,101,20))
	rmse_rf_ens_mean = []
	rmse_et_ens_mean = []
	for size in ensemble_size:
		rmse_rf = np.empty(exp_num)
		rmse_et = np.empty(exp_num)
		for exp in range(exp_num):
			rmse_rf[exp] = random_forest_tree(xtrain,xtest,ytrain,ytest,exp,size)
			rmse_et[exp] = extra_tree(xtrain,xtest,ytrain,ytest,exp,size)
		rmse_rf_ens_mean.append(rmse_rf.mean())
		rmse_et_ens_mean.append(rmse_et.mean())
	print('For ensemble size 20 to 100 by 20')
	print('Random Forest',rmse_rf_ens_mean)
	print('Extra Trees',rmse_et_ens_mean)

def main():
	df = load_data()
	xtrain,xtest,ytrain,ytest=train_test_split_data(df)
	ten_experinments(xtrain,xtest,ytrain,ytest)
	different_ensemble_sizes(xtrain,xtest,ytrain,ytest)

if __name__ == '__main__':
	main()