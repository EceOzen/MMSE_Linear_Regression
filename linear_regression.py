# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:22:23 2023

@author: Ece
"""
# Import library
import pandas  as pd #Data manipulation
import numpy as np #Data manipulation
import matplotlib.pyplot as plt # Visualization
import seaborn as sns #Visualization
plt.rcParams['figure.figsize'] = [8,5]
plt.rcParams['font.size'] =14
plt.rcParams['font.weight']= 'bold'
plt.style.use('seaborn-whitegrid')
from scipy.stats import boxcox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Import dataset
df = pd.read_csv('insurance.csv')
print('\nNumber of rows and columns in the data set: ',df.shape)
print('')

#Lets look into top few rows and columns in the dataset
df.head()

df.plot(kind='scatter', 
        x='bmi', 
        y='charges', 
        color='red')
plt.xlabel('Boby Mass Index$(kg/m^2)$: as Independent variable')
plt.ylabel('Insurance Charges: as Dependent variable')
plt.title('Charge Vs BMI')
plt.show()

df.describe()

# correlation plot
corr = df.corr()
sns.heatmap(corr, cmap = 'Wistia', annot= True);

# Dummy variable
categorical_columns = ['sex','children', 'smoker', 'region']
df_encode = pd.get_dummies(data = df, prefix = 'OHE', prefix_sep='_',
               columns = categorical_columns,
               drop_first =True,
              dtype='int8')

# Lets verify the dummay variable process
print('Columns in original data frame:\n',df.columns.values)
print('\nNumber of rows and columns in the dataset:',df.shape)
print('\nColumns in data frame after encoding dummy variable:\n',df_encode.columns.values)
print('\nNumber of rows and columns in the dataset:',df_encode.shape)

y_bc,lam, ci= boxcox(df_encode['charges'],alpha=0.05)

#df['charges'] = y_bc  
# it did not perform better for this model, so log transform is used
ci,lam

## Log transform
df_encode['charges'] = np.log(df_encode['charges'])

X = df_encode.drop('charges',axis=1) # Independet variable
y = df_encode['charges'] # dependent variable

MMSE = 1000
MMSE_SK = 1000

for x in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=23)

    # Step 1: add x0 =1 to dataset
    X_train_0 = np.c_[np.ones((X_train.shape[0],1)),X_train]
    X_test_0 = np.c_[np.ones((X_test.shape[0],1)),X_test]

    # Step2: build model
    theta = np.matmul(np.linalg.inv( np.matmul(X_train_0.T,X_train_0) ), np.matmul(X_train_0.T,y_train)) 

    # The parameters for linear regression model
    parameter = ['theta_'+str(i) for i in range(X_train_0.shape[1])]
    columns = ['intersect:x_0=1'] + list(X.columns.values)
    parameter_df = pd.DataFrame({'Parameter':parameter,'Columns':columns,'theta':theta})

    lin_reg = LinearRegression()
    lin_reg.fit(X_train,y_train) # Note: x_0 =1 is no need to add, sklearn will take care of it.

    #Parameter
    sk_theta = [lin_reg.intercept_]+list(lin_reg.coef_)
    parameter_df = parameter_df.join(pd.Series(sk_theta, name='Sklearn_theta'))
    parameter_df

    # Normal equation
    y_pred_norm =  np.matmul(X_test_0,theta)

    #Evaluvation: MSE
    J_mse = np.sum((y_pred_norm - y_test)**2)/ X_test_0.shape[0]
    if(J_mse < MMSE):
        MMSE = J_mse

    # R_square 
    sse = np.sum((y_pred_norm - y_test)**2)
    sst = np.sum((y_test - y_test.mean())**2)
    R_square = 1 - (sse/sst)
    print('The Mean Square Error(MSE) or J(theta) is: ',J_mse)
    print('R square obtain for normal equation method is :',R_square)

    # sklearn regression module
    y_pred_sk = lin_reg.predict(X_test)

    #Evaluvation: MSE
    J_mse_sk = mean_squared_error(y_pred_sk, y_test)
    if(J_mse_sk < MMSE_SK):
        MMSE_SK = J_mse_sk

    # R_square
    R_square_sk = lin_reg.score(X_test,y_test)
    print('The Mean Square Error(MSE) or J(theta) is: ',J_mse_sk)
    print('R square obtain for scikit learn library is :',R_square_sk)
    
    


print(J_mse)
print(J_mse_sk)

