import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logistic_model import logReg,predict

##reading data
train=pd.read_csv('sample_data/mnist_train_small.csv')
test=pd.read_csv('sample_data/mnist_test.csv')

##creating dataframes for train and test
df = pd.DataFrame(train)
dff = pd.DataFrame(test)

##extracting y out of dataset
y_train=df.iloc[:,0]
y_test=dff.iloc[:,0]

##extracting x out of dataset
x_train=df.drop(df.columns[0], axis=1)
x_test=dff.drop(dff.columns[0], axis=1)


##converting dataframe to numpy array
x_train.to_numpy()
x_test.to_numpy()
y_train.to_numpy()
y_test.to_numpy()

##normalise x
x_train=x_train/255.0
x_test=x_test/255.0

y_train=pd.get_dummies(y_train).values
x_train=x_train.values
x_test=x_test.values



##returns parameters and loss from gradient descent
lr=0.001
beta,Lvals=logReg(x_train,y_train,lr)

##predicted list
predictions=predict(x_test,beta)


##accuracy tester
c=x_test.shape[0]
r=0
for i in range(c):
  if predictions[i]==y_test[i]:
    r+=1
accuracy=(r/c)*100

print ("accuracy:"+str(accuracy))
