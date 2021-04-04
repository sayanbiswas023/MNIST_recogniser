import pandas as pd
import matplotlib.pyplot as plt
from Neural_Network import DeepNN

train=pd.read_csv('sample_data/mnist_train_small.csv')
test=pd.read_csv('sample_data/mnist_test.csv')

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
y_train.to_numpy().astype('int')
y_test.to_numpy().astype('int')

##normalise x
x_train=x_train/255.0
x_test=x_test/255.0

##y_train=pd.get_dummies(y_train).values
x_train=x_train.values
x_test=x_test.values




num_iterations = 200000 ## set the number of iterations
learning_rate = 0.01 ## set the base learning rate
num_inputs = 28*28 ## number of inputs
num_outputs = 10 ## number of outputs
hidden_size = 300 ## size of hidden layer

## training and accuracy evaluation
model = DeepNN(num_inputs,hidden_size,num_outputs)
cost_dict, tests_dict = model.train(x_train,y_train,num_iterations=num_iterations,learning_rate=learning_rate)
accu = model.testing(x_test,y_test) 
