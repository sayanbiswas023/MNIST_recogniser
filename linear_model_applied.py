import numpy as np
import pandas as pd
from linear_model import linearregression

def accuracy(y_true,y_pred):
    accuracy=np.sum(y_true==y_pred)/len(y_true)
    return accuracy

##reading the data
train=pd.read_csv('sample_data/mnist_train_small.csv')
test=pd.read_csv('sample_data/mnist_test.csv')

##dataframes splitted
df = pd.DataFrame(train)
dff = pd.DataFrame(test)

df.to_numpy()
dff.to_numpy()

##y extracted
y_train=df.iloc[:,0]
y_test=dff.iloc[:,0]

##x extracted
x_train=df.drop(df.columns[0], axis=1)
x_test=dff.drop(dff.columns[0], axis=1)

##dataframes converted to numpy array
x_train=x_train.to_numpy()
x_test=x_test.to_numpy()
y_train=y_train.to_numpy()
y_test=y_test.to_numpy()

#making a copy of y_test for performing accuracy tests
y_check=y_test

##normalising the numpy arrays
x_train = (x_train  - x_train .min()) / (x_train .max() - x_train .min())
y_train = (y_train  - y_train .min()) / (y_train .max() - y_train .min())
x_test = (x_test  - x_test .min()) / (x_test .max() - x_test .min())
y_test = (y_test  - y_test .min()) / (y_test .max() - y_test .min())

##training data fitted
regressor = linearregression(learning_rate=0.05, n_iters=1000)
regressor.fit(x_train, y_train)

##making predictions
predictions = regressor.predict(x_test)

##accuracy check
print("Accuracy:",accuracy(y_test,predictions))
