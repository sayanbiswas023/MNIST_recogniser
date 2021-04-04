import numpy as np
import copy

class DeepNN:
    first_layer = {} ##dictionary for weight and bias of 1st layer
    second_layer = {} ##dictionary for weight and bias of 1st layer

    def __init__(self, inputs, hidden, outputs):
        ## initialize the model weights and biases of the first and second hidden layer 
        ##hidden:no. of neurones in each layer(taken equal)
        self.first_layer['weight'] = np.random.randn(hidden,inputs) / np.sqrt(inputs)
        self.first_layer['bias'] = np.random.randn(hidden,1) / np.sqrt(hidden)
        self.second_layer['weight'] = np.random.randn(outputs,hidden) / np.sqrt(hidden)
        self.second_layer['bias'] = np.random.randn(outputs,1) / np.sqrt(hidden)
        self.input_size = inputs
        self.hid_size = hidden
        self.output_size = outputs

    def __activfunc(self,Z,type = 'ReLU',deri = False):##3 possible activation functions can be taken
        # implement the activation function
        if type == 'ReLU':
            if deri == True:
                return np.array([1 if i>0 else 0 for i in np.squeeze(Z)])
            else:
                return np.array([i if i>0 else 0 for i in np.squeeze(Z)])
        elif type == 'Sigmoid':
            if deri == True:
                return 1/(1+np.exp(-Z))*(1-1/(1+np.exp(-Z)))
            else:
                return 1/(1+np.exp(-Z))
        
    def __Softmax(self,z): ## softmax function
        return 1/sum(np.exp(z)) * np.exp(z)

    def __cross_entropy_error(self,v,y): ## cross entropy error
        return -np.log(v[y])

    def __forward(self,x,y): ##forward process,find prediction list and respective costs
    
        Z = np.matmul(self.first_layer['weight'],x).reshape((self.hid_size,1)) + self.first_layer['bias']##unactivated output of first layer
        H = np.array(self.__activfunc(Z)).reshape((self.hid_size,1))##1st layer output activated
        U = np.matmul(self.second_layer['weight'],H).reshape((self.output_size,1)) + self.second_layer['bias']##unactivated output of second layer
        predict_list = np.squeeze(self.__Softmax(U))##activated output of second layer and = the final output
        error = self.__cross_entropy_error(predict_list,y)
        
        dic = {
            'Z':Z,
            'H':H,
            'U':U,
            'f_X':predict_list.reshape((1,self.output_size)),
            'error':error
        }
        return dic

    def __back_propagation(self,x,y,f_result):## back propagation process to compute the gradients
        E = np.array([0]*self.output_size).reshape((1,self.output_size))##vector with all 0 elements..here of size 10X1
        E[0][y] = 1 ##one_hot
        dU = (-(E - f_result['f_X'])).reshape((self.output_size,1))## matrix subtraction for each of the labels
        db2 = copy.copy(dU) ##d(b2)=dU
        dW2 = np.matmul(dU,f_result['H'].transpose())##dW=matrix mult of gradient of output and trnspose of input...elementwise
        delta = np.matmul(self.second_layer['weight'].transpose(),dU)
        db1 = delta.reshape(self.hid_size,1)*self.__activfunc(f_result['Z'],deri=True).reshape(self.hid_size,1)##error wrt input=error wrt output @ derivative of activation function
        dW1 = np.matmul(db1.reshape((self.hid_size,1)),x.reshape((1,784)))##dW=matrix mult of gradient of output and trnspose of input...elementwise

        grad = {
            'dW2':dW2,
            'db2':db2,
            'db1':db1,
            'dW1':dW1
        }
        return grad ## dictionary of gradients

    def __optimize(self,b_result, learning_rate): ## update the parameters
        self.second_layer['weight'] -= learning_rate*b_result['dW2']
        self.second_layer['bias'] -= learning_rate*b_result['db2']
        self.first_layer['bias'] -= learning_rate*b_result['db1']
        self.first_layer['weight'] -= learning_rate*b_result['dW1']

    def __loss(self,x_train,y_train):## implement the loss function on the entire training set
        loss = 0
        for i in range(len(x_train)):
            y = y_train[i]
            x = x_train[i][:]
            loss += self.__forward(x,y)['error']
        return loss

    def train(self, x_train, y_train, num_iterations = 1000, learning_rate = 0.5): ## generate a random list of indices for the training set
        rand_indices = np.random.choice(len(x_train), num_iterations, replace=True)
        
        def l_rate(base_rate, ite, num_iterations, schedule = False):## determine whether to use the learning schedule
            if schedule == True:
                return base_rate * 10 ** (-np.floor(ite/num_iterations*5))
            else:
                return base_rate

        count = 1
        loss_dict = {}
        test_dict = {}

        for i in rand_indices:
            f_result = self.__forward(x_train[i],y_train[i]) ##calls forward path
            b_result = self.__back_propagation(x_train[i],y_train[i],f_result) ##calls bacward path
            self.__optimize(b_result,l_rate(learning_rate,i,num_iterations,True))
            
            if count % 1000 == 0:
                if count % 5000 == 0: ##shows loss and accuracy every 5000 iterations
                    loss = self.__loss(x_train,y_train)
                    test = self.testing(x_test,y_test)
                    print('Training for {} iterations complete,'.format(count),'Loss = {}'.format(loss))
                    loss_dict[str(count)]=loss
                    test_dict[str(count)]=test
                else: ##shows progress every 1000 iters
                    print('Training for {} iterations complete,'.format(count))
            count += 1

        print('Training finished!')
        return loss_dict, test_dict

    def testing(self,x_test, y_test): ## test the model on the training dataset
        total_correct = 0
        for n in range(len(x_test)):
            y = y_test[n]
            x = x_test[n][:]
            prediction = np.argmax(self.__forward(x,y)['f_X'])
            if (prediction == y):
                total_correct += 1
        print('Accuarcy Test: ',total_correct*100/len(x_test))
        return total_correct/np.float(len(x_test))
