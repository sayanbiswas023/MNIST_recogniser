import numpy as np
import pandas as pd


class logisticregression:
    ##init
   def __init__(self,lr):
        self.lr=lr
    ##softmax
   def softmax(self,u):
        expu=np.exp(u)
        return expu/np.sum(expu)
   ##cross entropy cost func
   def crossEntropy(self,p,q):
     return -np.vdot(p,np.log(q))

   ##cost evaluation
   def eval_L(self,X,Y,beta):
     N=X.shape[0]
     L=0.0
     for i in range(N):
       XiHat=X[i] ##augmented feature vector
       Yi=Y[i] ##ground truth
       qi=self.softmax(beta @ XiHat) ##probability vector

       L+=self.crossEntropy(Yi,qi) ##loss for each i
  
     return L

   ##logisticregression function 
   def logReg(self,X,Y,lr):
     numEpochs=100 ##epochs defined
     n,d=X.shape   ##n:n_samples,d:n_features
     X=np.insert(X,0,1,axis=1)
     k=Y.shape[1]
     beta=np.zeros((k,d+1))
     Lvals=[]
     for ep in range(numEpochs):

       L=self.eval_L(X,Y,beta)
       Lvals.append(L)

       print("Epoch is: "+str(ep)+" Cost is:"+str(L))

       prm=np.random.permutation(n) ##random values chosen to perform stochastic gradient descent

       for i in prm:
         XiHat=X[i]
         Yi=Y[i]

         qi=self.softmax(beta @ XiHat)##10X785 @ 785X1
         grad_Li=np.outer(qi-Yi,XiHat) ##gradient of loss

         beta-=lr*grad_Li ##parameter update
    
     return beta,Lvals

   ##predictor
   def predict(self,X,beta):
     X=np.insert(X,0,1,axis=1)
     N=X.shape[0]

     predictions=[]
     for i in range (N):
       XiHat=X[i]
       qi=self.softmax(beta @ XiHat)

       p=np.argmax(qi)

       predictions.append(p)
  
     return predictions
