import numpy as np
import pandas as pd


##softmax
def softmax(u):
  expu=np.exp(u)
  return expu/np.sum(expu)

##cross entropy cost func
def crossEntropy(p,q):
  return -np.vdot(p,np.log(q))

##cost evaluation
def eval_L(X,Y,beta):
  N=X.shape[0]
  L=0.0
  for i in range(N):
    XiHat=X[i]
    Yi=Y[i]
    qi=softmax(beta @ XiHat)

    L+=crossEntropy(Yi,qi)
  
  return L

##logisticregression function 
def logReg(X,Y,lr):
  numEpochs=5 ##epochs defined
  n,d=X.shape   ##N-n_samples,d-n_features
  X=np.insert(X,0,1,axis=1)
  k=Y.shape[1]
  beta=np.zeros((k,d+1))
  Lvals=[]
  for ep in range(numEpochs):

    L=eval_L(X,Y,beta)
    Lvals.append(L)

    print("Epoch is: "+str(ep)+" Cost is:"+str(L))

    prm=np.random.permutation(n)

    for i in prm:
      XiHat=X[i]
      Yi=Y[i]

      qi=softmax(beta @ XiHat)
      grad_Li=np.outer(qi-Yi,XiHat)

      beta-=lr*grad_Li
    
  return beta,Lvals

##predictor
def predict(X,beta):
  X=np.insert(X,0,1,axis=1)
  N=X.shape[0]

  predictions=[]
  for i in range (N):
    XiHat=X[i]
    qi=softmax(beta @ XiHat)

    p=np.argmax(qi)

    predictions.append(p)
  
  return predictions
