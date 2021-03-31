from KNN import KNN
import time
# predict labels for batch_size number of test images at a time.
batch_size = 2000 ##predicts in batches of 2000s,not needed,just in case dataset is too large,what number of samples to be trained with can be chosen

kval = [1,3,5,7,9,11,13,15,17,19,21,23,25]##for these values of K,operation is performed


classifier =KNN()
classifier.train(x_train, y_train)


accuracy=[]
for k in kval:
  predictions = []
  
  for i in range(5):   ##5 is just the number of batches to be predicted.....len(x_test)/batch size
    # predicts from i * batch_size to (i+1) * batch_size
    print("Computing batch " + str(i+1) + "/" + str(5) + "...")   ##gives a hacker-hacker feel when a msg pops saying the batch is trained...lol
    tic = time.time()
    predts = classifier.predict(x_test[i * batch_size:(i+1) * batch_size], k)   ##most important...predict function called
    toc = time.time()
    predictions = predictions + list(predts)   ##predictions list updated

    print("Completed this batch in " + str(toc-tic) + " Secs.")
  predictions = np.array(predictions).astype(np.int)   ##list converted to numpy array
  print("Completed predicting the test data foor k = "+str(k))

  acc=np.sum(predictions == y_test)/len(y_test)   ##accuracy function
  accuracy.append(acc)   ##acc is accuracy for each value of K
  
  plt.plot(kval,(accuracy))   ##plots accuracyies against kvals so that the best value of k can be chosen

