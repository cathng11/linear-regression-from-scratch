import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class LinearRegression:
  def __init__(self):
    self.Theta=[]
    self.min_x=[]
    self.max_x=[]
    self.min_y=0
    self.max_y=0

  def normalize(self,data):
    try:
        columnCount = len(data[0])
        for column in range(columnCount):
            data[:, column] = (data[:, column] - np.min(data[:, column]))/(np.max(data[:, column]) - np.min(data[:, column]))
        return data
    except:
        data = (data - np.min(data))/(np.max(data) - np.min(data))
        return data
  def fit(self,X,y, alpha=0.1, epoch=2000):
    for i in range(X.shape[1]):
      self.min_x.append(np.min(X[:,i]))
      self.max_x.append(np.max(X[:,i]))
    self.min_y=np.min(y)
    self.max_y=np.max(y)
    try:
      X,y=MinMaxScaler().fit_transform(np.array(X)),MinMaxScaler(feature_range=(0, 1)).fit_transform(np.array(y).reshape(len(y),-1))    
      y=(y.reshape(-1,len(y)))[0]
    except:
      X,y=normalize(np.array(X)),normalize(np.array(y))
    m = X.shape[0]
    ones =np.ones((m,1))  
    X = np.concatenate((ones, X), axis=1)
    n = X.shape[1]  

    try:
      Theta=[np.median(X) for x in range(n)]
    except:
      Theta = np.ones(n)    
    h = np.dot(X, Theta)   # Compute hypothesis
    # Gradient descent algorithm
    cost = np.ones(epoch)
    accuracy=[]
    for i in range (0, epoch):
      Theta[0] = Theta[0] - (alpha / X.shape[0]) * sum(h-y)
      for j in range(1, n):
        Theta[j]= Theta[j] - (alpha/ X.shape[0]) * sum((h-y) * X[:, j])
      h  = np.dot(X, Theta)
      cost[i] = 1/(2*m) * sum(np.square(h-y)) 
      score_r2=1-(np.sum(((y-h)**2))/np.sum((y-np.mean(y))**2))
      if score_r2 < 0:
        continue
      else:
        accuracy.append(1-(np.sum(((y-h)**2))/np.sum((y-np.mean(y))**2)))
    self.Theta=Theta
    return cost, Theta, accuracy

  def predict(self,X):
    y_predict_list=[]
    for x in X:
      a=[]
      for i in range(len(x)):
        if x[i]-self.min_x[i]==0:
          a.append(0)
        else:
          a.append((x[i]-self.min_x[i])/(self.max_x[i]-self.min_x[i]))    
      X_predict = np.concatenate(([1], np.array(a)), axis = 0)
      y_predict=np.dot(self.Theta, X_predict)
      y_predict_list.append(y_predict*(self.max_y-self.min_y)+self.min_y)
    return y_predict_list

  def score(self,y_pred,y):
    return 1-(np.sum(((y-y_pred)**2))/np.sum((y-np.mean(y))**2))
if __name__ == "__main__":
    X = np.array([[10,5,7,4,4,5,8,8,7,6],[1,2,1,2.5,1,0.5,0.5,1,0.3,0.3]])
    y = np.array([1,1,1,1,0,0,0,1,0,0])
    X = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)
    X = X.transpose()    

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(np.array([[1,4,3]]))
    print(predictions)
    # print("LR classification accuracy:", regressor.score(predictions,y_test))