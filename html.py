from math import *
import numpy as np
from random import random
# np.array([0,1,2]) * np.array([1,2,3])

def sigmoid(k):
    return 1.0 / (1 + np.exp(-k))

# return scala
class LR:
    def f(X,w):
        return 1 - sigmoid(X.dot(w))

    def loss(X,w,Y):
        return abs(sum(Y.dot(X.dot(w)) - np.log(1+np.exp(X.dot(w))))) + np.linalg.norm(w, 2)

    # return w.shape
    def grad(X,w,Y=0):
        grad = np.zeros(w.shape)
        for i in range(X.shape[0]):
            x = X[i]
            y = Y[i]
            grad += y * x - 1 / (1 + np.exp(x * w)) * x
        return grad + 0.5 * w

class LM:
    def f(X,w):
        return X.dot(w)

    def loss(X,w,Y):
        n = X.shape[0]
        return sum((X.dot(w) - Y) ** 2) / n

    def grad(X,w,Y):
        n = X.shape[0]
        return sum(X * 2) / n

def err_rate(f,X,w,Y,threshold=0.5):
    return 1.0 - sum(threshold + f(X,w) > Y) / len(Y)

def gd(model,X,Y,w,learn_rate=0.1,stop_err=0.1, max_iter=20, print_iter=10, classification=True):
    assert X.shape[0]==Y.shape[0],'X Y shape diff'
    assert X.shape[1]==w.shape[0],'X w shape diff'
    
    f = model.f
    f_grad = model.grad
    f_loss = model.loss

    print('iter\tloss')
    cnt = 0
    while(cnt < max_iter):
        loss = f_loss(X,w,Y)
        grad = f_grad(X,w,Y)
        
        if cnt % print_iter == 0: 
            print('#%d\t%.2f' % (cnt, loss))
        cnt += 1
        
        if loss < stop_err:
            print('#%d\t%.2f w%s' % (cnt, loss, w))
            if classification:
                print('train error rate:%.2f' % err_rate(f, X, w, Y, threshold=0.5))
            return w
        else:
            w = w - grad * learn_rate

points = []
points += [[0.1 * i * random(), 0.5, 1] for i in range(100)]
points += [[0.9 * i * random(), 0.2, 1] for i in range(100)]

labels = []
labels += [0 for i in range(100)]
labels += [1 for i in range(100)]

print("\n========== Linear Regression ==========")
w = gd(LR,
    X=np.array(points),
    Y=np.array(labels),
    w=np.array([random(),random(),1]),
    learn_rate=1e-7,
    stop_err=10,
    max_iter=10000,
    print_iter=100,
    classification=True)

print("\n========== Linear Models ==========")
w = gd(LM,
    X=np.array([[0.1,0.2,1],[0.3,0.4,1],[0.5,0.6,1], [0.6,0.7,1]]),
    Y=np.array([0.1, 0.1, 0.2, 0.3]),
    w=np.array([5,10,10]),
    learn_rate=0.01,
    stop_err=1,
    max_iter=1000,
    print_iter=50,
    classification=False)