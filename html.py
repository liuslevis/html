from math import *
import numpy as np
from numpy import transpose as trans
from random import random
from sklearn import datasets


def sigmoid(k):
    return 1.0 / (1 + np.exp(-k))

class Model:
    def f(W,x):
        pass
    def loss(X,w,Y):
        pass
    def grad(X,w,Y):
        pass
    def hessian(X,w,Y):
        pass

class LR(Model):
    # return scala
    def f(X,w):
        return 1 - sigmoid(X.dot(w))
    # return scala
    def loss(X,w,Y):
        return abs(sum(Y.dot(X.dot(w)) - np.log(1+np.exp(X.dot(w))))) + np.linalg.norm(w, 2)
    # return w.shape array
    def grad(X,w,Y):
        # grad = np.zeros(w.shape, dtype=np.float64)
        # for i in range(X.shape[0]):
        #     x = X[i]
        #     y = Y[i]
        #     grad += y * x - 1 / (1 + np.exp(x * w)) * x
        # return grad + 0.5 * w
        # Shorter version
        pi = LR.f(X,w)
        return sum((X.T * (Y - pi)).T).T

    def hessian(X,w,Y):
        # calc W
        n = X.shape[0]
        W = np.zeros([n,n])
        for i in range(n):
            for j in range(n):
                W[i][j] = LR.f(X[i],w) * (1 - LR.f(X[j],w))
        H = - np.dot(np.dot(X.T, W), X)
        return H

class LM(Model):
    # return scala
    def f(X,w):
        return X.dot(w)
    # return scala
    def loss(X,w,Y):
        n = X.shape[0]
        return sum((X.dot(w) - Y) ** 2) / n
    # return w.shape array
    def grad(X,w,Y):
        n = X.shape[0]
        return sum(X * 2) / n
    # return len(w)*len(w) array
    def hessian(X,w,Y):
        pass

class Opt:
    def err_rate(f,X,w,Y,threshold=0.5):
        return 1.0 - sum(threshold + f(X,w) > Y) / len(Y)

    def sgd(Model,X,Y,w,batch,learn_rate=0.1,stop_err=0.1, max_iter=20, print_iter=10, classification=True):
        assert X.shape[0]==Y.shape[0],'X Y shape diff'
        assert X.shape[1]==w.shape[0],'X w shape diff'
        assert batch <= X.shape[0]
        print('iter\tloss\ttrain error')
        cnt = 0
        n = X.shape[0]
        while(cnt < max_iter):
            loss = None
            grad = None
            train_err = None
            if batch < n:
                selIdx = np.random.choice(n, batch, replace=False)
                batchX = X[selIdx,:]
                batchY = Y[selIdx]
                loss = Model.loss(batchX,w,batchY)
                grad = Model.grad(batchX,w,batchY)
                train_err = Opt.err_rate(Model.f, batchX, w, batchY, threshold=0.5)
            else:
                loss = Model.loss(X,w,Y)
                grad = Model.grad(X,w,Y)
                train_err = Opt.err_rate(Model.f, X, w, Y, threshold=0.5)

            if cnt % print_iter == 0: 
                print('#%d\t%.2f\t%.2f' % (cnt, loss, train_err))
            cnt += 1
            
            if train_err < stop_err:
                print('#%d\t%.2f\t w%s' % (cnt, train_err, w))
                if classification:
                    print('train error rate:%.2f' % train_err)
                return w
            else:
                #print('grad w', grad.shape, w.shape)
                w = w - grad * learn_rate

    def gd(Model,X,Y,w,learn_rate=0.1,stop_err=0.1, max_iter=20, print_iter=10, classification=True):
        batch = X.shape[0]
        return Opt.sgd(Model,X,Y,w,batch, learn_rate,stop_err, max_iter, print_iter, classification)

    def newton(Model,X,Y,w,learn_rate,stop_err,max_iter=20,print_iter=20,classification=True):
        assert X.shape[0]==Y.shape[0],'X Y shape diff'
        assert X.shape[1]==w.shape[0],'X w shape diff'
        print('iter\tloss\ttrain error')
        cnt = 0
        while(cnt < max_iter):
            loss = Model.loss(X,w,Y)
            g = Model.grad(X,w,Y)
            H = Model.hessian(X,w,Y)
            H_inv = np.linalg.inv(H)
            w = w - H_inv.dot(g)
            train_err = Opt.err_rate(Model.f, X, w, Y, threshold=0.5)
            if cnt % print_iter == 0:
                print('#%d\t%.2f\t%.2f' % (cnt, loss, train_err))
            cnt += 1
            if train_err < stop_err:
                print('train error rate:%.2f' % train_err)
                return w

class Data:
    def load_iris_X_Y_w():
        iris = datasets.load_iris()
        sel = iris.target!=2
        N = sum(sel)
        features = len(iris.feature_names) + 1
        X = np.ones((N, features))
        X[:, :-1] = iris.data[sel]
        Y = iris.target[sel]
        w = np.array([0.3 for i in range(features)])
        return X,Y,w

points = []
points += [[0.1 * i * random(), 0.5, 1] for i in range(100)]
points += [[0.9 * i * random(), 0.2, 1] for i in range(100)]

labels = []
labels += [0 for i in range(100)]
labels += [1 for i in range(100)]

def main():
    X,Y,w = Data.load_iris_X_Y_w()

    print("\n========== Linear Regression Newton ==========")
    w = Opt.newton(LR,
        X=X,
        Y=Y,
        w=w,
        learn_rate=1e-3,
        stop_err=0.05,
        max_iter=100,
        print_iter=1,
        classification=True)
    return 

    print("\n========== Linear Regression GD ==========")
    w = Opt.gd(LR,
        X=X,#np.array(points),
        Y=Y,#np.array(labels),
        w=w,#np.array([random(),random(),1]),
        learn_rate=1e-3,
        stop_err=0.01,
        max_iter=1000,
        print_iter=1,
        classification=True)

    print("\n========== Linear Regression SGD ==========")
    w = Opt.sgd(LR,
        X=np.array(points),
        Y=np.array(labels),
        w=np.array([random(),random(),1]),
        batch=50,
        learn_rate=1e-3,
        stop_err=0.01,
        max_iter=1000,
        print_iter=100,
        classification=True)

    print("\n========== Linear Models ==========")
    w = Opt.gd(LM,
        X=np.array([[0.1,0.2,1],[0.3,0.4,1],[0.5,0.6,1], [0.6,0.7,1]]),
        Y=np.array([0.1, 0.1, 0.2, 0.3]),
        w=np.array([5,10,10]),
        learn_rate=0.01,
        stop_err=0.1,
        max_iter=1000,
        print_iter=50,
        classification=False)

if __name__ == '__main__':
    main()