from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

from engine import Variable, get_gradients

X, y = make_regression(n_samples=100, n_features=1, noise=10)
# print(X,y)

# Linear regression
# y = a * x + b
a = Variable(1)
b = Variable(1)

Js = []
alpha = 0.1
for e in range(100):
    J = Variable(0)
    for i in range(len(X)):
        y_pred = a * Variable(X[i][0]) + b
        y_true = Variable(y[i])
        J += (y_pred - y_true) ** 2
    J /= Variable(len(X))
    Js.append(J.value)
    print(f'Epoch = {e}, J = {J.value}')
    gradients = get_gradients(J)
    a.value -= alpha * gradients[a]
    b.value -= alpha * gradients[b]

plt.plot(Js)
plt.show()