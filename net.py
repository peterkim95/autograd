import random

from engine import Variable


class Neuron:
    def __init__(self, nin):
        self.w = [Variable(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Variable(0)

    def __call__(self, x):
        z = self.b
        for i in range(len(x)):
            z += x[i] * self.w[i]
        return z.relu()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [n.parameters() for n in self.neurons]


class Network:
    def __init__(self, dim):
        assert len(dim) >= 2
        self.layers = [Layer(dim[i-1], dim[i]) for i in range(1, len(dim))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        x = [a.exp() for a in x]
        denom = Variable(0)
        for a in x:
            denom += a
        x = [a / denom for a in x]
        return x

    def parameters(self):
        return [layer.parameters() for layer in self.layers]
