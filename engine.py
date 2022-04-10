from collections import defaultdict

import numpy as np


class Variable:
    def __init__(self, value, local_gradients=()):
        self.value = value
        self.local_gradients = local_gradients

    def __add__(self, other):
        return add(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __pow__(self, power, modulo=None):
        return pow(self, Variable(power))

    def __neg__(self):
        return self * Variable(-1)

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1)

    def __repr__(self):
        return f'Variable({self.value})'

    def relu(self):
        return relu(self)

    def ln(self):
        return ln(self)

    def exp(self):
        return exp(self)


def add(a, b):
    value = a.value + b.value
    local_gradients = ((a, 1), (b, 1))
    return Variable(value, local_gradients)


def mul(a, b):
    value = a.value * b.value
    local_gradients = ((a, b.value), (b, a.value))
    return Variable(value, local_gradients)


def pow(a, power):
    value = a.value ** power.value
    local_gradients = ((a, power.value * (a.value ** (power.value-1))),)
    return Variable(value, local_gradients)


def exp(a):
    value = np.exp(a.value)
    local_gradients = ((a, value),)
    return Variable(value, local_gradients)


def ln(a):
    value = np.log(a.value)
    local_gradients = ((a, 1/a.value),)
    return Variable(value, local_gradients)


def relu(a):
    value = max(0, a.value)
    local_gradients = ((a, 1 if a.value > 0 else 0),)
    return Variable(value, local_gradients)


def get_gradients(variable):
    gradients = defaultdict(lambda: 0)

    def compute_gradients(variable, path_value):
        for child_variable, local_gradient in variable.local_gradients:
            value_of_path_to_child = path_value * local_gradient
            gradients[child_variable] += value_of_path_to_child
            compute_gradients(child_variable, value_of_path_to_child)

    compute_gradients(variable, 1)
    return gradients