"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(a, b):
    return a * b

def id(a):
    return a

def add(a, b):
    return a + b

def neg(a):
    return -a

def lt(a, b):
    return a < b

def eq(a, b):
    return a == b

def max(a, b):
    return a if a > b else b

def is_close(a, b):
    return abs(a - b) < 1e-2

def sigmoid(a):
    if a < 0:
        return 1 / (1 + math.exp(-1 * a))
    return math.exp(a) / (1 + math.exp(a))

def relu(a):
    return max(0, a)

def log(a): 
    return math.log(a)

def exp(a):
    return math.exp(a)

def inv(a):
    return 1 / a

def log_back(a, b):
    return b / a

def inv_back(a, b):
    return - b / (a * a)

def relu_back(a, b):
    if a < 0:
        return 0
    return b

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(f: Callable, a: Iterable):
    for item in a: 
        yield f(item)

def zipWith(f: Callable, a: Iterable, b: Iterable):
    for item_a, item_b in zip(a, b):
        yield f(item_a, item_b)

def reduce(f: Callable, a: Iterable):
    res = None
    for item in a:
        if res is None:
            res = item
        else: 
            res = f(res, item)
    return res

def negList(a: list):
    return list(map(neg, a))

def addLists(a: Iterable, b: Iterable):
    return zipWith(add, a, b)

def sum(a: Iterable):
    res = reduce(add, a)
    if res is None:
        return 0
    return res

def prod(a: Iterable):
    return reduce(mul, a)
