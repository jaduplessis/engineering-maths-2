from scipy.optimize import fsolve
import scipy as sci
from vector_functions import *
from Stats_functions import *
import sympy as sym
import math
import numpy as np
from numpy import linalg as la
import re
from sympy import cos, sin, pi, sympify, exp, pprint, init_printing, simplify
from sympy.vector import ParametricRegion, vector_integrate
import matplotlib.pyplot as plt
from partial_differentiation_functions import *
init_printing()

x, y, z, i, j, k, t, u, v, r, a, b, c, n, L, w, T, s \
    = sym.symbols('x y z i j k t u v r a b c n L w T s')


field = x**2 -3*x + x*y + y**2 + z**3 - 3*z
print(stationary_values_3variables(field))

set1 = [10, 15, 25]
set2 = [20, 25, 5]
sig = 0.05

print(independence_test(sig, [set1, set2]))


































































































































































































