from scipy.optimize import fsolve
import vector_functions as vt
import Stats_functions as st
import sympy as sym
import math
import numpy as np
from numpy import linalg as la
from scipy import integrate
import scipy

#x, y, z, i, j, k, t = sym.symbols('x y z i j k t')




function = lambda x, y, z: x + y + z
z_bounds = [lambda z, y: y - z, lambda z, y: y + z]
y_bounds = [lambda y: 0, lambda y: y]
x_bounds = [0, 1]




print(triple_integral(function, x_bounds, y_bounds, z_bounds))
































