from scipy.optimize import fsolve
import vector_functions as vt
import Stats_functions as st
import sympy as sym
import math
import numpy as np
from numpy import linalg as la

x, y, z, i, j, k, t = sym.symbols('x y z i j k t')

function = [y**4, 2*y**3, 0]
param = [t, sym.exp(-t), 0]
bounds = [0, 1/4]



















































