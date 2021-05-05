import sympy as sym
import re
from sympy import cos, sin, pi
x, y, z, i, j, k, t, u, v, r, a, b, c, n, L = sym.symbols('x y z i j k t u v r a b c n L')


def continuous(function, period):
    a_0 = sym.integrate(function, (t, -period / 2, period / 2))
    a_0 = 2 / period * a_0
    print("a_0 value is: " + str(a_0))

    integral_b = function * sin(n * t * 2*pi / period)
    integral_a = function * cos(n * t * 2*pi / period)

    b_n = sym.integrate(integral_b, (t, -period / 2, period / 2))
    a_n = sym.integrate(integral_a, (t, period / 2, -period / 2))

    if a_n != 0:
        sub_a1 = str(2 / period * a_n.args[0][0])
        sub_a2 = re.sub(r"cos\(pi\*n\)", "(-1)**n", sub_a1)
        a_n = re.sub("sin\(pi\*n\)", "0", sub_a2)

    if b_n != 0:
        sub_b1 = str(2 / period * b_n.args[0][0])
        sub_b2 = re.sub(r"cos\(pi\*n\)", "(-1)**n", sub_b1)
        b_n = re.sub("sin\(pi\*n\)", "0", sub_b2)

    return a_n, b_n


def discrete(function_1, function_2, period):
    a_01 = sym.integrate(function_1, (t, -period / 2, 0))
    a_02 = sym.integrate(function_2, (t, 0, period / 2))
    a_0 = 2 / period * (a_01 + a_02)
    print("a_0 value is: " + str(a_0))

    integral_a1 = function_1 * cos(n * t * 2*pi / period)
    integral_a2 = function_2 * cos(n * t * 2*pi / period)

    integral_b1 = function_1 * sin(n * t * 2*pi / period)
    integral_b2 = function_2 * sin(n * t * 2*pi / period)

    a_n1 = sym.integrate(integral_a1, (t, -period / 2, 0))
    if a_n1 != 0:
        a_n1 = a_n1.args[0][0]

    a_n2 = sym.integrate(integral_a2, (t, 0, period / 2))
    if a_n2 != 0:
        a_n2 = a_n2.args[0][0]

    b_n1 = sym.integrate(integral_b1, (t, -period / 2, 0))
    if b_n1 != 0:
        b_n1 = b_n1.args[0][0]

    b_n2 = sym.integrate(integral_b2, (t, 0, period / 2))
    if b_n2 != 0:
        b_n2 = b_n2.args[0][0]

    a_n = 2 / period * (a_n1 + a_n2)
    b_n = 2 / period * (b_n1 + b_n2)

    sub_a1 = str(a_n)
    sub_a2 = re.sub(r"cos\(pi\*n\)", "(-1)**n", sub_a1)
    A_n = re.sub("sin\(pi\*n\)", "0", sub_a2)
    print("Reformatted version of a_n is: {}".format(A_n))

    sub_b1 = str(b_n)
    sub_b2 = re.sub(r"cos\(pi\*n\)", "(-1)**n", sub_b1)
    B_n = re.sub("sin\(pi\*n\)", "0", sub_b2)
    print("Reformatted version of b_n is: {}".format(B_n))

    return a_0, a_n, b_n
