import sympy as sym
import re
from sympy import cos, sin, pi, sympify, exp
x, y, z, i, j, k, t, u, v, r, a, b, c, n, L, w, T, s \
    = sym.symbols('x y z i j k t u v r a b c n L w T s')
A = sym.symbols('A', real=True, positive=True)


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


def discontinuous(function_1, function_2, period):
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
    sub_a2 = re.sub("cos\(pi\*n\)", "(-1)**n", sub_a1)
    A_n = re.sub("sin\(pi\*n\)", "0", sub_a2)
    print("Reformatted version of a_n is: {}".format(A_n))

    sub_b1 = str(b_n)
    sub_b2 = re.sub("cos\(pi\*n\)", "(-1)**n", sub_b1)
    B_n = re.sub("sin\(pi\*n\)", "0", sub_b2)
    print("Reformatted version of b_n is: {}".format(B_n))

    return a_0, a_n, b_n


def half_range_sine_series_continuous(function, period):
    integral = function * sin(n*t*pi / period)
    b_n = sym.integrate(integral, (t, 0, period))
    if b_n != 0:
        for args in b_n.args:
            if sympify(args[0]).is_real is None:
                b_n = 2/period * args[0]
                string = str(b_n)
                sub = re.sub("cos\(pi\*n\)", "(-1)**n", string)
                B_n = re.sub("sin\(pi\*n\)", "0", sub)
                print("Reformatted version of b_n is: {}".format(B_n))
    else:
        print("Reformatted version of a_n is: {}".format(b_n))

    integral_1 = function * sin(pi*t / period)
    b_1 = 2 / period * sym.integrate(integral_1, (t, 0, period))
    print("\nb_n in orignal format is: {}".format(b_n))
    print("b_1 is: {}".format(b_1))
    return b_n, b_1


def half_range_cosine_series_continuous(function, period):
    integral = function * cos(n * t * pi / period)
    a_n = sym.integrate(integral, (t, 0, period))

    if a_n != 0:
        for args in a_n.args:
            if sympify(args[0]).is_real is None:
                a_n = 2/period * args[0]
                string = str(a_n)
                sub = re.sub("cos\(pi\*n\)", "(-1)**n", string)
                A_n = re.sub("sin\(pi\*n\)", "0", sub)
                print("Reformatted version of a_n is: {}".format(A_n))
    else:
        print("Reformatted version of a_n is: {}".format(a_n))

    a_0 = 2 / period * sym.integrate(function, (t, 0, period))
    print("a_n in orignal format is: {}".format(a_n))
    print("a_0 is: {}".format(a_0))

    return a_n, a_0


def half_range_cosine_series_discontinuous(function1, function2, period):
    integral1 = function1 * cos(n * t * pi / period)
    a_n1 = sym.integrate(integral1, (t, 0, period/2))
    for args in a_n1.args:
        if sympify(args[0]).is_real is None:
            a_n1 = args[0]
            break

    integral2 = function2 * cos(n * t * pi / period)
    a_n2 = sym.integrate(integral2, (t, period/2, period))

    for args in a_n2.args:
        if sympify(args[0]).is_real is None:
            a_n2 = args[0]
            break

    a_n = 2 / period * (a_n1 + a_n2)
    string = str(a_n)
    sub = re.sub("cos\(pi\*n\)", "(-1)**n", string)
    A_n = re.sub("sin\(pi\*n\)", "0", sub)
    print("Reformatted version of a_n is: {}".format(A_n))

    a_01 = 2 / period * sym.integrate(function1, (t, 0, period/2))
    a_02 = 2 / period * sym.integrate(function2, (t, period/2, period))
    a_0 = a_01 + a_02
    print("\na_n in orignal format is: {}".format(a_n))
    print("a_0 is: {}".format(a_0))

    return a_n, a_0


def half_range_sine_series_discontinuous(function1, function2, period):
    integral1 = function1 * sin(n * t * pi / period)
    b_n1 = sym.integrate(integral1, (t, 0, period/2))
    for args in b_n1.args:
        if sympify(args[0]).is_real is None:
            b_n1 = args[0]
            break

    integral2 = function2 * sin(n * t * pi / period)
    b_n2 = sym.integrate(integral2, (t, period/2, period))

    for args in b_n2.args:
        if sympify(args[0]).is_real is None:
            b_n2 = args[0]
            break

    b_n = 2 / period * (b_n1 + b_n2)
    string = str(b_n)
    sub = re.sub("cos\(pi\*n\)", "(-1)**n", string)
    B_n = re.sub("sin\(pi\*n\)", "0", sub)
    print("Reformatted version of b_n is: {}".format(B_n))

    term1 = function1 * sin(pi*t / period)
    term2 = function2 * sin(pi*t / period)

    b_01 = 2 / period * sym.integrate(term1, (t, 0, period/2))
    b_02 = 2 / period * sym.integrate(term2, (t, period/2, period))
    b_1 = b_01 + b_02
    print("\nb_n in orignal format is: {}".format(b_n))
    print("b_1 is: {}".format(b_1))

    return b_n, b_1


def fourier_transform(function1, function2, limits1, limits2):
    j = sym.sqrt(-1)
    simplified1, simplified2 = 0, 0
    integral = function1 * exp(-j * w * t)
    ans = sym.integrate(integral, (t, limits1[0], limits1[1])).args
    for args in ans:
        if args[0] != 0 and type(args[0]) != sym.integrals.integrals.Integral:
            simplified1 = sym.simplify(args[0])
    print("Integral of function 1 is: {}".format(simplified1))

    integral = function2 * exp(-j * w * t)
    ans = sym.integrate(integral, (t, limits2[0], limits2[1])).args
    for args in ans:
        if args[0] != 0 and type(args[0]) != sym.integrals.integrals.Integral:
            simplified2 = sym.simplify(args[0])
    print("Integral of function 2 is: {}".format(simplified2))

    total = sym.simplify(simplified1 + simplified2)
    print("Fourier transform is: {}".format(total))
    return total

    # example format
    # function1 = sin(a*t)
    # limits1 = [-pi/a, pi/a]
    # function2 = 0
    # limits2 = [0, 0]


def transfer_function(left_side, right_side):
    j = sym.I
    df2 = j**2 * w**2
    df = j*w

    Y = (left_side[0]*df2 + left_side[1]*df + left_side[2])
    U = (right_side[0]*df2 + right_side[1]*df + right_side[2])
    G = sym.radsimp(U/Y)
    print(G)
    return G
    # example format - input coefficients
    # left_side = [1, 3, 7]
    # right_side = [0, 3, 2]
    # [second order, first order, function]


def laplace_transform(function):
    return sym.laplace_transform(function, t, s, noconds=True)
    # format
    # j = sym.I
    # func = t**2*exp(j*w*t)
    # for H(t-a) = sym.Heaviside(t-a) | function = func * H(t-a)


def inverse_laplace_transform(function):
    return sym.inverse_laplace_transform(function, s, t)
    # format
    # A = sym.symbols('A', real=True, positive=True)
