import sympy as sym
import re
from sympy import cos, sin, pi, sympify, exp
import partial_differentiation_functions as pdf
x, y, z, i, j, k, t, u, v, r, a, b, c, n, L, w, T, s \
    = sym.symbols('x y z i j k t u v r a b c n L w T s')

k = sym.symbols('k', real=True, positive=True)
c = sym.symbols('c', real=True, nonzero=True)
n = sym.symbols('n', integer=True)
A, B, C, D = sym.symbols('A B C D')
X = A*cos(k*x) + B*sin(k*x)
T = C*cos(k*c*t) + D*sin(k*c*t)
u_t = sym.diff(T, t)
u_x = sym.diff(X, x)

conditions = {X: {"term": 'X', "variable": None}, # x, "values": {0: 0, pi: 0}},
              T: {"term": 'T', "variable": t, "values": {0: sin(x)}},
              u_x: {"term": 'u_x', "variable": x, "values": {0: 0, pi: 0}},
              u_t: {"term": 'u_t', "variable": t, "values": {0: 0}}
              }

#conditions = {X: {"term": 'X', "variable": x, "values": {0: 0, pi: 0}},
#              T: {"term": 'T', "variable": t, "values": {0: 0}},
#              u_x: {"term": 'u_x', "variable": None}, # x, "values": {0: 0, pi: 0}},
#              u_t: {"term": 'u_t', "variable": t, "values": {0: sin(x)}}
#              }
period = pi


for keys, values in conditions.items():
    if values['variable'] is not None:
        for initial_value, results in values['values'].items():
            if sym.simplify(results).is_constant():
                variable = values['variable']
                boundary = keys.subs(variable, initial_value)
                solved_boundary = list(sym.solve(boundary, dict=True)[0].items())[0]
                string = str(solved_boundary[1])

                if re.search('tan\(', string):
                    X = X.subs(k, n*pi/initial_value)
                    u_x = u_x.subs(k, n*pi/initial_value)
                    T = T.subs(k, n*pi/initial_value)
                    u_t = u_t.subs(k, n*pi/initial_value)

                else:
                    X = X.subs(solved_boundary[0], solved_boundary[1])
                    u_x = u_x.subs(solved_boundary[0], solved_boundary[1])
                    T = T.subs(solved_boundary[0], solved_boundary[1])
                    u_t = u_t.subs(solved_boundary[0], solved_boundary[1])
            else:
                function = results
                function_value = initial_value
                function_variable = values['variable']
                function_type = values['term']
total_function = b*(T*X).subs({A: 1, B: 1, C: 1, D: 1})

if function_type == 'X' or function_type == 'T':
    diff = total_function.subs(function_variable, function_value)
elif function_type == 'u_x':
    diff = sym.diff(total_function, x).subs(function_variable, function_value)
else:
    diff = sym.diff(total_function, t).subs(function_variable, function_value)

if re.search('sin', str(X)):
    print('sin')
    bn = []
    for i in [1, 2, 3]:
        eq = (diff-function).subs(n, i)
        solution = sym.solve(eq, dict=True)[0]
        if not re.search('sin|cos|tan', str(solution[b])):
            bn.append(solution[b])

    if len(bn) == 1:
        final = total_function.subs(b, bn[0])
        print("Final solution to u(x,t): {}".format(final))
    else:
        b_n, b_1 = pdf.half_range_sine_series_continuous(function, period, x)


elif re.search('cos', str(X)):
    an = []
    for i in [1, 2, 3]:
        eq = (diff-function).subs(n, i)
        solution = sym.solve(eq, dict=True)[0]
        if not re.search('sin|cos|tan', str(solution[b])):
            an.append(solution[b])

    if len(an) == 1:
        final = total_function.subs(b, an[0])
        print("Final solution to u(x,t): {}".format(final))
    else:
        a_n, a_0 = pdf.half_range_cosine_series_continuous(function, period, x)
        const = a_0 / 2
        print("Result is: {} + summation of {}".format(const, a_n*total_function))
