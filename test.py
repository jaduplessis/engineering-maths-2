from unittest import TestCase
import partial_differentiation_functions as pdf
import sympy as sym
from sympy import cos, sin, pi, sqrt, exp, Heaviside, gamma
x, y, z, i, j, k, t, u, v, r, a, b, c, n, L, w, T \
    = sym.symbols('x y z i j k t u v r a b c n L w T')


class Test(TestCase):
    def test1(self):
        self.assertEqual((0, 0, -2.0*cos(pi*n)/(pi*n) + 2.0/(pi*n)),
                         pdf.discontinuous(-1, 1, 2))

    def test2(self):
        self.assertEqual((1.00000000000000, 1.0*sin(pi*n)/(pi*n), -3.0*cos(pi*n)/(pi*n) + 3.0/(pi*n)),
                         pdf.discontinuous(-1, 2, 2))

    def test3(self):
        self.assertEqual((3/2, (L*sin(pi*n)/(pi*n) - L*cos(pi*n)/(pi**2*n**2) + L/(pi**2*n**2))/L,
                          (L*cos(pi*n)/(pi*n) - L*sin(pi*n)/(pi**2*n**2))/L),
                         pdf.discontinuous(1, 1 - t/L, 2*L))

#    def test4(self):
#        self.assertEqual(2*a*sin(T*w)/w, pdf.fourier_transform(a, [-T, T]))

    def test5(self):
        self.assertEqual((sqrt(5)*(-sin(sqrt(5)*t/2) + sqrt(5)*cos(sqrt(5)*t/2))*exp(t/2)*Heaviside(t)/5),
                         pdf.initial_value_problems_laplace([2, -2, 3], 0, 1, 0))


