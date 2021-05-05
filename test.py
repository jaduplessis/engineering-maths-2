from unittest import TestCase
import partial_differentiation_functions as pdf
import sympy as sym
from sympy import cos, sin, pi
x, y, z, i, j, k, t, u, v, r, a, b, c, n, L = sym.symbols('x y z i j k t u v r a b c n L')


class Test(TestCase):
    def test1(self):
        self.assertEqual((0, 0, -2.0*cos(pi*n)/(pi*n) + 2.0/(pi*n)),
                         pdf.discrete(-1, 1, 2))

    def test2(self):
        self.assertEqual((1.00000000000000, 1.0*sin(pi*n)/(pi*n), -3.0*cos(pi*n)/(pi*n) + 3.0/(pi*n)),
                         pdf.discrete(-1, 2, 2))

    def test3(self):
        self.assertEqual((3/2, (L*sin(pi*n)/(pi*n) - L*cos(pi*n)/(pi**2*n**2) + L/(pi**2*n**2))/L,
                          (L*cos(pi*n)/(pi*n) - L*sin(pi*n)/(pi**2*n**2))/L),
                         pdf.discrete(1, 1 - t/L, 2*L))
