# -*- coding: utf-8 -*-
#
from dolfin import dx, ds, Expression, dot, grad, pi, assemble, \
    lhs, rhs

import time_steppers as ts


class HeatCylindrical(ts.ParabolicProblem):

    def __init__(self, V, u, v, b,
                 kappa, rho, cp,
                 source,
                 dirichlet_bcs=[],
                 neumann_bcs={},
                 robin_bcs={},
                 dx=dx,
                 ds=ds
                 ):
        super(HeatCylindrical, self).__init__()
        self.dirichlet_bcs = dirichlet_bcs
        r = Expression('x[0]', degree=1, domain=V.mesh())
        self.V = V
        self.dx_multiplier = 2*pi*r

        self.F0 = kappa * r * dot(grad(u), grad(v / (rho * cp))) \
            * 2*pi * dx
        # F -= dot(b, grad(u)) * v * 2*pi*r * dx_workpiece(0)
        if b:
            self.F0 += (b[0] * u.dx(0) + b[1] * u.dx(1)) * v * 2*pi*r * dx
        # Joule heat
        self.F0 -= 1.0 / (rho * cp) * source * v * 2*pi*r * dx
        # Neumann boundary conditions
        for k, nGradT in neumann_bcs.iteritems():
            self.F0 -= r * kappa * nGradT * v / (rho * cp) \
                * 2 * pi * ds(k)
        # Robin boundary conditions
        for k, value in robin_bcs.iteritems():
            alpha, u0 = value
            self.F0 -= r * kappa * alpha * (u - u0) * v / (rho * cp) \
                * 2 * pi * ds(k)
        return

    def get_system(self, t):
        # Don't use assemble_system()! See bugs
        # <https://bitbucket.org/fenics-project/dolfin/issue/257/system_assembler-bilinear-and-linear-forms>,
        # <https://bitbucket.org/fenics-project/dolfin/issue/78/systemassembler-problem-with-subdomains-on>.
        return assemble(lhs(self.F0)), assemble(rhs(self.F0))

    def get_bcs(self, t):
        return self.dirichlet_bcs