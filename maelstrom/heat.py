# -*- coding: utf-8 -*-
#
from dolfin import (
    dx, ds, dot, grad, pi, assemble, lhs, rhs, SpatialCoordinate
    )


class Heat(object):

    def __init__(self, V, u, v, b,
                 kappa, rho, cp,
                 source,
                 dirichlet_bcs=None,
                 neumann_bcs=None,
                 robin_bcs=None,
                 my_dx=dx,
                 my_ds=ds
                 ):
        super(Heat, self).__init__()

        dirichlet_bcs = [] if dirichlet_bcs is None else dirichlet_bcs
        neumann_bcs = {} if neumann_bcs is None else neumann_bcs
        robin_bcs = {} if robin_bcs is None else robin_bcs

        self.dirichlet_bcs = dirichlet_bcs
        self.V = V

        r = SpatialCoordinate(V.mesh())[0]
        self.dx_multiplier = 2*pi*r
        self.dx = my_dx
        self.ds = my_ds

        self.F0 = kappa * r * dot(grad(u), grad(v / (rho * cp))) \
            * 2*pi * my_dx
        # F -= dot(b, grad(u)) * v * 2*pi*r * dx_workpiece(0)
        if b:
            self.F0 += (b[0] * u.dx(0) + b[1] * u.dx(1)) * v * 2*pi*r * my_dx
        # Joule heat
        self.F0 -= 1.0 / (rho * cp) * source * v * 2*pi*r * my_dx
        # Neumann boundary conditions
        for k, nGradT in neumann_bcs.items():
            self.F0 -= r * kappa * nGradT * v / (rho * cp) \
                * 2 * pi * my_ds(k)
        # Robin boundary conditions
        for k, value in robin_bcs.items():
            alpha, u0 = value
            self.F0 -= r * kappa * alpha * (u - u0) * v / (rho * cp) \
                * 2 * pi * ds(k)
        return

    # pylint: disable=unused-argument
    def get_system(self, t):
        # Don't use assemble_system()! See bugs
        # <https://bitbucket.org/fenics-project/dolfin/issue/257/system_assembler-bilinear-and-linear-forms>,
        # <https://bitbucket.org/fenics-project/dolfin/issue/78/systemassembler-problem-with-subdomains-on>.
        return assemble(lhs(self.F0)), assemble(rhs(self.F0))

    def get_preconditioner(self, t):
        return None

    # pylint: disable=unused-argument
    def get_bcs(self, t):
        return self.dirichlet_bcs
