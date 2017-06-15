# -*- coding: utf-8 -*-
#
from dolfin import (
    dx, ds, dot, grad, pi, assemble, lhs, rhs, SpatialCoordinate,
    TrialFunction, TestFunction, KrylovSolver, Function, assemble_system
    )


class Heat(object):
    '''
    Compute

       \\alpha M u + \\beta F(u) = b

    or its inverse with

        F(u) =
            \\int_\\Omega \\kappa r \\dot(\\grad(u), \\grad(v/\\rho, c_p))
                \\times 2\\pi \, \\text{d}x
            + \\int_\\Omega \\dot(c, \\grad(u)) v \\times 2\\pi r \, \\text{d}x
            - \\int_\\Omega \\frace{1}{\\rho c_p} f v
                \\times 2\\pi r \,\\text{d}x
            - \\int_\\Gamma r \\kappa * \\dot(n,\\grad(T)) v
                \\frac{1}{\\rho c_p} \\times 2\\pi \,\\text{d}s
            - \\int_\\Gamma  r \\kappa  \\alpha (u - u_0) v
                \\frac{1}{\\rho c_p} \\times 2\\pi \,\\text{d}s

    plus Dirichlet conditions.
    '''
    def __init__(
            self, Q, convection,
            kappa, rho, cp,
            source,
            dirichlet_bcs=None,
            neumann_bcs=None,
            robin_bcs=None,
            my_dx=dx,
            my_ds=ds
            ):
        super(Heat, self).__init__()
        self.Q = Q

        dirichlet_bcs = dirichlet_bcs or []
        neumann_bcs = neumann_bcs or {}
        robin_bcs = robin_bcs or {}

        u = TrialFunction(Q)
        v = TestFunction(Q)

        self.M = assemble(u * v * dx)

        self.dirichlet_bcs = dirichlet_bcs

        r = SpatialCoordinate(Q.mesh())[0]
        rho_cp = rho * cp
        b = convection
        # F0 = kappa * r * dot(grad(u), grad(v / rho_cp)) * 2*pi * my_dx

        # # Convection
        # # F -= dot(b, grad(u)) * v * 2*pi*r * dx_workpiece(0)
        # if b:
        #     F0 += (b[0] * u.dx(0) + b[1] * u.dx(1)) * v * 2*pi*r * my_dx

        # # Joule heat
        # F0 -= source * v / rho_cp * 2*pi*r * my_dx

        # # Neumann boundary conditions
        # for k, nGradT in neumann_bcs.items():
        #     F0 -= r * kappa * nGradT * v / rho_cp * 2*pi * my_ds(k)
        # # Robin boundary conditions
        # for k, value in robin_bcs.items():
        #     alpha, u0 = value
        #     F0 -= r * kappa * alpha * (u - u0) * v / rho_cp * 2*pi * ds(k)

        F0 = kappa * r * dot(grad(u), grad(v / rho_cp)) * 2*pi * my_dx
        # F -= dot(b, grad(u)) * v * 2*pi*r * dx_workpiece(0)
        if b:
            F0 += (b[0] * u.dx(0) + b[1] * u.dx(1)) * v * 2*pi*r * my_dx
        # Joule heat
        F0 -= source * v / rho_cp * 2*pi*r * my_dx
        # Neumann boundary conditions
        for k, nGradT in neumann_bcs.items():
            F0 -= r * kappa * nGradT * v / rho_cp * 2*pi * my_ds(k)
        # Robin boundary conditions
        for k, value in robin_bcs.items():
            alpha, u0 = value
            F0 -= \
                r * kappa * alpha * (u - u0) * v / rho_cp * 2*pi * my_ds(k)

        # Don't use assemble_system()! See bugs
        # <https://bitbucket.org/fenics-project/dolfin/issue/257/system_assembler-bilinear-and-linear-forms>,
        # <https://bitbucket.org/fenics-project/dolfin/issue/78/systemassembler-problem-with-subdomains-on>.
        # self.A, self.b = assemble_system(F0)
        self.F0 = F0

        # Don't use assemble_system()! See bugs
        # <https://bitbucket.org/fenics-project/dolfin/issue/257/system_assembler-bilinear-and-linear-forms>,
        # <https://bitbucket.org/fenics-project/dolfin/issue/78/systemassembler-problem-with-subdomains-on>.
        self.A, self.b = assemble_system(lhs(self.F0), rhs(self.F0))
        return

    # pylint: disable=unused-argument
    def eval_alpha_M_beta_F(self, alpha, beta, u, t):
        # Evaluate  alpha * M * u + beta * F(u, t).
        uvec = u.vector()
        return alpha * (self.M * uvec) + beta * (self.A * uvec + self.b)

    def solve_alpha_M_beta_F(self, alpha, beta, b, t):
        # Solve  alpha * M * u + beta * F(u, t) == b  with Dirichlet
        # conditions.
        matrix = alpha * self.M + beta * self.A

        right_hand_side = self.b.copy()
        if b:
            right_hand_side += b

        for bc in self.dirichlet_bcs:
            bc.apply(matrix, right_hand_side)

        solver = KrylovSolver('gmres', 'hypre_amg')
        solver.parameters['relative_tolerance'] = 1.0e-13
        solver.parameters['absolute_tolerance'] = 0.0
        solver.parameters['maximum_iterations'] = 100
        solver.parameters['monitor_convergence'] = True
        solver.set_operator(matrix)

        u = Function(self.Q)
        solver.solve(u.vector(), right_hand_side)
        return u

    def solve_stationary(self):
        return self.solve_alpha_M_beta_F(alpha=0.0, beta=1.0, b=None, t=0.0)


class Heat2(object):

    def __init__(
            self, V, u, v, b,
            kappa, rho, cp,
            source,
            dirichlet_bcs=None,
            neumann_bcs=None,
            robin_bcs=None,
            my_dx=dx,
            my_ds=ds
            ):
        super(Heat2, self).__init__()

        dirichlet_bcs = dirichlet_bcs or []
        neumann_bcs = neumann_bcs or {}
        robin_bcs = robin_bcs or {}

        self.dirichlet_bcs = dirichlet_bcs
        self.V = V

        # r = SpatialCoordinate(V.mesh())[0]
        # self.dx_multiplier = 2*pi*r
        # self.dx = my_dx
        # self.ds = my_ds

        # rho_cp = rho * cp
        # self.F0 = kappa * r * dot(grad(u), grad(v / rho_cp)) * 2*pi * my_dx
        # # F -= dot(b, grad(u)) * v * 2*pi*r * dx_workpiece(0)
        # if b:
        #     self.F0 += (b[0] * u.dx(0) + b[1] * u.dx(1)) * v * 2*pi*r * my_dx
        # # Joule heat
        # self.F0 -= source * v / rho_cp * 2*pi*r * my_dx
        # # Neumann boundary conditions
        # for k, nGradT in neumann_bcs.items():
        #     self.F0 -= r * kappa * nGradT * v / rho_cp * 2*pi * my_ds(k)
        # # Robin boundary conditions
        # for k, value in robin_bcs.items():
        #     alpha, u0 = value
        #     self.F0 -= \
        #         r * kappa * alpha * (u - u0) * v / rho_cp * 2*pi * my_ds(k)

        # self.u = u

        r = SpatialCoordinate(V.mesh())[0]
        self.dx_multiplier = 2*pi*r
        self.dx = my_dx
        self.ds = my_ds

        u = TrialFunction(V)
        v = TestFunction(V)
        rho_cp = rho * cp
        self.F0 = kappa * r * dot(grad(u), grad(v / rho_cp)) * 2*pi * my_dx
        # F -= dot(b, grad(u)) * v * 2*pi*r * dx_workpiece(0)
        if b:
            self.F0 += (b[0] * u.dx(0) + b[1] * u.dx(1)) * v * 2*pi*r * my_dx
        # Joule heat
        self.F0 -= source * v / rho_cp * 2*pi*r * my_dx
        # Neumann boundary conditions
        for k, nGradT in neumann_bcs.items():
            self.F0 -= r * kappa * nGradT * v / rho_cp * 2*pi * my_ds(k)
        # Robin boundary conditions
        for k, value in robin_bcs.items():
            alpha, u0 = value
            self.F0 -= \
                r * kappa * alpha * (u - u0) * v / rho_cp * 2*pi * my_ds(k)
        return

    # pylint: disable=unused-argument
    def get_system(self, t):
        return assemble(lhs(self.F0)), assemble(rhs(self.F0))

    # pylint: disable=unused-argument
    def get_bcs(self, t):
        return self.dirichlet_bcs

    def ssolve(self):
        # solve(self.F0 == 0, self.u, bcs=self.dirichlet_bcs)

        # solver = KrylovSolver('gmres', 'ilu')
        solver = KrylovSolver('gmres', 'hypre_amg')
        solver.parameters['relative_tolerance'] = 1.0e-13
        solver.parameters['absolute_tolerance'] = 0.0
        solver.parameters['maximum_iterations'] = 100
        solver.parameters['monitor_convergence'] = True

        # Don't use assemble_system()! See bugs
        # <https://bitbucket.org/fenics-project/dolfin/issue/257/system_assembler-bilinear-and-linear-forms>,
        # <https://bitbucket.org/fenics-project/dolfin/issue/78/systemassembler-problem-with-subdomains-on>.
        A = assemble(lhs(self.F0))
        b = assemble(rhs(self.F0))

        for bc in self.dirichlet_bcs:
            bc.apply(A, b)

        solver.set_operator(A)
        u = Function(self.V)
        solver.solve(u.vector(), b)
        return u
