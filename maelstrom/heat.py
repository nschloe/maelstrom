# -*- coding: utf-8 -*-
#
from dolfin import (
    dx, ds, dot, grad, pi, assemble, lhs, rhs, SpatialCoordinate,
    TrialFunction, TestFunction, KrylovSolver, Function, assemble_system,
    LUSolver
    )


def F(u, v, kappa, rho, cp,
      convection,
      source,
      r,
      neumann_bcs,
      robin_bcs,
      my_dx,
      my_ds
      ):
    '''
    Compute

    .. math::

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

    or its inverse with Dirichlet conditions. Used for time-stepping

    .. math::

        u' = F(u).
    '''
    rho_cp = rho * cp

    F0 = kappa * r * dot(grad(u), grad(v / rho_cp)) * 2*pi * my_dx

    # F -= dot(b, grad(u)) * v * 2*pi*r * dx_workpiece(0)
    b = convection
    if b:
        F0 += (b[0] * u.dx(0) + b[1] * u.dx(1)) * v * 2*pi*r * my_dx

    # Joule heat
    F0 -= source * v / rho_cp * 2*pi*r * my_dx

    # Neumann boundary conditions
    for k, n_grad_T in neumann_bcs.items():
        F0 -= r * kappa * n_grad_T * v / rho_cp * 2*pi * my_ds(k)

    # Robin boundary conditions
    for k, value in robin_bcs.items():
        alpha, u0 = value
        F0 -= \
            r * kappa * alpha * (u - u0) * v / rho_cp * 2*pi * my_ds(k)

    # # Add SUPG stabilization.
    # rho_cp = rho[wpi](background_temp)*cp[wpi]
    # k = kappa[wpi](background_temp)
    # Rdx = u_t * 2*pi*r*dx(wpi) \
    #     + dot(u_1, grad(trial)) * 2*pi*r*dx(wpi) \
    #     - 1.0/(rho_cp) * div(k*r*grad(trial)) * 2*pi*dx(wpi)
    # #F -= dot(tau*u_1, grad(v)) * Rdx
    # #F -= tau * inner(u_1, grad(v)) * 2*pi*r*dx(wpi)
    # #plot(tau, mesh=V.mesh(), title='u_tau')
    # #interactive()
    # #F -= tau * v * 2*pi*r*dx(wpi)
    # #F -= tau * Rdx
    return F0


class Heat(object):
    def __init__(
            self, Q,
            kappa, rho, cp,
            convection,
            source,
            dirichlet_bcs=None,
            neumann_bcs=None,
            robin_bcs=None,
            my_dx=dx,
            my_ds=ds
            ):
        # TODO stabilization
        # About stabilization for reaction-diffusion-convection:
        # http://www.ewi.tudelft.nl/fileadmin/Faculteit/EWI/Over_de_faculteit/Afdelingen/Applied_Mathematics/Rapporten/doc/06-03.pdf
        # http://www.xfem.rwth-aachen.de/Project/PaperDownload/Fries_ReviewStab.pdf
        #
        # R = u_t \
        #     + dot(u0, grad(trial)) \
        #     - 1.0/(rho(293.0)*cp) * div(kappa*grad(trial))
        # F -= R * dot(tau*u0, grad(v)) * dx
        #
        # Stabilization
        # tau = stab.supg2(
        #         mesh,
        #         u0,
        #         kappa/(rho(293.0)*cp),
        #         Q.ufl_element().degree()
        #         )
        super(Heat, self).__init__()
        self.Q = Q

        dirichlet_bcs = dirichlet_bcs or []
        neumann_bcs = neumann_bcs or {}
        robin_bcs = robin_bcs or {}

        self.convection = convection

        u = TrialFunction(Q)
        v = TestFunction(Q)

        # If there are sharp temperature gradients, numerical oscillations may
        # occur. This happens because the resulting matrix is not an M-matrix,
        # caused by the fact that A1 puts positive elements in places other
        # than the main diagonal. To prevent that, it is suggested by
        # Großmann/Roos to use a vertex-centered discretization for the mass
        # matrix part.
        # Check
        # https://bitbucket.org/fenics-project/ffc/issues/145/uflacs-error-for-vertex-quadrature-scheme
        self.M = assemble(
              u * v * dx,
              form_compiler_parameters={
                  'quadrature_rule': 'vertex',
                  'representation': 'quadrature'
                  }
              )

        mesh = Q.mesh()
        r = SpatialCoordinate(mesh)[0]
        self.F0 = F(
                u, v, kappa, rho, cp,
                convection,
                source,
                r,
                neumann_bcs,
                robin_bcs,
                my_dx,
                my_ds
                )

        self.dirichlet_bcs = dirichlet_bcs

        self.A, self.b = assemble_system(-lhs(self.F0), rhs(self.F0))
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

        right_hand_side = - beta * self.b.copy()
        if b:
            right_hand_side += b

        for bc in self.dirichlet_bcs:
            bc.apply(matrix, right_hand_side)

        # TODO proper preconditioner for convection
        if self.convection:
            # solver = LUSolver()
            # Use HYPRE-Euclid instead of ILU for parallel computation.
            solver = KrylovSolver('gmres', 'hypre_euclid')
        else:
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
