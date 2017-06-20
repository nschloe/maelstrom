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
            \\int_\\Omega \\kappa r
                \\langle\\nabla(u), \\nabla(v/\\rho, c_p)\\rangle
                \\times 2\\pi \\, \\text{d}x
            + \\int_\\Omega \\langle c, \\nabla(u)\\rangle v
                \\times 2\\pi r\\,\\text{d}x
            - \\int_\\Omega \\frac{1}{\\rho c_p} f v
                \\times 2\\pi r \\,\\text{d}x\\\\
            - \\int_\\Gamma r \\kappa * \\langlen,\\nabla(T)\\rangle v
                \\frac{1}{\\rho c_p} \\times 2\\pi \\,\\text{d}s
            - \\int_\\Gamma  r \\kappa  \\alpha (u - u_0) v
                \\frac{1}{\\rho c_p} \\times 2\\pi \\,\\text{d}s

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
        F0 -= r * kappa * alpha * (u - u0) * v / rho_cp * 2*pi * my_ds(k)

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
        # Convert to proper `float`s to avoid accidental conversion to
        # numpy.arrays, cf.
        # <https://bitbucket.org/fenics-project/dolfin/issues/874/genericvector-numpyfloat-numpyarray-not>
        alpha = float(alpha)
        beta = float(beta)
        return alpha * (self.M * uvec) + beta * (self.A * uvec + self.b)

    def solve_alpha_M_beta_F(self, alpha, beta, b, t):
        # Solve  alpha * M * u + beta * F(u, t) == b  with Dirichlet
        # conditions.
        matrix = alpha * self.M + beta * self.A

        # See above for float conversion
        right_hand_side = - float(beta) * self.b.copy()
        if b:
            right_hand_side += b

        for bc in self.dirichlet_bcs:
            bc.apply(matrix, right_hand_side)

        # TODO proper preconditioner for convection
        if self.convection:
            # Use HYPRE-Euclid instead of ILU for parallel computation.
            # However, this PC sometimes fails.
            # solver = KrylovSolver('gmres', 'hypre_euclid')
            # Fallback:
            solver = LUSolver()
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
