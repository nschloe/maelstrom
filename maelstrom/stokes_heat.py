# -*- coding: utf-8 -*-
#
from dolfin import (
    NonlinearProblem, dx, ds, Function, split, TestFunctions, as_vector,
    assemble, derivative, Constant, Expression, inner, triangle, grad, pi, dot,
    DirichletBC, FunctionSpace, info, TrialFunction, norm
    )

from . import heat


class FixedPointSolver(object):
    '''
    Fixed-point iteration for
        F(x) + x = x
    '''
    def __init__(self):
        self.parameters = {
                'maximum_iterations': 0,
                'absolute_tolerance': 0.0,
                'relative_tolerance': 0.0,
                'report': True
                }
        return

    def _report(self, k, nrm, nrm0):
        if self.parameters['report']:
            print(('Fixed-point iteration %d:' % k) +
                  (' r (abs) = %e (tol = %e)'
                   % (nrm, self.parameters['absolute_tolerance'])) +
                  (' r (rel) = %e (tol = %e)'
                   % (nrm/nrm0, self.parameters['relative_tolerance']))
                  )
        return

    def solve(self, problem, vector):
        res = vector.copy()
        problem.F(res, vector)
        nrm = norm(res)
        nrm0 = nrm
        k = 0
        self._report(k, nrm, nrm0)
        while nrm > self.parameters['absolute_tolerance'] \
                or nrm / nrm0 > self.parameters['relative_tolerance']:
            problem.F(res, vector)
            vector[:] += res
            nrm = norm(res)
            k += 1
            self._report(k, nrm, nrm0)
        return


class Stokes(object):

    def __init__(self, u, p, v, q, f):
        super(Stokes, self).__init__()
        self.u = u
        self.p = p
        self.v = v
        self.q = q
        self.f = f
        return

    def F0(self, mu):
        u = self.u
        p = self.p
        v = self.v
        q = self.q
        f = self.f
        mu = Constant(mu)
        # Momentum equation (without the nonlinear term).
        r = Expression('x[0]', degree=1, cell=triangle)
        F0 = mu * inner(r * grad(u), grad(v)) * 2 * pi * dx \
            + mu * u[0] / r * v[0] * 2 * pi * dx \
            - dot(f, v) * 2 * pi * r * dx
        if len(u) == 3:
            # Discard nonlinear component.
            F0 += mu * u[2] / r * v[2] * 2 * pi * dx
        F0 += (p.dx(0) * v[0] + p.dx(1) * v[1]) * 2 * pi * r * dx

        # Incompressibility condition.
        # div_u = 1/r * div(r*u)
        F0 += (1.0 / r * (r * u[0]).dx(0) + u[1].dx(1)) * q \
            * 2 * pi * r * dx
        return F0


def dbcs_to_productspace(W, bcs_list):
    new_bcs = []
    for k, bcs in enumerate(bcs_list):
        for bc in bcs:
            C = bc.function_space().component()
            # pylint: disable=len-as-condition
            if len(C) == 0:
                new_bcs.append(DirichletBC(W.sub(k),
                                           bc.value(),
                                           bc.domain_args[0]))
            elif len(C) == 1:
                new_bcs.append(DirichletBC(W.sub(k).sub(int(C[0])),
                                           bc.value(),
                                           bc.domain_args[0]))
            else:
                raise RuntimeError('Illegal number of subspace components.')
    return new_bcs


class StokesHeat(NonlinearProblem):

    def __init__(self, WPQ,
                 kappa, rho, mu, cp,
                 theta_average,
                 g, extra_force,
                 heat_source,
                 u_bcs, p_bcs,
                 theta_dirichlet_bcs=None,
                 theta_neumann_bcs=None,
                 theta_robin_bcs=None,
                 my_dx=dx,
                 my_ds=ds
                 ):

        theta_dirichlet_bcs = theta_dirichlet_bcs or {}
        theta_neumann_bcs = theta_neumann_bcs or {}
        theta_robin_bcs = theta_robin_bcs or {}

        super(StokesHeat, self).__init__()
        # Translate the boundary conditions into the product space.
        self.bcs = dbcs_to_productspace(
            WPQ,
            [u_bcs, p_bcs, theta_dirichlet_bcs]
            )

        self.uptheta = Function(WPQ)
        u, p, theta = split(self.uptheta)
        v, q, zeta = TestFunctions(WPQ)

        # Right-hand side for momentum equation.
        f = rho(theta) * g
        if extra_force is not None:
            f += as_vector((extra_force[0], extra_force[1], 0.0))

        self.stokes = Stokes(u, p, v, q, f)

        self.heat = heat.HeatCylindrical(
            WPQ.sub(2), theta, zeta,
            u,
            kappa, rho(theta_average), cp,
            source=heat_source,
            dirichlet_bcs=theta_dirichlet_bcs,
            neumann_bcs=theta_neumann_bcs,
            robin_bcs=theta_robin_bcs,
            my_dx=my_dx,
            my_ds=my_ds
            )

        self.F0 = self.stokes.F0(mu) + self.heat.F0
        self.jacobian = derivative(self.F0, self.uptheta)
        return

    def set_parameter(self, mu):
        self.F0 = self.stokes.F0(mu) + self.heat.F0
        self.jacobian = derivative(self.F0, self.uptheta)
        return

    def F(self, b, x):
        self.uptheta.vector()[:] = x
        assemble(self.F0,
                 tensor=b,
                 form_compiler_parameters={'optimize': True}
                 )
        for bc in self.bcs:
            bc.apply(b, x)
        return

    def J(self, A, x):
        self.uptheta.vector()[:] = x
        assemble(self.jacobian,
                 tensor=A,
                 form_compiler_parameters={'optimize': True}
                 )
        for bc in self.bcs:
            bc.apply(A)
        return


def solve():
    return _solve_newton()

def _solve_newton():
    WPQ = FunctionSpace(
        mesh, MixedElement([W_element, P_element, Q_element])
        )
    uptheta0 = Function(WPQ)

    # Initial guess
    assign(uptheta0.sub(0), u0)
    assign(uptheta0.sub(1), p0)
    assign(uptheta0.sub(2), theta0)

    stokes_heat_problem = StokesHeat(
        WPQ,
        k_wpi_const,
        rho_wpi,
        mu_wpi(theta_average),
        cp_wpi_const,
        theta_average,
        g, extra_force,
        joule_wpi,
        u_bcs, p_bcs,
        theta_dirichlet_bcs=theta_bcs_d,
        theta_neumann_bcs=theta_bcs_n,
        my_dx=dx_submesh,
        my_ds=ds_submesh
        )
    # solver = FixedPointSolver()
    from dolfin import PETScSNESSolver
    solver = PETScSNESSolver()
    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/SNES/SNESType.html
    solver.parameters['method'] = 'newtonls'
    # The Jacobian system for Stokes (+heat) are hard to solve.
    # Use LU for now.
    solver.parameters['linear_solver'] = 'lu'
    solver.parameters['maximum_iterations'] = 5
    solver.parameters['absolute_tolerance'] = 1.0e-10
    solver.parameters['relative_tolerance'] = 0.0
    solver.parameters['report'] = True

    solver.solve(stokes_heat_problem, uptheta0.vector())

    # u0, p0, theta0 = split(uptheta0)
    # Create a *deep* copy of u0, p0, theta0 to be able to deal with them
    # as actually separate entities vectors.
    u0, p0, theta0 = uptheta0.split(deepcopy=True)
    return u0, p0, theta0


def _solve_fixed_point():
    # Solve the coupled heat-Stokes equation approximately. Do this
    # iteratively by solving the heat equation, then solving Stokes with
    # the updated heat, the heat equation with the updated velocity and so
    # forth until the change is 'small'.
    WP = FunctionSpace(mesh, MixedElement([W_element, P_element]))
    # Initialize functions.
    up0 = Function(WP)
    u0, p0 = up0.split()

    theta1 = Function(Q)
    while True:
        heat_problem = cyl_heat.HeatCylindrical(
            Q, TrialFunction(Q), TestFunction(Q),
            b=u0,
            kappa=k_wpi_const,
            rho=rho_wpi_const,
            cp=cp_wpi_const,
            source=joule_wpi,
            dirichlet_bcs=theta_bcs_d,
            neumann_bcs=theta_bcs_n,
            my_dx=dx_submesh,
            my_ds=ds_submesh
            )
        _solve_stationary(heat_problem, theta1, verbose=False)

        f = rho_wpi(theta0) * g
        if extra_force:
            f += as_vector((extra_force[0], extra_force[1], 0.0))

        # Solve problem for velocity, pressure.
        # up1 = up0.copy()
        cyl_stokes.stokes_solve(
            up0,
            mu_wpi(theta_average),
            u_bcs, p_bcs,
            f=f,
            dx=dx_submesh,
            tol=1.0e-10,
            verbose=False,
            maxiter=1000
            )

        plot(u0)
        plot(theta0)

        theta_diff = errornorm(theta0, theta1)
        info('||theta - theta0|| = %e' % theta_diff)
        # info('||u - u0||         = %e' % u_diff)
        # info('||p - p0||         = %e' % p_diff)
        # diff = theta_diff + u_diff + p_diff
        diff = theta_diff
        info('sum = %e' % diff)

        # # Show the iterates.
        # plot(theta0, title='theta0')
        # plot(u0, title='u0')
        # interactive()
        # #exit()
        if diff < 1.0e-10:
            break

        theta0.assign(theta1)

    # Create a *deep* copy of u0, p0, to be able to deal with them as actually
    # separate entities.
    # u0, p0 = up0.split(deepcopy=True)
    return u0, p0, theta0
