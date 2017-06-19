# -*- coding: utf-8 -*-
#
from dolfin import (
    NonlinearProblem, dx, ds, Function, split, TestFunctions, as_vector,
    assemble, derivative, FunctionSpace, MixedElement, assign, norm, info,
    errornorm
    )

from . import heat
from . import helpers
from . import stokes


def _average(u):
    '''Computes the average value of a function u over its domain.
    '''
    return assemble(u * dx) \
        / assemble(1.0 * dx(u.function_space().mesh()))


class StokesHeat(NonlinearProblem):

    def __init__(
            self, WPQ,
            kappa, rho, rho_const, mu, cp,
            g, extra_force,
            heat_source,
            u_bcs, p_bcs,
            theta_dirichlet_bcs=None,
            theta_neumann_bcs=None,
            theta_robin_bcs=None,
            my_dx=dx,
            my_ds=ds
            ):
        super(StokesHeat, self).__init__()

        theta_dirichlet_bcs = theta_dirichlet_bcs or {}
        theta_neumann_bcs = theta_neumann_bcs or {}
        theta_robin_bcs = theta_robin_bcs or {}

        # Translate the Dirichlet boundary conditions into the product space.
        self.dirichlet_bcs = helpers.dbcs_to_productspace(
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

        self.stokes_F = stokes.F(u, p, v, q, f, mu)

        self.heat_F = heat.F(
                theta, zeta,
                kappa=kappa, rho=rho_const, cp=cp,
                convection=u,
                source=heat_source,
                neumann_bcs=theta_neumann_bcs,
                robin_bcs=theta_robin_bcs,
                my_dx=my_dx,
                my_ds=my_ds
                )

        self.F0 = self.stokes_F + self.heat_F
        self.jacobian = derivative(self.F0, self.uptheta)
        return

    def F(self, b, x):
        self.uptheta.vector()[:] = x
        assemble(
            self.F0,
            tensor=b,
            form_compiler_parameters={'optimize': True}
            )
        for bc in self.dirichlet_bcs:
            bc.apply(b, x)
        return

    def J(self, A, x):
        self.uptheta.vector()[:] = x
        assemble(
            self.jacobian,
            tensor=A,
            form_compiler_parameters={'optimize': True}
            )
        for bc in self.dirichlet_bcs:
            bc.apply(A)
        return


def solve(
        mesh,
        W_element, P_element, Q_element,
        u0, p0, theta0,
        kappa, rho, mu, cp,
        g, extra_force,
        heat_source,
        u_bcs, p_bcs,
        theta_dirichlet_bcs,
        theta_neumann_bcs,
        dx_submesh, ds_submesh
        ):
    # First do a fixed_point iteration. This is usually quite robust and leads
    # to a point from where Newton can converge reliably.
    u0, p0, theta0 = _solve_fixed_point(
        mesh,
        W_element, P_element, Q_element,
        u0, p0, theta0,
        kappa, rho, mu, cp,
        g, extra_force,
        heat_source,
        u_bcs, p_bcs,
        theta_dirichlet_bcs,
        theta_neumann_bcs,
        dx_submesh, ds_submesh,
        tol=1.0e-2
        )

    WPQ = FunctionSpace(
        mesh, MixedElement([W_element, P_element, Q_element])
        )
    uptheta0 = Function(WPQ)

    # Initial guess
    assign(uptheta0.sub(0), u0)
    assign(uptheta0.sub(1), p0)
    assign(uptheta0.sub(2), theta0)

    rho_const = rho(_average(theta0))

    stokes_heat_problem = StokesHeat(
        WPQ,
        kappa, rho, rho_const, mu, cp,
        g, extra_force,
        heat_source,
        u_bcs, p_bcs,
        theta_dirichlet_bcs=theta_dirichlet_bcs,
        theta_neumann_bcs=theta_neumann_bcs,
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
    solver.parameters['maximum_iterations'] = 100
    # TODO tighten tolerance. might not always work though...
    solver.parameters['absolute_tolerance'] = 1.0e-3
    solver.parameters['relative_tolerance'] = 0.0
    solver.parameters['report'] = True

    solver.solve(stokes_heat_problem, uptheta0.vector())

    # u0, p0, theta0 = split(uptheta0)
    # Create a *deep* copy of u0, p0, theta0 to be able to deal with them
    # as actually separate entities vectors.
    u0, p0, theta0 = uptheta0.split(deepcopy=True)
    return u0, p0, theta0


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
            print('Fixed-point iteration {}:'.format(k) +
                  ' r (abs) = {:e} (tol = {:e})'.format(
                      nrm, self.parameters['absolute_tolerance']
                  ) +
                  ' r (rel) = {:e} (tol = {:e})'.format(
                      nrm/nrm0, self.parameters['relative_tolerance']
                  ))
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


def _solve_fixed_point(
        mesh,
        W_element, P_element, Q_element,
        u0, p0, theta0,
        kappa, rho, mu, cp,
        g, extra_force,
        heat_source,
        u_bcs, p_bcs,
        theta_dirichlet_bcs,
        theta_neumann_bcs,
        my_dx, my_ds,
        tol=1.0e-10
        ):
    # Solve the coupled heat-Stokes equation approximately. Do this
    # iteratively by solving the heat equation, then solving Stokes with the
    # updated heat, the heat equation with the updated velocity and so forth
    # until the change is 'small'.
    WP = FunctionSpace(mesh, MixedElement([W_element, P_element]))
    Q = FunctionSpace(mesh, Q_element)
    # Initialize functions.
    up0 = Function(WP)
    u0, p0 = up0.split()

    theta1 = Function(Q)
    while True:
        heat_problem = heat.Heat(
            Q,
            kappa=kappa, rho=rho(theta0), cp=cp,
            convection=u0,
            source=heat_source,
            dirichlet_bcs=theta_dirichlet_bcs,
            neumann_bcs=theta_neumann_bcs,
            my_dx=my_dx,
            my_ds=my_ds
            )

        theta1.assign(heat_problem.solve_stationary())

        f = rho(theta0) * g
        if extra_force:
            f += as_vector((extra_force[0], extra_force[1], 0.0))

        # Solve problem for velocity, pressure.
        # up1 = up0.copy()
        stokes.stokes_solve(
            up0,
            mu,
            u_bcs, p_bcs,
            f,
            my_dx=my_dx,
            tol=1.0e-10,
            verbose=False,
            maxiter=1000
            )

        # from dolfin import plot
        # plot(u0)
        # plot(theta0)

        theta_diff = errornorm(theta0, theta1)
        info('||theta - theta0|| = {:e}'.format(theta_diff))
        # info('||u - u0||         = {:e}'.format(u_diff))
        # info('||p - p0||         = {:e}'.format(p_diff))
        # diff = theta_diff + u_diff + p_diff
        diff = theta_diff
        info('sum = {:e}'.format(diff))

        # # Show the iterates.
        # plot(theta0, title='theta0')
        # plot(u0, title='u0')
        # interactive()
        # #exit()
        if diff < tol:
            break

        theta0.assign(theta1)

    # Create a *deep* copy of u0, p0, to be able to deal with them as
    # actually separate entities.
    # u0, p0 = up0.split(deepcopy=True)
    return u0, p0, theta0
