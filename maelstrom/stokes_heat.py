# -*- coding: utf-8 -*-
#
from dolfin import (
    Function, as_vector, FunctionSpace, MixedElement, info, errornorm
    )

from . import heat
from . import stokes


def solve_fixed_point(
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
        max_iter,
        tol
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
    for _ in range(max_iter):
        heat_problem = heat.Heat(
            Q,
            kappa=kappa,
            rho=rho(theta0),
            cp=cp,
            convection=u0,
            source=heat_source,
            dirichlet_bcs=theta_dirichlet_bcs,
            neumann_bcs=theta_neumann_bcs,
            my_dx=my_dx,
            my_ds=my_ds
            )

        theta1.assign(heat_problem.solve_stationary())

        # Solve problem for velocity, pressure.
        f = rho(theta0) * g  # coupling
        if extra_force:
            f += as_vector((extra_force[0], extra_force[1], 0.0))
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
    u0, p0 = up0.split(deepcopy=True)
    return u0, p0, theta0
