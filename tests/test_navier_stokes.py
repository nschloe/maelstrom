# -*- coding: utf-8 -*-
#
import maelstrom.navier_stokes as ns_cyl
import helpers

from dolfin import (
    Expression, UnitSquareMesh, triangle, plot, interactive, RectangleMesh, pi,
    Point
    )
import numpy
import pytest
import sympy

# Turn down the log level to only error messages.
# set_log_level(WARNING)
# set_log_level(ERROR)
# set_log_level(0)


def problem_flat_cylindrical():
    '''Nothing interesting happening in the domain.
    '''
    def mesh_generator(n):
        return UnitSquareMesh(n, n, 'left/right')
    cell_type = triangle
    # Coordinates ordered as (r, z, phi).
    x = sympy.DeferredVector('x')
    u = (0.0 * x[0], 0.0 * x[1], 0.0 * x[2])
    p = -9.81 * x[1]
    solution = {
        'u': {'value': u, 'degree': 1},
        'p': {'value': p, 'degree': 1}
        }
    f = {
        'value': _get_navier_stokes_rhs_cylindrical(u, p),
        'degree': numpy.infty
        }
    mu = 1.0
    rho = 1.0
    return mesh_generator, solution, f, mu, rho, cell_type


def problem_whirl_cylindrical():
    '''Example from Teodora I. Mitkova's text
    "Finite-Elemente-Methoden fur die Stokes-Gleichungen", adapted for
    cylindrical Navier-Stokes.
    '''
    alpha = 1.0

    def mesh_generator(n):
        # return UnitSquareMesh(n, n, 'left/right')
        return RectangleMesh(
                Point(alpha, 0.0), Point(1.0 + alpha, 1.0),
                n, n, 'left/right'
                )
    cell_type = triangle
    x = sympy.DeferredVector('x')
    # Note that the exact solution is indeed div-free.
    x0 = x[0] - alpha
    x1 = x[1]
    u = (
        x0**2 * (1 - x0)**2 * 2 * x1 * (1 - x1) * (2 * x1 - 1) / x[0],
        x1**2 * (1 - x1)**2 * 2 * x0 * (1 - x0) * (1 - 2 * x0) / x[0],
        0
        )
    p = x0 * (1 - x0) * x1 * (1 - x1)
    solution = {
        'u': {'value': u, 'degree': numpy.infty},
        'p': {'value': p, 'degree': 4}
        }
    plot_solution = False
    if plot_solution:
        sol_u = Expression(
            (sympy.printing.ccode(u[0]), sympy.printing.ccode(u[1])),
            degree=numpy.infty,
            t=0.0,
            cell=cell_type,
            )
        plot(sol_u, mesh=mesh_generator(20))
        interactive()
    f = {
        'value': _get_navier_stokes_rhs_cylindrical(u, p),
        'degree': numpy.infty
        }
    mu = 1.0
    rho = 1.0
    return mesh_generator, solution, f, mu, rho, cell_type


def problem_guermond1_cylindrical():
    '''Cylindrical variation of Guermond's test problem.
    '''
    alpha = 1.5

    def mesh_generator(n):
        return RectangleMesh(
            Point(-1+alpha, -1), Point(1+alpha, 1),
            n, n, 'crossed'
            )

    cell_type = triangle
    x = sympy.DeferredVector('x')
    t = sympy.symbols('t')
    x0 = x[0] - alpha
    x1 = x[1]
    # m = sympy.exp(t) - 0.0
    m = sympy.sin(t) + 1.0
    u = (
        + pi*m*2 * sympy.sin(pi*x1) * sympy.cos(pi*x1) * sympy.sin(pi*x0)**2
        / x[0],
        - pi*m*2 * sympy.sin(pi*x0) * sympy.cos(pi*x0) * sympy.sin(pi*x1)**2
        / x[0],
        0
        )
    p = m * sympy.cos(pi * x0) * sympy.sin(pi * x1)
    solution = {
        'u': {'value': u, 'degree': numpy.infty},
        'p': {'value': p, 'degree': numpy.infty}
        }
    f = {
        'value': _get_navier_stokes_rhs_cylindrical(u, p),
        'degree': numpy.infty
        }
    mu = 1.0
    rho = 1.0
    return mesh_generator, solution, f, mu, rho, cell_type


def problem_taylor_cylindrical():
    '''Taylor--Green vortex, cf.
    <http://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex>.
    '''
    alpha = 1.0

    def mesh_generator(n):
        return RectangleMesh(
            Point(0.0+alpha, 0.0), Point(2*pi+alpha, 2*pi),
            n, n, 'crossed'
            )
    mu = 1.0
    rho = 1.0
    cell_type = triangle
    x = sympy.DeferredVector('x')
    t = sympy.symbols('t')
    x0 = x[0] - alpha
    x1 = x[1]
    F = 1 - 2*mu*t
    u = (
        + sympy.sin(x0) * sympy.cos(x1) * F / x[0],
        - sympy.cos(x0) * sympy.sin(x1) * F / x[0],
        0
        )
    p = rho/4 * (sympy.cos(2*x0) + sympy.cos(2*x1)) * F**2
    solution = {
        'u': {'value': u, 'degree': numpy.infty},
        'p': {'value': p, 'degree': numpy.infty}
        }
    f = {
        'value': _get_navier_stokes_rhs_cylindrical(u, p),
        'degree': numpy.infty
        }
    return mesh_generator, solution, f, mu, rho, cell_type


@pytest.mark.parametrize('problem', [
    # problem_flat_cylindrical,
    problem_whirl_cylindrical,
    # problem_guermond1_cylindrical,
    # problem_taylor_cylindrical,
    ])
@pytest.mark.parametrize('method', [
    ns_cyl.IPCS
    ])
def test_order(problem, method):
    '''Test order of time discretization.
    '''
    # TODO add test for spatial order
    # Methods together with the expected order of convergence.
    helpers._assert_time_order(problem, method)
    return


def _get_navier_stokes_rhs_cylindrical(u, p):
    '''Given a solution u of the cylindrical Navier-Stokes equations, return
    a matching right-hand side f.
    '''
    x = sympy.DeferredVector('x')
    t, mu, rho = sympy.symbols('t, mu, rho')

    # Make sure that the exact solution is indeed analytically div-free.
    d = + 1.0 / x[0] * sympy.diff(rho * x[0] * u[0], x[0]) \
        + sympy.diff(rho * u[1], x[1])
    d = sympy.simplify(d)
    assert d == 0

    # Get right-hand side associated with this solution, i.e., according
    # the Navier-Stokes
    #
    #     rho (du_r/dt + u_r du_r/dr + u_y du_x/dy)
    #         = - dp/dr + mu [1/r d/dr(r u_r/dr) + 1/r^2 d^2u_r/dphi^2 +
    #                         d^2u_r/dz^2 - u_r/r^2 - 2/r^2 du_phi/dphi] + f_r,
    #     ...
    #     ...
    #     1/r d/dr(rho r u_r) + 1/r d/dphi(rho u_phi) + d/dz(rho u_z) = 0.
    #
    # IMPORTANT:
    # The order of the variables is assumed to be (r, z, phi).
    #
    f0 = rho * (
        + sympy.diff(u[0], t)
        + u[0] * sympy.diff(u[0], x[0])
        + u[1] * sympy.diff(u[0], x[1])
        + u[2]/x[0] * sympy.diff(u[0], x[2])
        - u[2] ** 2 / x[0]
        ) \
        + sympy.diff(p, x[0]) \
        - mu * (
            + 1/x[0] * sympy.diff(x[0] * sympy.diff(u[0], x[0]), x[0])
            + sympy.diff(u[0], x[1], 2)
            + 1/x[0]**2 * sympy.diff(u[0], x[2], 2)
            - u[0] / x[0] ** 2
            - 2 / x[0] ** 2 * sympy.diff(u[2], x[2])
            )

    f1 = rho * (
        + sympy.diff(u[1], t)
        + u[0] * sympy.diff(u[1], x[0])
        + u[1] * sympy.diff(u[1], x[1])
        + u[2]/x[0] * sympy.diff(u[1], x[2])
        ) \
        + sympy.diff(p, x[1]) \
        - mu * (
            + 1/x[0] * sympy.diff(x[0] * sympy.diff(u[1], x[0]), x[0])
            + sympy.diff(u[1], x[1], 2)
            + 1/x[0]**2 * sympy.diff(u[1], x[2], 2)
            )

    f2 = rho * (
            + sympy.diff(u[2], t)
            + u[0] * sympy.diff(u[2], x[0])
            + u[1] * sympy.diff(u[2], x[1])
            + u[2]/x[0] * sympy.diff(u[2], x[2])
            + u[0]*u[2]/x[0]
            ) \
        + 1/x[0] * sympy.diff(p, x[2]) \
        - mu * (
            + 1/x[0] * sympy.diff(x[0] * sympy.diff(u[2], x[0]), x[0])
            + sympy.diff(u[2], x[1], 2)
            + 1/x[0]**2 * sympy.diff(u[2], x[2], 2)
            - u[2]/x[0]**2
            + 2/x[0]**2 * sympy.diff(u[0], x[2])
            )
    f = (
        sympy.simplify(f0),
        sympy.simplify(f1),
        sympy.simplify(f2)
        )
    return f


if __name__ == '__main__':
    mesh_sizes = [8, 16, 32]
    # mesh_sizes = [10, 20, 40, 80]
    Dt = [0.5**k for k in range(20)]
    errors = helpers.compute_time_errors(
        # problem_flat_cylindrical,
        problem_whirl_cylindrical,
        # problem_guermond1_cylindrical,
        # problem_taylor_cylindrical,
        ns_cyl.IPCS,
        mesh_sizes, Dt
        )
    helpers.show_timeorder_info(Dt, mesh_sizes, errors)
