# -*- coding: utf-8 -*-
#
import helpers
import maelstrom.navier_stokes_cartesian as ns_car

from dolfin import UnitSquareMesh, triangle, RectangleMesh, pi, Point
import numpy
import pytest
import sympy

# Turn down the log level to only error messages.
# set_log_level(WARNING)
# set_log_level(ERROR)
# set_log_level(0)


def problem_flat():
    '''Nothing interesting happening in the domain.
    '''
    def mesh_generator(n):
        return UnitSquareMesh(n, n, 'left/right')
    cell_type = triangle
    x = sympy.DeferredVector('x')
    u = (0.0 * x[0], 0.0 * x[1])
    p = -9.81 * x[1]
    solution = {'u': u,
                'p': p
                }
    f = _get_navier_stokes_rhs(u, p)
    mu = 1.0
    rho = 1.0
    return mesh_generator, solution, f, mu, rho, cell_type


def problem_whirl():
    '''Example from Teodora I. Mitkova's text
    "Finite-Elemente-Methoden fur die Stokes-Gleichungen".
    '''
    def mesh_generator(n):
        return UnitSquareMesh(n, n, 'left/right')
    cell_type = triangle
    x = sympy.DeferredVector('x')
    # t = sympy.symbols('t')

    # Note that the exact solution is indeed div-free.
    u = (
        x[0]**2 * (1 - x[0])**2 * 2 * x[1] * (1 - x[1]) * (2 * x[1] - 1),
        x[1]**2 * (1 - x[1])**2 * 2 * x[0] * (1 - x[0]) * (1 - 2 * x[0])
        )
    p = x[0] * (1 - x[0]) * x[1] * (1 - x[1])
    solution = {'u': u, 'p': p}
    f = _get_navier_stokes_rhs(u, p)
    mu = 1.0
    rho = 1.0
    return mesh_generator, solution, f, mu, rho, cell_type


def problem_guermond1():
    '''Problem 1 from section 3.7.1 in
        An overview of projection methods for incompressible flows;
        Guermond, Minev, Shen;
        Comp. Meth. in Appl. Mech. and Eng., vol. 195, 44-47, pp. 6011-6045;
        <http://www.sciencedirect.com/science/article/pii/S0045782505004640>.
    '''
    def mesh_generator(n):
        return RectangleMesh(Point(-1, -1), Point(1, 1), n, n, 'crossed')
    cell_type = triangle
    x = sympy.DeferredVector('x')
    t = sympy.symbols('t')
    # m = sympy.exp(t) - 0.0
    m = sympy.sin(t)
    u = (
        +pi*m*2*sympy.sin(pi*x[1])*sympy.cos(pi*x[1])*sympy.sin(pi*x[0])**2,
        -pi*m*2*sympy.sin(pi*x[0])*sympy.cos(pi*x[0])*sympy.sin(pi*x[1])**2
        )
    p = m * sympy.cos(pi * x[0]) * sympy.sin(pi * x[1])
    solution = {
        'u': {'value': u, 'degree': numpy.infty},
        'p': {'value': p, 'degree': numpy.infty}
        }
    f = {
        'value': _get_navier_stokes_rhs(u, p),
        'degree': numpy.infty
        }
    mu = 1.0
    rho = 1.0
    return mesh_generator, solution, f, mu, rho, cell_type


def problem_guermond2():
    '''Problem 2 from section 3.7.2 in
        An overview of projection methods for incompressible flows;
        Guermond, Minev, Shen;
        Comp. Meth. in Appl. Mech. and Eng., vol. 195, 44-47, pp. 6011-6045;
        <http://www.sciencedirect.com/science/article/pii/S0045782505004640>.
    '''
    def mesh_generator(n):
        return RectangleMesh(Point(0, 0), Point(1, 1), n, n, 'crossed')
    cell_type = triangle
    x = sympy.DeferredVector('x')
    t = sympy.symbols('t')
    u = (sympy.sin(x[0] + t) * sympy.sin(x[1] + t),
         sympy.cos(x[0] + t) * sympy.cos(x[1] + t)
         )
    p = sympy.sin(x[0] - x[1] + t)
    solution = {'u': u,
                'p': p
                }
    f = _get_navier_stokes_rhs(u, p)
    mu = 1.0
    rho = 1.0
    return mesh_generator, solution, f, mu, rho, cell_type


def problem_taylor():
    '''Taylor--Green vortex, cf.
    <http://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex>.
    '''
    def mesh_generator(n):
        return RectangleMesh(
            Point(0.0, 0.0), Point(2*pi, 2*pi),
            n, n, 'crossed'
            )
    mu = 1.0
    rho = 1.0
    cell_type = triangle
    x = sympy.DeferredVector('x')
    t = sympy.symbols('t')
    x0 = x[0]
    x1 = x[1]
    # F = sympy.exp(-2*mu*t)
    F = 1 - 2*mu*t
    u = (sympy.sin(x0) * sympy.cos(x1) * F,
         -sympy.cos(x0) * sympy.sin(x1) * F,
         0
         )
    p = rho/4 * (sympy.cos(2*x0) + sympy.cos(2*x1)) * F**2
    solution = {'u': u,
                'p': p
                }
    f = _get_navier_stokes_rhs(u, p)
    return mesh_generator, solution, f, mu, rho, cell_type


# TODO add test for spatial order
@pytest.mark.parametrize('problem', [
    # problem_flat,
    problem_whirl,
    # problem_guermond1,
    # problem_taylor,
    ])
@pytest.mark.parametrize('method', [
    ns_car.IPCS
    ])
def test_order(problem, method):
    '''Test order of time discretization.
    '''
    helpers._assert_time_order, problem, method
    return


def _get_navier_stokes_rhs(u, p):
    '''Given a solution u of the Cartesian Navier-Stokes equations, return
    a matching right-hand side f.
    '''
    x = sympy.DeferredVector('x')
    t, mu, rho = sympy.symbols('t, mu, rho')

    # Make sure that the exact solution is indeed analytically div-free.
    d = sympy.diff(u[0], x[0]) + sympy.diff(u[1], x[1])
    d = sympy.simplify(d)
    assert d == 0

    # Get right-hand side associated with this solution, i.e., according
    # the Navier-Stokes
    #
    #     rho (du_x/dt + u_x du_x/dx + u_y du_x/dy)
    #         = - dp/dx + mu [d^2u_x/dx^2 + d^2u_x/dy^2] + f_x,
    #     rho (du_y/dt + u_x du_y/dx + u_y du_y/dy)
    #         = - dp/dx + mu [d^2u_y/dx^2 + d^2u_y/dy^2] + f_y,
    #     du_x/dx + du_y/dy = 0.
    #
    #     rho (du/dt + (u.\nabla)u) = -\nabla p + mu [\div(\nabla u)] + f,
    #     div(u) = 0.
    #
    f0 = rho * (sympy.diff(u[0], t)
                + u[0] * sympy.diff(u[0], x[0])
                + u[1] * sympy.diff(u[0], x[1])
                ) \
        + sympy.diff(p, x[0]) \
        - mu * (sympy.diff(u[0], x[0], 2) + sympy.diff(u[0], x[1], 2))
    f1 = rho * (sympy.diff(u[1], t)
                + u[0] * sympy.diff(u[1], x[0])
                + u[1] * sympy.diff(u[1], x[1])
                ) \
        + sympy.diff(p, x[1]) \
        - mu * (sympy.diff(u[1], x[0], 2) + sympy.diff(u[1], x[1], 2))

    f = (
        sympy.simplify(f0),
        sympy.simplify(f1)
        )
    return f


if __name__ == '__main__':
    mesh_sizes = [8, 16, 32]
    Dt = [0.5 ** k for k in range(20)]
    errors = helpers.compute_time_errors(
        # problem_flat,
        # problem_whirl,
        problem_guermond1,
        # problem_guermond2,
        # problem_taylor,
        ns_car.IPCS,
        mesh_sizes, Dt
        )
    helpers.show_timeorder_info(Dt, mesh_sizes, errors)
