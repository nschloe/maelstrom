# -*- coding: utf-8 -*-
#
#  Copyright (c) 2012--2014, Nico Schlömer, <nico.schloemer@gmail.com>
#  All rights reserved.
#
#  This file is part of Maelstrom.
#
#  Maelstrom is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  Maelstrom is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Maelstrom.  If not, see <http://www.gnu.org/licenses/>.
#
# Use nose unit testing for its test generator capabilities.
import nose

from dolfin import UnitSquareMesh, triangle, RectangleMesh, pi
import sympy as smp

import itertools

import maelstrom.navier_stokes_cartesian as ns_car

import helpers

# Turn down the log level to only error messages.
#set_log_level(WARNING)
#set_log_level(ERROR)
#set_log_level(0)


def test_generator():
    '''Test order of time discretization.
    '''
    # TODO add test for spatial order
    problems = [problem_whirl,
                ]
    # Methods together with the expected order of convergence.
    methods = [ns_car.chorin_step]
    # Loop over all methods and check the order of convergence.
    for method, problem in itertools.product(methods, problems):
        yield _check_time_order, problem, method


def _get_navier_stokes_rhs(u, p):
    '''Given a solution u of the Cartesian Navier-Stokes equations, return
    a matching right-hand side f.
    '''
    x = smp.DeferredVector('x')
    t, mu, rho = smp.symbols('t, mu, rho')

    # Make sure that the exact solution is indeed analytically div-free.
    d = smp.diff(u[0], x[0]) + smp.diff(u[1], x[1])
    d = smp.simplify(d)
    nose.tools.assert_equal(d, 0)

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
    f0 = rho * (smp.diff(u[0], t)
                + u[0] * smp.diff(u[0], x[0])
                + u[1] * smp.diff(u[0], x[1])
                ) \
        + smp.diff(p, x[0]) \
        - mu * (smp.diff(u[0], x[0], 2) + smp.diff(u[0], x[1], 2))
    f1 = rho * (smp.diff(u[1], t)
                + u[0] * smp.diff(u[1], x[0])
                + u[1] * smp.diff(u[1], x[1])
                ) \
        + smp.diff(p, x[1]) \
        - mu * (smp.diff(u[1], x[0], 2) + smp.diff(u[1], x[1], 2))

    f = (smp.simplify(f0),
         smp.simplify(f1)
         )
    return f


def problem_flat():
    '''Nothing interesting happening in the domain.
    '''
    def mesh_generator(n):
        return UnitSquareMesh(n, n, 'left/right')
    cell_type = triangle
    x = smp.DeferredVector('x')
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
    '''Pistol Pete's example from Teodora I. Mitkova's text
    "Finite-Elemente-Methoden fur die Stokes-Gleichungen".
    '''
    def mesh_generator(n):
        return UnitSquareMesh(n, n, 'left/right')
    cell_type = triangle
    x = smp.DeferredVector('x')
    t = smp.symbols('t')

    # Note that the exact solution is indeed div-free.
    u = (x[0] ** 2 * (1 - x[0]) ** 2 * 2 * x[1] * (1 - x[1]) * (2 * x[1] - 1),
         x[1] ** 2 * (1 - x[1]) ** 2 * 2 * x[0] * (1 - x[0]) * (1 - 2 * x[0])
         )
    p = x[0] * (1 - x[0]) * x[1] * (1 - x[1])
    solution = {'u': u,
                'p': p
                }
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
        return RectangleMesh(-1, -1, 1, 1, n, n, 'crossed')
    cell_type = triangle
    x = smp.DeferredVector('x')
    t = smp.symbols('t')
    #m = smp.exp(t) - 0.0
    m = smp.sin(t)
    u = (pi * m * 2 * smp.sin(pi * x[1]) * smp.cos(pi * x[1])
         * smp.sin(pi * x[0]) ** 2,
         -pi * m * 2 * smp.sin(pi * x[0]) * smp.cos(pi * x[0])
         * smp.sin(pi * x[1]) ** 2
         )
    p = m * smp.cos(pi * x[0]) * smp.sin(pi * x[1])
    solution = {'u': u,
                'p': p
                }
    f = _get_navier_stokes_rhs(u, p)
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
        return RectangleMesh(0, 0, 1, 1, n, n, 'crossed')
    cell_type = triangle
    x = smp.DeferredVector('x')
    t = smp.symbols('t')
    u = (smp.sin(x[0] + t) * smp.sin(x[1] + t),
         smp.cos(x[0] + t) * smp.cos(x[1] + t)
         )
    p = smp.sin(x[0] - x[1] + t)
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
        return RectangleMesh(0.0, 0.0, 2*pi, 2*pi, n, n, 'crossed')
    mu = 1.0
    rho = 1.0
    cell_type = triangle
    x = smp.DeferredVector('x')
    t = smp.symbols('t')
    x0 = x[0]
    x1 = x[1]
    #F = smp.exp(-2*mu*t)
    F = 1 - 2*mu*t
    u = (smp.sin(x0) * smp.cos(x1) * F,
         -smp.cos(x0) * smp.sin(x1) * F,
         0
         )
    p = rho/4 * (smp.cos(2*x0) + smp.cos(2*x1)) * F**2
    solution = {'u': u,
                'p': p
                }
    f = _get_navier_stokes_rhs(u, p)
    return mesh_generator, solution, f, mu, rho, cell_type


if __name__ == '__main__':
    mesh_sizes = [10, 20, 40]
    Dt = [0.5 ** k for k in range(20)]
    errors = helpers.compute_time_errors(
        #problem_flat,
        #problem_whirl,
        problem_guermond1,
        #problem_guermond2,
        #problem_taylor,
        ns_car.IPCS,
        mesh_sizes, Dt
        )
    helpers.show_timeorder_info(Dt, mesh_sizes, errors)
