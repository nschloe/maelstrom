# -*- coding: utf-8 -*-
#
#  Copyright (c) 2012--2014, Nico Schlömer, <nico.schloemer@gmail.com>
#  All rights reserved.
#
#  This file is part of PyMHD.
#
#  PyMHD is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  PyMHD is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with PyMHD.  If not, see <http://www.gnu.org/licenses/>.
#
# Use nose unit testing for its test generator capabilities.
import nose

from dolfin import Expression, UnitSquareMesh, triangle, \
    plot, interactive, RectangleMesh, pi
import sympy as smp

import itertools

import pymhd.navier_stokes_cylindrical as ns_cyl

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


def _get_navier_stokes_rhs_cylindrical(u, p):
    '''Given a solution u of the cylindrical Navier-Stokes equations, return
    a matching right-hand side f.
    '''
    x = smp.DeferredVector('x')
    t, mu, rho = smp.symbols('t, mu, rho')

    # Make sure that the exact solution is indeed analytically div-free.
    d = 1.0 / x[0] * smp.diff(rho * x[0] * u[0], x[0]) \
        + smp.diff(rho * u[1], x[1])
    d = smp.simplify(d)
    nose.tools.assert_equal(d, 0)

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
    f0 = rho * (smp.diff(u[0], t)
                + u[0]      * smp.diff(u[0], x[0])
                + u[1]      * smp.diff(u[0], x[1])
                + u[2]/x[0] * smp.diff(u[0], x[2])
                - u[2] ** 2 / x[0]
                ) \
        + smp.diff(p, x[0]) \
        - mu * (1/x[0] * smp.diff(x[0] * smp.diff(u[0], x[0]), x[0])
                + smp.diff(u[0], x[1], 2)
                + 1/x[0]**2 * smp.diff(u[0], x[2], 2)
                - u[0] / x[0] ** 2
                - 2 / x[0] ** 2 * smp.diff(u[2], x[2])
                )

    f1 = rho * (smp.diff(u[1], t)
                + u[0]      * smp.diff(u[1], x[0])
                + u[1]      * smp.diff(u[1], x[1])
                + u[2]/x[0] * smp.diff(u[1], x[2])
                ) \
        + smp.diff(p, x[1]) \
        - mu * (1/x[0] * smp.diff(x[0] * smp.diff(u[1], x[0]), x[0])
                + smp.diff(u[1], x[1], 2)
                + 1/x[0]**2 * smp.diff(u[1], x[2], 2)
                )

    f2 = rho * (smp.diff(u[2], t)
                + u[0]      * smp.diff(u[2], x[0])
                + u[1]      * smp.diff(u[2], x[1])
                + u[2]/x[0] * smp.diff(u[2], x[2])
                + u[0]*u[2]/x[0]
                ) \
        + 1/x[0] * smp.diff(p, x[2]) \
        - mu * (1/x[0] * smp.diff(x[0] * smp.diff(u[2], x[0]), x[0])
                + smp.diff(u[2], x[1], 2)
                + 1/x[0]**2 * smp.diff(u[2], x[2], 2)
                - u[2]/x[0]**2
                + 2/x[0]**2 * smp.diff(u[0], x[2])
                )
    f = (smp.simplify(f0),
         smp.simplify(f1),
         smp.simplify(f2)
         )
    return f


def problem_flat_cylindrical():
    '''Nothing interesting happening in the domain.
    '''
    def mesh_generator(n):
        return UnitSquareMesh(n, n, 'left/right')
    cell_type = triangle
    # Coordinates ordered as (r, z, phi).
    x = smp.DeferredVector('x')
    u = (0.0 * x[0], 0.0 * x[1], 0.0 * x[2])
    p = -9.81 * x[1]
    solution = {'u': u,
                'p': p
                }
    f = _get_navier_stokes_rhs_cylindrical(u, p)
    mu = 1.0
    rho = 1.0
    return mesh_generator, solution, f, mu, rho, cell_type


def problem_whirl_cylindrical():
    '''Pistol Pete's example from Teodora I. Mitkova's text
    "Finite-Elemente-Methoden fur die Stokes-Gleichungen", adapted for
    cylindrical Navier-Stokes.
    '''
    alpha = 1.0

    def mesh_generator(n):
        #return UnitSquareMesh(n, n, 'left/right')
        return RectangleMesh(alpha, 0.0,
                             1.0 + alpha, 1.0,
                             n, n, 'left/right')
    cell_type = triangle
    x = smp.DeferredVector('x')
    # Note that the exact solution is indeed div-free.
    x0 = x[0] - alpha
    x1 = x[1]
    u = (x0 ** 2 * (1 - x0) ** 2 * 2 * x1 * (1 - x1) * (2 * x1 - 1) / x[0],
         x1 ** 2 * (1 - x1) ** 2 * 2 * x0 * (1 - x0) * (1 - 2 * x0) / x[0],
         0
         )
    p = x0 * (1 - x0) * x1 * (1 - x1)
    solution = {'u': u,
                'p': p
                }
    plot_solution = False
    if plot_solution:
        sol_u = Expression((smp.printing.ccode(u[0]),
                            smp.printing.ccode(u[1])),
                           t=0.0,
                           cell=cell_type,
                           )
        plot(sol_u,
             mesh=mesh_generator(20)
             )
        interactive()
    f = _get_navier_stokes_rhs_cylindrical(u, p)
    mu = 1.0
    rho = 1.0
    return mesh_generator, solution, f, mu, rho, cell_type


def problem_guermond1_cylindrical():
    '''Cylindrical variation of Guermond's test problem.
    '''
    alpha = 1.5

    def mesh_generator(n):
        return RectangleMesh(-1+alpha, -1, 1+alpha, 1, n, n, 'crossed')

    cell_type = triangle
    x = smp.DeferredVector('x')
    t = smp.symbols('t')
    x0 = x[0] - alpha
    x1 = x[1]
    #m = smp.exp(t) - 0.0
    m = smp.sin(t) + 1.0
    u = (pi * m * 2 * smp.sin(pi * x1) * smp.cos(pi * x1)
         * smp.sin(pi * x0) ** 2 / x[0],
         -pi * m * 2 * smp.sin(pi * x0) * smp.cos(pi * x0)
         * smp.sin(pi * x1) ** 2 / x[0],
         0
         )
    p = m * smp.cos(pi * x0) * smp.sin(pi * x1)
    solution = {'u': u,
                'p': p
                }
    f = _get_navier_stokes_rhs_cylindrical(u, p)
    mu = 1.0
    rho = 1.0
    return mesh_generator, solution, f, mu, rho, cell_type


def problem_taylor_cylindrical():
    '''Taylor--Green vortex, cf.
    <http://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex>.
    '''
    alpha = 1.0

    def mesh_generator(n):
        return RectangleMesh(0.0+alpha, 0.0, 2*pi+alpha, 2*pi, n, n, 'crossed')
    mu = 1.0
    rho = 1.0
    cell_type = triangle
    x = smp.DeferredVector('x')
    t = smp.symbols('t')
    x0 = x[0] - alpha
    x1 = x[1]
    F = 1 - 2*mu*t
    u = (smp.sin(x0) * smp.cos(x1) * F / x[0],
         -smp.cos(x0) * smp.sin(x1) * F / x[0],
         0
         )
    p = rho/4 * (smp.cos(2*x0) + smp.cos(2*x1)) * F**2
    solution = {'u': u,
                'p': p
                }
    f = _get_navier_stokes_rhs_cylindrical(u, p)
    print f
    exit()
    return mesh_generator, solution, f, mu, rho, cell_type


if __name__ == '__main__':
    mesh_sizes = [10, 20, 30]
    #mesh_sizes = [10, 20, 40, 80]
    Dt = [0.5 ** k for k in range(20)]
    errors = helpers.compute_time_errors(
        problem_flat_cylindrical,
        #problem_whirl_cylindrical,
        #problem_guermond1_cylindrical,
        #problem_taylor_cylindrical,
        ns_cyl.IPCS,
        mesh_sizes, Dt
        )
    helpers.show_timeorder_info(Dt, mesh_sizes, errors)
