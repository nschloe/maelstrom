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

from dolfin import FunctionSpace, errornorm, RectangleMesh, Measure, \
    CellFunction, FacetFunction, triangle, \
    Expression, MPI

import maelstrom.maxwell_cylindrical as mcyl
import sympy as smp
import numpy
from matplotlib import pyplot as pp

# Turn down the log level to only error messages.
#set_log_level(WARNING)
#set_log_level(ERROR)
#set_log_level(0)


def test_generator():
    '''Test order of time discretization.
    '''
    problems = [problem_coscos]
    # Loop over all methods and check the order of convergence.
    for problem in problems:
        yield _check_order, problem


def _check_order(problem):
    mesh_sizes = [20, 40, 80]
    errors, order = _compute_errors(problem, mesh_sizes)

    # The test is considered passed if the numerical order of convergence
    # matches the expected order in at least the first step in the coarsest
    # spatial discretization, and is not getting worse as the spatial
    # discretizations are refining.
    tol = 0.1
    k = 0
    expected_order = 1
    for i in range(order.shape[0]):
        nose.tools.assert_almost_equal(order[i], expected_order,
                                       delta=tol
                                       )
        while k + 1 < len(order[i]) \
                and abs(order[i][k + 1] - expected_order) < tol:
            k += 1
    return errors


def _compute_errors(problem, mesh_sizes):
    mesh_generator, solution, solution_degree, f, f_degree, cell_type = problem()

    sol0 = Expression(smp.printing.ccode(solution[0]),
                      t=0.0,
                      degree=solution_degree,
                      cell=cell_type
                      )
    sol1 = Expression(smp.printing.ccode(solution[1]),
                      t=0.0,
                      degree=solution_degree,
                      cell=cell_type
                      )

    errors = numpy.empty(len(mesh_sizes))
    hmax = numpy.empty(len(mesh_sizes))
    for k, mesh_size in enumerate(mesh_sizes):
        mesh, dx, ds = mesh_generator(mesh_size)
        hmax[k] = MPI.max(mesh.hmax())
        V = FunctionSpace(mesh, 'CG', 1)
        # TODO don't hardcode Mu, Sigma, ...
        phi_approx = mcyl.solve_maxwell(V, dx, ds,
                                        Mu={0: 1.0},
                                        Sigma={0: 1.0},
                                        omega=1.0,
                                        f_list=[{0: f}],
                                        convections={},
                                        tol=1.0e-12,
                                        bcs=None,
                                        compute_residuals=False,
                                        verbose=False
                                        )
        #plot(sol0, mesh=mesh, title='sol')
        #plot(phi_approx[0][0], title='approx')
        ##plot(fenics_sol - theta_approx, title='diff')
        #interactive()
        #exit()
        #
        e_r = errornorm(sol0, phi_approx[0][0])
        e_i = errornorm(sol1, phi_approx[0][1])
        errors[k] = numpy.sqrt(e_r ** 2 + e_i ** 2)

    # Compute the numerical order of convergence.
    order = numpy.empty(len(errors) - 1)
    for i in range(len(errors) - 1):
        order[i] = numpy.log(errors[i + 1] / errors[i]) \
            / numpy.log(hmax[i + 1] / hmax[i])

    return errors, order, hmax


def _show_order_info(problem, mesh_sizes):
    '''Performs consistency check for the given problem/method combination and
    show some information about it. Useful for debugging.
    '''
    errors, order, hmax = _compute_errors(problem, mesh_sizes)

    # Print the data
    print
    print('hmax = %e    error = %e' % (hmax[0], errors[0]))
    for j in range(len(errors) - 1):
        print('hmax = %e    error = %e    conv. order = %e'
              % (hmax[j + 1], errors[j + 1], order[j])
              )

    # Plot the actual data.
    for i, mesh_size in enumerate(mesh_sizes):
        pp.loglog(hmax, errors[i], '-o', label=mesh_size)

    # Compare with order curves.
    pp.autoscale(False)
    e0 = errors[-1][0]
    for o in range(7):
        pp.loglog([hmax[0], hmax[-1]],
                  [e0, e0 * (Dt[-1] / Dt[0]) ** o],
                  color='0.7')
    pp.xlabel('dt')
    pp.ylabel('||u-u_h||')
    pp.legend(loc=4)
    pp.show()
    return


def problem_coscos():
    '''cosine example.
    '''
    def mesh_generator(n):
        mesh = RectangleMesh(1.0, 0.0, 2.0, 1.0, n, n, 'left/right')
        domains = CellFunction('uint', mesh)
        domains.set_all(0)
        dx = Measure('dx')[domains]
        boundaries = FacetFunction('uint', mesh)
        boundaries.set_all(0)
        ds = Measure('ds')[boundaries]
        return mesh, dx, ds

    x = smp.DeferredVector('x')

    # Choose the solution, the parameters specifically, such that the boundary
    # conditions are fulfilled exactly
    alpha = 1.0
    r1 = 1.0
    beta = numpy.cos(alpha * r1) - r1 * alpha * numpy.sin(alpha * r1)

    phi = (beta * (1.0 - smp.cos(alpha * x[0])),
           beta * (1.0 - smp.cos(alpha * x[0]))
           )
    phi_degree = numpy.infty

    # Produce a matching right-hand side.
    mu = 1.0
    #sigma = 1.0
    #omega = 1.0
    #f_sympy = (- smp.diff(1/(mu*x[0]) * smp.diff(x[0]*phi[0], x[0]), x[0])
    #           - smp.diff(1/(mu*x[0]) * smp.diff(x[0]*phi[0], x[1]), x[1])
    #           - omega*sigma*phi[1],
    #           - smp.diff(1/(mu*x[0]) * smp.diff(x[0]*phi[1], x[0]), x[0])
    #           - smp.diff(1/(mu*x[0]) * smp.diff(x[0]*phi[1], x[1]), x[1])
    #           + omega*sigma*phi[0]
    #           )
    f_sympy = (-smp.diff(1 / (mu * x[0]) * smp.diff(x[0] * phi[0], x[0]), x[0])
               #- smp.diff(1/(mu*x[0]) * smp.diff(x[0]*phi[0], x[1]), x[1])
               #- omega*sigma*phi[1],
               ,
               #- smp.diff(1/(mu*x[0]) * smp.diff(x[0]*phi[1], x[0]), x[0])
               #- smp.diff(1/(mu*x[0]) * smp.diff(x[0]*phi[1], x[1]), x[1])
               + 0.0 * phi[0]
               )

    f = (Expression(smp.printing.ccode(f_sympy[0])),
         Expression(smp.printing.ccode(f_sympy[1]))
         )
    f_degree = numpy.infty

    # Show the solution and the right-hand side.
    n = 50
    mesh, dx, ds = mesh_generator(n)
    #plot(Expression(smp.printing.ccode(phi[0])), mesh=mesh, title='phi.real')
    #plot(Expression(smp.printing.ccode(phi[1])), mesh=mesh, title='phi.imag')
    #plot(f[0], mesh=mesh, title='f.real')
    #plot(f[1], mesh=mesh, title='f.imag')
    #interactive()
    return mesh_generator, phi, phi_degree, f, f_degree, triangle


if __name__ == '__main__':
    # For debugging purposes, show some info.
    #mesh_sizes = [20, 40, 80]
    mesh_sizes = [2]
    _show_order_info(problem_coscos, mesh_sizes)
