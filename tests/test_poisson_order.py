# -*- coding: utf-8 -*-
#
from __future__ import print_function

from dolfin import (
    FunctionSpace, errornorm, UnitSquareMesh, triangle, Expression,
    mpi_comm_world, pi, DirichletBC, MPI
    )
import matplotlib.pyplot as plt
import numpy
import pytest
import sympy
import warnings

from maelstrom import heat

import helpers

MAX_DEGREE = 5


def problem_sinsin():
    '''cosine example.
    '''
    def mesh_generator(n):
        return UnitSquareMesh(n, n, 'left/right')

    x = sympy.DeferredVector('x')

    # Choose the solution such that the boundary conditions are fulfilled
    # exactly.
    solution = {
        'value': sympy.sin(pi * x[0]) * sympy.sin(pi * x[1]),
        'degree': MAX_DEGREE
        }

    # Produce a matching right-hand side.
    phi = solution['value']
    rhs_sympy = sympy.simplify(
        - sympy.diff(x[0] * sympy.diff(phi, x[0]), x[0])
        - sympy.diff(x[0] * sympy.diff(phi, x[1]), x[1])
        )

    rhs = {
        'value': Expression(helpers.ccode(rhs_sympy), degree=MAX_DEGREE),
        'degree': MAX_DEGREE
        }

    return mesh_generator, solution, rhs, triangle


@pytest.mark.parametrize(
    'problem', [
        problem_sinsin
        ])
def test_residual(problem):
    mesh_size = 16
    mesh_generator, _, f, _ = problem()
    mesh, dx, _ = mesh_generator(mesh_size)

    Q = FunctionSpace(mesh, 'CG', 1)

    kappa = 1.0
    rho = 1.0
    cp = 1.0
    convection = None
    dirichlet = [DirichletBC()]
    rhs = f['value']

    prob = heat.Heat(
            Q,
            kappa, rho, cp,
            convection,
            rhs,
            dirichlet_bcs=dirichlet
            )
    sol = prob.solve_stationary()

    assert False
    return


# @pytest.mark.parametrize(
#     'problem', [
#         problem_coscos
#         ])
# def test_order(problem):
#     '''Assert the correct discretization order.
#     '''
#     mesh_sizes = [16, 32, 64]
#     errors, hmax = _compute_errors(problem, mesh_sizes)
#
#     # Compute the numerical order of convergence.
#     order = helpers.compute_numerical_order_of_convergence(hmax, errors)
#
#     # The test is considered passed if the numerical order of convergence
#     # matches the expected order in at least the first step in the coarsest
#     # spatial discretization, and is not getting worse as the spatial
#     # discretizations are refining.
#     tol = 0.1
#     expected_order = 1
#     assert (order > expected_order - tol).all()
#     return


def _compute_errors(problem, mesh_sizes):
    mesh_generator, solution, f, cell_type = problem()

    if solution['degree'] > MAX_DEGREE:
        warnings.warn(
            'Expression degree (%r) > maximum degree (%d). Truncating.'
            % (solution['degree'], MAX_DEGREE)
            )
        degree = MAX_DEGREE
    else:
        degree = solution['degree']

    sol = Expression(
            helpers.ccode(solution['value']),
            t=0.0,
            degree=degree,
            cell=cell_type
            )

    errors = numpy.empty(len(mesh_sizes))
    hmax = numpy.empty(len(mesh_sizes))
    for k, mesh_size in enumerate(mesh_sizes):
        mesh = mesh_generator(mesh_size)
        hmax[k] = MPI.max(mpi_comm_world(), mesh.hmax())
        Q = FunctionSpace(mesh, 'CG', 1)
        # TODO don't hardcode parameters
        prob = heat.Heat(
                Q,
                kappa=1.0, rho=1.0, cp=1.0,
                convection=None,
                source=f['value'],
                dirichlet_bcs=[DirichletBC(Q, 0.0, 'on_boundary')]
                )
        phi_approx = prob.solve_stationary()
        errors[k] = errornorm(sol, phi_approx)

    return errors, hmax


def _show_order_info(problem, mesh_sizes):
    '''Performs consistency check for the given problem/method combination and
    show some information about it. Useful for debugging.
    '''
    errors, hmax = _compute_errors(problem, mesh_sizes)
    order = helpers.compute_numerical_order_of_convergence(hmax, errors)

    # Print the data
    print()
    print('hmax            ||u - u_h||     conv. order')
    print('%e    %e' % (hmax[0], errors[0]))
    for j in range(len(errors) - 1):
        print(32 * ' ' + '%2.5f' % order[j])
        print('%e    %e' % (hmax[j + 1], errors[j + 1]))

    # Plot the actual data.
    for mesh_size in mesh_sizes:
        plt.loglog(hmax, errors, '-o', label=mesh_size)

    # Compare with order curves.
    plt.autoscale(False)
    e0 = errors[0]
    for order in range(4):
        plt.loglog(
            [hmax[0], hmax[-1]],
            [e0, e0 * (hmax[-1] / hmax[0]) ** order],
            color='0.7'
            )
    plt.xlabel('hmax')
    plt.ylabel('||u-u_h||')
    plt.legend()
    plt.show()
    return


if __name__ == '__main__':
    mesh_sizes_ = [16, 32, 64, 128]
    _show_order_info(problem_sinsin, mesh_sizes_)
