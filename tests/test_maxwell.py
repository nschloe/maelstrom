# -*- coding: utf-8 -*-
#
import helpers
import maelstrom.maxwell_cylindrical as mcyl

from dolfin import (
    FunctionSpace, errornorm, RectangleMesh, Measure, CellFunction,
    FacetFunction, triangle, Expression, MPI, mpi_comm_world, Point
    )
import matplotlib.pyplot as plt
import numpy
import pytest
import sympy
import warnings

# Turn down the log level to only error messages.
# set_log_level(WARNING)
# set_log_level(ERROR)
# set_log_level(0)

MAX_DEGREE = 10


def problem_coscos():
    '''cosine example.
    '''
    def mesh_generator(n):
        mesh = RectangleMesh(
            Point(1.0, 0.0), Point(2.0, 1.0),
            n, n, 'left/right'
            )
        domains = CellFunction('uint', mesh)
        domains.set_all(0)
        dx = Measure('dx', subdomain_data=domains)
        boundaries = FacetFunction('uint', mesh)
        boundaries.set_all(0)
        ds = Measure('ds', subdomain_data=boundaries)
        return mesh, dx, ds

    x = sympy.DeferredVector('x')

    # Choose the solution, the parameters specifically, such that the boundary
    # conditions are fulfilled exactly
    alpha = 1.0
    r1 = 1.0
    beta = numpy.cos(alpha * r1) - r1 * alpha * numpy.sin(alpha * r1)

    phi = {'value': (beta * (1.0 - sympy.cos(alpha * x[0])),
                     beta * (1.0 - sympy.cos(alpha * x[0]))
                     ),
           'degree': numpy.infty
           }

    # Produce a matching right-hand side.
    mu = 1.0
    # sigma = 1.0
    # omega = 1.0
    # f_sympy = (
    #     - sympy.diff(1/(mu*x[0]) * sympy.diff(x[0]*phi[0], x[0]), x[0])
    #     - sympy.diff(1/(mu*x[0]) * sympy.diff(x[0]*phi[0], x[1]), x[1])
    #     - omega*sigma*phi[1],
    #     - sympy.diff(1/(mu*x[0]) * sympy.diff(x[0]*phi[1], x[0]), x[0])
    #     - sympy.diff(1/(mu*x[0]) * sympy.diff(x[0]*phi[1], x[1]), x[1])
    #     + omega*sigma*phi[0]
    #     )
    f_sympy = (
        -sympy.diff(
            1 / (mu * x[0]) * sympy.diff(x[0] * phi['value'][0], x[0]),
            x[0]
            ),
        # - sympy.diff(1/(mu*x[0]) * sympy.diff(x[0]*phi[0], x[1]), x[1])
        # - omega*sigma*phi[1],
        # - sympy.diff(1/(mu*x[0]) * sympy.diff(x[0]*phi[1], x[0]), x[0])
        # - sympy.diff(1/(mu*x[0]) * sympy.diff(x[0]*phi[1], x[1]), x[1])
        + 0.0 * phi['value'][0]
        )

    f = {
        'value': (
            Expression(sympy.printing.ccode(f_sympy[0]), degree=MAX_DEGREE),
            Expression(sympy.printing.ccode(f_sympy[1]), degree=MAX_DEGREE)
            ),
        'degree': MAX_DEGREE
        }

    # Show the solution and the right-hand side.
    n = 50
    mesh, dx, ds = mesh_generator(n)
    # plot(
    #     Expression(sympy.printing.ccode(phi[0])), mesh=mesh, title='phi.real'
    #     )
    # plot(
    #     Expression(sympy.printing.ccode(phi[1])), mesh=mesh, title='phi.imag'
    #     )
    # plot(f[0], mesh=mesh, title='f.real')
    # plot(f[1], mesh=mesh, title='f.imag')
    # interactive()
    return mesh_generator, phi, f, triangle


@pytest.mark.parametrize(
    'problem', [
        problem_coscos
        ])
def test_order(problem):
    '''Assert the correct discretization order.
    '''
    mesh_sizes = [16, 32, 64]
    errors, hmax = _compute_errors(problem, mesh_sizes)

    # Compute the numerical order of convergence.
    order = helpers._compute_numerical_order_of_convergence(hmax, errors)

    # The test is considered passed if the numerical order of convergence
    # matches the expected order in at least the first step in the coarsest
    # spatial discretization, and is not getting worse as the spatial
    # discretizations are refining.
    tol = 0.1
    expected_order = 1
    assert (order > expected_order - tol).all()
    return


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
            (
                sympy.printing.ccode(solution['value'][0]),
                sympy.printing.ccode(solution['value'][1])
            ),
            t=0.0,
            degree=degree,
            cell=cell_type
            )

    errors = numpy.empty(len(mesh_sizes))
    hmax = numpy.empty(len(mesh_sizes))
    for k, mesh_size in enumerate(mesh_sizes):
        mesh, dx, ds = mesh_generator(mesh_size)
        hmax[k] = MPI.max(mpi_comm_world(), mesh.hmax())
        V = FunctionSpace(mesh, 'CG', 1)
        # TODO don't hardcode Mu, Sigma, ...
        phi_approx = mcyl.solve_maxwell(
                V, dx,
                Mu={0: 1.0},
                Sigma={0: 1.0},
                omega=1.0,
                f_list=[{0: f['value']}],
                convections={},
                tol=1.0e-12,
                bcs=None,
                compute_residuals=False,
                verbose=False
                )
        # plot(sol0, mesh=mesh, title='sol')
        # plot(phi_approx[0][0], title='approx')
        # #plot(fenics_sol - theta_approx, title='diff')
        # interactive()
        # exit()
        #
        errors[k] = errornorm(sol, phi_approx[0])

    return errors, hmax


def _show_order_info(problem, mesh_sizes):
    '''Performs consistency check for the given problem/method combination and
    show some information about it. Useful for debugging.
    '''
    errors, hmax = _compute_errors(problem, mesh_sizes)
    order = helpers._compute_numerical_order_of_convergence(hmax, errors)

    # Print the data
    print
    print('hmax            ||u - u_h||     conv. order')
    print('%e    %e' % (hmax[0], errors[0]))
    for j in range(len(errors) - 1):
        print(32 * ' ' + '%2.5f' % order[j])
        print('%e    %e' % (hmax[j + 1], errors[j + 1]))

    # Plot the actual data.
    for i, mesh_size in enumerate(mesh_sizes):
        plt.loglog(hmax, errors, '-o', label=mesh_size)

    # Compare with order curves.
    plt.autoscale(False)
    e0 = errors[0]
    for order in range(2):
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
    mesh_sizes = [16, 32, 64, 128]
    _show_order_info(problem_coscos, mesh_sizes)
