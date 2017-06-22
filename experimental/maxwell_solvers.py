# -*- coding: utf-8 -*-
#
import matplotlib.pyplot as plt
import numpy
from numpy import pi

import maelstrom.maxwell as cmx


def _convert_to_complex(A):
    '''Convert from the format

         [ Re(A) -Im(A) ]
         [ Im(A)  Re(A) ]

    or the format

         [ Re(A)  Im(A) ]
         [ Im(A) -Re(A) ]

    into proper complex-valued format.
    '''
    m, n = A.shape
    assert m == n

    # Prepare index sets.
    I0 = numpy.array(range(0, n, 2))
    I1 = numpy.array(range(1, n, 2))

    # <http://stackoverflow.com/questions/7609108/slicing-sparse-scipy-matrix>
    ReA0 = A[I0[:, numpy.newaxis], I0]
    ReA1 = A[I1[:, numpy.newaxis], I1]

    # Make sure those are equal
    diffA = ReA0 - ReA1
    alpha = numpy.sqrt(numpy.vdot(diffA.data, diffA.data))
    diffB = ReA0 + ReA1
    beta = numpy.sqrt(numpy.vdot(diffB.data, diffB.data))
    if alpha < 1.0e-10:
        ReA = ReA0
    elif beta < 1.0e-10:
        ReA = ReA0
    else:
        raise ValueError('||ReA0 - ReA1||_fro = %e' % alpha)

    ImA0 = A[I0[:, numpy.newaxis], I1]
    ImA1 = A[I1[:, numpy.newaxis], I0]

    diffA = ImA0 + ImA1
    diffB = ImA0 - ImA1
    alpha = numpy.sqrt(numpy.vdot(diffA.data, diffA.data))
    beta = numpy.sqrt(numpy.vdot(diffB.data, diffB.data))
    if alpha < 1.0e-10:
        ImA = -ImA0
    elif beta < 1.0e-10:
        ImA = ImA0
    else:
        raise ValueError('||ImA0 - ImA1||_fro = %e' % alpha)
    # Now form the complex-valued matrix.
    return ReA + 1j * ImA


def _pyamg_test(V, dx, Mu, Sigma, omega, coils):
    import pyamg
    import krypy
    import scipy.sparse
    from maelstrom.solver_diagnostics import solver_diagnostics

    # Only calculate in one coil.
    v_ref = 1.0
    voltages_list = [{coils[0]['rings'][0]: v_ref}]

    # pylint: disable=unused-variable
    voltages_degree = 2
    A, P, b_list, M, W = cmx.build_system(
            V, dx,
            Mu, Sigma,  # dictionaries
            omega,
            voltages_list,  # dictionary
            f_degree=voltages_degree,
            convections={},
            bcs=[]
            )

    # Convert the matrix and rhs into scipy objects.
    rows, cols, values = A.data()
    A = scipy.sparse.csr_matrix((values, cols, rows))

    rows, cols, values = P.data()
    P = scipy.sparse.csr_matrix((values, cols, rows))

    # b = b_list[0].array()
    # b = b.reshape(M, 1)

    Ac = _convert_to_complex(A)
    Pc = _convert_to_complex(P)

    parameter_sweep = False
    if parameter_sweep:
        # Find good AMG parameters for P.
        solver_diagnostics(
                Pc,
                fname='my_maxwell_solver_diagnostic',
                # definiteness='positive',
                # symmetry='hermitian'
                )

    # Do a MINRES iteration for P^{-1}A.
    # Create solver
    ml = pyamg.smoothed_aggregation_solver(
        Pc,
        strength=('symmetric', {'theta': 0.0}),
        smooth=(
            'energy',
            {
                'weighting': 'local',
                'krylov': 'cg',
                'degree': 2,
                'maxiter': 3
            }
            ),
        Bimprove='default',
        aggregate='standard',
        presmoother=(
            'block_gauss_seidel',
            {'sweep': 'symmetric', 'iterations': 1}
            ),
        postsmoother=(
            'block_gauss_seidel',
            {'sweep': 'symmetric', 'iterations': 1}
            ),
        max_levels=25,
        max_coarse=300,
        coarse_solver='pinv'
        )

    def _apply_inverse_prec_exact(rhs):
        x_init = numpy.zeros((n, 1), dtype=complex)
        out = krypy.linsys.cg(Pc, rhs, x_init,
                              tol=1.0e-13,
                              M=ml.aspreconditioner(cycle='V')
                              )
        if out['info'] != 0:
            print('Preconditioner did not converge; last residual: %g'
                  % out['relresvec'][-1]
                  )
        # # Forget about the cycle used to gauge the residual norm.
        # self.tot_amg_cycles += [len(out['relresvec']) - 1]
        return out['xk']

    # Test preconditioning with approximations of P^{-1}, i.e., systems with
    # P are solved with k number of AMG cycles.
    Cycles = [1, 2, 5, 10]
    ch = plt.cm.get_cmap('cubehelix')
    # Construct right-hand side.
    m, n = Ac.shape
    b = numpy.random.rand(n) + 1j * numpy.random.rand(n)
    # pylint: disable=cell-var-from-loop
    for k, cycles in enumerate(Cycles):
        def _apply_inverse_prec_cycles(rhs):
            x_init = numpy.zeros((n, 1), dtype=complex)
            x = numpy.empty((n, 1), dtype=complex)
            residuals = []
            x[:, 0] = ml.solve(
                    rhs,
                    x0=x_init,
                    maxiter=cycles,
                    tol=0.0,
                    accel=None,
                    residuals=residuals
                    )
            # # Alternative for one cycle:
            # amg_prec = ml.aspreconditioner( cycle='V' )
            # x = amg_prec * rhs
            return x

        prec = scipy.sparse.linalg.LinearOperator(
                (n, n),
                _apply_inverse_prec_cycles,
                # _apply_inverse_prec_exact,
                dtype=complex
                )
        out = krypy.linsys.gmres(
                Ac, b,
                M=prec,
                maxiter=100,
                tol=1.0e-13,
                explicit_residual=True
                )
        print(cycles)
        # a lpha = float(cycles-1) / max(Cycles)
        alpha = float(k) / len(Cycles)
        plt.semilogy(
                out['relresvec'], '.-',
                label=cycles,
                color=ch(alpha)
                # color = '%e' % alpha
                )
    plt.legend(title='Number of AMG cycles for P^{~1}')
    plt.title('GMRES convergence history for P^{~1}A (%d x %d)' % Ac.shape)
    plt.show()
    return


def _show_currentloop_field():
    '''http://www.netdenizen.com/emagnettest/offaxis/?offaxisloop
    '''
    from numpy import sqrt

    r = numpy.linspace(0.0, 3.0, 51)
    z = numpy.linspace(-1.0, 1.0, 51)
    R, Z = numpy.meshgrid(r, z)

    a = 1.0
    V = 230 * sqrt(2.0)
    rho = 1.535356e-08
    II = V/rho
    mu0 = pi * 4e-7

    alpha = R / a
    beta = Z / a
    gamma = Z / R
    Q = (1+alpha)**2 + beta**2
    k = sqrt(4*alpha / Q)

    from scipy.special import ellipk
    from scipy.special import ellipe
    Kk = ellipk(k**2)
    Ek = ellipe(k**2)

    B0 = mu0*II / (2*a)

    V = B0 / (pi*sqrt(Q)) \
        * (Ek * (1.0 - alpha**2 - beta**2)/(Q - 4*alpha) + Kk)
    U = B0 * gamma / (pi*sqrt(Q)) \
        * (Ek * (1.0 + alpha**2 + beta**2)/(Q - 4*alpha) - Kk)

    Q = plt.quiver(R, Z, U, V)
    plt.quiverkey(
            Q, 0.7, 0.92, 1e4, '$1e4$',
            labelpos='W',
            fontproperties={'weight': 'bold'},
            color='r'
            )
    plt.show()
    return
