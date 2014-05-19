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
'''
Numerical solution schemes for the Stokes equation in cylindrical coordinates.
'''

from dolfin import MixedFunctionSpace, DirichletBC, \
    TestFunctions, TrialFunctions, inner, grad, dx, dot, div, \
    assemble_system, KrylovSolver, PETScKrylovSolver, Function, \
    has_petsc, PETScOptions, PETScPreconditioner, SubSpace


# -----------------------------------------------------------------------------
def solve(W, P,
          mu,
          u_bcs, p_bcs,
          f,
          verbose=True,
          tol=1.0e-10
          ):
    # Some initial sanity checks.
    assert mu > 0.0

    WP = MixedFunctionSpace([W, P])

    # Translate the boundary conditions into the product space.
    # This conditional loop is able to deal with conditions of the kind
    #
    #     DirichletBC(W.sub(1), 0.0, right_boundary)
    #
    new_bcs = []
    for k, bcs in enumerate([u_bcs, p_bcs]):
        for bc in bcs:
            space = bc.function_space()
            C = space.component()
            if len(C) == 0:
                new_bcs.append(DirichletBC(WP.sub(k),
                                           bc.value(),
                                           bc.domain_args[0]))
            elif len(C) == 1:
                new_bcs.append(DirichletBC(WP.sub(k).sub(int(C[0])),
                                           bc.value(),
                                           bc.domain_args[0]))
            else:
                raise RuntimeError('Illegal number of subspace components.')

    # Define variational problem
    (u, p) = TrialFunctions(WP)
    (v, q) = TestFunctions(WP)

    # Build system.
    # The sign of the div(u)-term is somewhat arbitrary since the right-hand
    # side is 0 here. We can either make the system symmetric or positive-
    # definite.
    # On a second note, we have
    #
    #    \int grad(p).v = - \int p * div(v) + \int_\Gamma p n.v.
    #
    # Since, we have either p=0 or n.v=0 on the boundary, we could as well
    # replace the term dot(grad(p), v) by -p*div(v).
    #
    a = mu * inner(grad(u), grad(v))*dx \
      - p * div(v) * dx \
      - q * div(u) * dx
    #a = mu * inner(grad(u), grad(v))*dx + dot(grad(p), v) * dx \
    #  - div(u) * q * dx
    L = dot(f, v)*dx
    A, b = assemble_system(a, L, new_bcs)

    if has_petsc():
        # For an assortment of preconditioners, see
        #
        #     Performance and analysis of saddle point preconditioners
        #     for the discrete steady-state Navier-Stokes equations;
        #     H.C. Elman, D.J. Silvester, A.J. Wathen;
        #     Numer. Math. (2002) 90: 665-688;
        #     <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.145.3554>.
        #
        # Set up field split.
        W = SubSpace(WP, 0)
        P = SubSpace(WP, 1)
        u_dofs = W.dofmap().dofs()
        p_dofs = P.dofmap().dofs()
        prec = PETScPreconditioner()
        prec.set_fieldsplit([u_dofs, p_dofs], ['u', 'p'])

        PETScOptions.set('pc_type', 'fieldsplit')
        PETScOptions.set('pc_fieldsplit_type', 'additive')
        PETScOptions.set('fieldsplit_u_pc_type', 'lu')
        PETScOptions.set('fieldsplit_p_pc_type', 'jacobi')

        ## <http://scicomp.stackexchange.com/questions/7288/which-preconditioners-and-solver-in-petsc-for-indefinite-symmetric-systems-sho>
        #PETScOptions.set('pc_type', 'fieldsplit')
        ##PETScOptions.set('pc_fieldsplit_type', 'schur')
        ##PETScOptions.set('pc_fieldsplit_schur_fact_type', 'upper')
        #PETScOptions.set('pc_fieldsplit_detect_saddle_point')
        ##PETScOptions.set('fieldsplit_u_pc_type', 'lsc')
        ##PETScOptions.set('fieldsplit_u_ksp_type', 'preonly')

        #PETScOptions.set('pc_type', 'fieldsplit')
        #PETScOptions.set('fieldsplit_u_pc_type', 'hypre')
        #PETScOptions.set('fieldsplit_u_ksp_type', 'preonly')
        #PETScOptions.set('fieldsplit_p_pc_type', 'jacobi')
        #PETScOptions.set('fieldsplit_p_ksp_type', 'preonly')

        ## From PETSc/src/ksp/ksp/examples/tutorials/ex42-fsschur.opts:
        #PETScOptions.set('pc_type', 'fieldsplit')
        #PETScOptions.set('pc_fieldsplit_type', 'SCHUR')
        #PETScOptions.set('pc_fieldsplit_schur_fact_type', 'UPPER')
        #PETScOptions.set('fieldsplit_p_ksp_type', 'preonly')
        #PETScOptions.set('fieldsplit_u_pc_type', 'bjacobi')

        ## From
        ##
        ##     Composable Linear Solvers for Multiphysics;
        ##     J. Brown, M. Knepley, D.A. May, L.C. McInnes, B. Smith;
        ##     <http://www.computer.org/csdl/proceedings/ispdc/2012/4805/00/4805a055-abs.html>;
        ##     <http://www.mcs.anl.gov/uploads/cels/papers/P2017-0112.pdf>.
        ##
        #PETScOptions.set('pc_type', 'fieldsplit')
        #PETScOptions.set('pc_fieldsplit_type', 'schur')
        #PETScOptions.set('pc_fieldsplit_schur_factorization_type', 'upper')
        ##
        #PETScOptions.set('fieldsplit_u_ksp_type', 'cg')
        #PETScOptions.set('fieldsplit_u_ksp_rtol', 1.0e-6)
        #PETScOptions.set('fieldsplit_u_pc_type', 'bjacobi')
        #PETScOptions.set('fieldsplit_u_sub_pc_type', 'cholesky')
        ##
        #PETScOptions.set('fieldsplit_p_ksp_type', 'fgmres')
        #PETScOptions.set('fieldsplit_p_ksp_constant_null_space')
        #PETScOptions.set('fieldsplit_p_pc_type', 'lsc')
        ##
        #PETScOptions.set('fieldsplit_p_lsc_ksp_type', 'cg')
        #PETScOptions.set('fieldsplit_p_lsc_ksp_rtol', 1.0e-2)
        #PETScOptions.set('fieldsplit_p_lsc_ksp_constant_null_space')
        ##PETScOptions.set('fieldsplit_p_lsc_ksp_converged_reason')
        #PETScOptions.set('fieldsplit_p_lsc_pc_type', 'bjacobi')
        #PETScOptions.set('fieldsplit_p_lsc_sub_pc_type', 'icc')

        # Create Krylov solver with custom preconditioner.
        solver = PETScKrylovSolver('gmres', prec)
        solver.set_operator(A)
    else:
        # Use the preconditioner as recommended in
        # <http://fenicsproject.org/documentation/dolfin/dev/python/demo/pde/stokes-iterative/python/documentation.html>,
        #
        #     prec = inner(grad(u), grad(v))*dx - p*q*dx
        #
        # although it doesn't seem to be too efficient.
        # The sign on the last term doesn't matter.
        prec = mu * inner(grad(u), grad(v))*dx \
             - p*q*dx
        M, _ = assemble_system(prec, L, new_bcs)
        #solver = KrylovSolver('tfqmr', 'amg')
        solver = KrylovSolver('gmres', 'amg')
        solver.set_operators(A, M)

    solver.parameters['monitor_convergence'] = verbose
    solver.parameters['report'] = verbose
    solver.parameters['absolute_tolerance'] = 0.0
    solver.parameters['relative_tolerance'] = tol
    solver.parameters['maximum_iterations'] = 500

    # Solve
    up = Function(WP)
    solver.solve(up.vector(), b)

    # Get sub-functions
    u, p = up.split()

    return u, p
# -----------------------------------------------------------------------------
