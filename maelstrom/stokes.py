# -*- coding: utf-8 -*-
#
'''
Numerical solution schemes for the Stokes equation in cylindrical coordinates.
'''
from dolfin import DirichletBC, TrialFunctions, \
    TestFunctions, Expression, grad, pi, dx, assemble_system, \
    KrylovSolver, SubSpace, PETScPreconditioner, PETScOptions, \
    PETScKrylovSolver, inner, solve


def stokes_solve(
        up_out,
        mu,
        u_bcs, p_bcs,
        f,
        dx=dx,
        verbose=True,
        tol=1.0e-10,
        maxiter=1000
        ):
    # Some initial sanity checks.
    assert mu > 0.0

    WP = up_out.function_space()

    # Translate the boundary conditions into the product space.
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

    # TODO define p*=-1 and reverse sign in the end to get symmetric system?

    # Define variational problem
    (u, p) = TrialFunctions(WP)
    (v, q) = TestFunctions(WP)

    r = Expression('x[0]', degree=1, domain=WP.mesh())

    print("mu = %e" % mu)

    # build system
    a = mu * inner(r * grad(u), grad(v)) * 2 * pi * dx \
        - ((r * v[0]).dx(0) + (r * v[1]).dx(1)) * p * 2 * pi * dx \
        + ((r * u[0]).dx(0) + (r * u[1]).dx(1)) * q * 2 * pi * dx
    #   - div(r*v)*p* 2*pi*dx \
    #   + q*div(r*u)* 2*pi*dx
    L = inner(f, v) * 2 * pi * r * dx

    A, b = assemble_system(a, L, new_bcs)

    mode = 'lu'

    if mode == 'lu':
        solve(A, up_out.vector(), b, 'lu')

    elif mode == 'gmres':
        # For preconditioners for the Stokes system, see
        #
        #     Fast iterative solvers for discrete Stokes equations;
        #     J. Peters, V. Reichelt, A. Reusken.
        #
        prec = mu * inner(r * grad(u), grad(v)) * 2 * pi * dx \
            - p * q * 2 * pi * r * dx
        P, btmp = assemble_system(prec, L, new_bcs)
        solver = KrylovSolver('tfqmr', 'hypre_amg')
        # solver = KrylovSolver('gmres', 'hypre_amg')
        solver.set_operators(A, P)

        solver.parameters['monitor_convergence'] = verbose
        solver.parameters['report'] = verbose
        solver.parameters['absolute_tolerance'] = 0.0
        solver.parameters['relative_tolerance'] = tol
        solver.parameters['maximum_iterations'] = maxiter

        # Solve
        solver.solve(up_out.vector(), b)
    elif mode == 'fieldsplit':
        raise NotImplementedError('Fieldsplit solver not yet implemented.')
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

        # Create Krylov solver with custom preconditioner.
        solver = PETScKrylovSolver('gmres', prec)
        solver.set_operator(A)

    return
