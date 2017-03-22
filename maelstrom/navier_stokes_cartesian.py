# -*- coding: utf-8 -*-
#
'''
Numerical solution schemes for the Navier--Stokes equation

        rho (u' + u.nabla(u)) = - nabla(p) + mu Delta(u) + f,
        div(u) = 0.

For an overview of methods, see

    An overview of projection methods for incompressible flows;
    Guermond, Minev, Shen;
    Comput. Methods Appl. Mech. Engrg., 195 (2006);
    <http://www.math.ust.hk/~mawang/teaching/math532/guermond-shen-2006.pdf>

or

    <http://mumerik.iwr.uni-heidelberg.de/Oberwolfach-Seminar/CFD-Course.pdf>.
'''

from dolfin import dot, inner, grad, dx, div, Function, TestFunction, solve, \
    Constant, DOLFIN_EPS, derivative, TrialFunction, FacetNormal, \
    assemble, ds, TestFunctions, split, PETScPreconditioner, \
    PETScKrylovSolver, as_backend_type, TrialFunctions, DirichletBC, \
    assemble_system, KrylovSolver, norm, info, plot, interactive, \
    PETScOptions

from warnings import warn
import numpy

# import Maelstrom.stabilization as stab
from Maelstrom.message import Message


def _rhs_weak(u, v, f, rho, mu):
    '''Right-hand side of the Navier--Stokes momentum equation in weak form.
    '''
    # Do no include the boundary term
    #
    #   - mu *inner(grad(u2)*n, v) * ds.
    #
    # This effectively means that at all boundaries where no sufficient
    # Dirichlet-conditions are posed, we assume grad(u)*n to vanish.
    #
    # It was first proposed in (with two intermediate steps)
    #
    #     Sur l'approximation de la solution des 'equations de Navier-Stokes
    #     par la m'ethode des pas fractionnaires (II);
    #     R. Temam;
    #     Arch. Ration. Mech. Anal. 33, (1969) 377-385;
    #     <http://link.springer.com/article/10.1007%2FBF00247696>.
    #
    # to replace the (weak form) convection <(u.\nabla)v, w> by something more
    # appropriate. Note, e.g., that
    #
    #       1/2 (  <(u.\nabla)v, w> - <(u.\nabla)w, v>)
    #     = 1/2 (2 <(u.\nabla)v, w> - <u, \nabla(v.w)>)
    #     = <(u.\nabla)v, w> - 1/2 \int u.\nabla(v.w)
    #     = <(u.\nabla)v, w> - 1/2 (-\int div(u)*(v.w)
    #                               +\int_\Gamma (n.u)*(v.w)
    #                              ).
    #
    # Since for solutions we have div(u)=0, n.u=0, we can consistently replace
    # the convection term <(u.\nabla)u, w> by the skew-symmetric
    #
    #     1/2 (<(u.\nabla)u, w> - <(u.\nabla)w, u>).
    #
    # One distinct advantage of this formulation is that here, the convective
    # term doesn't contribute to the total energy of the system since
    #
    # d/dt ||u||^2 = 2<d_t u, u>  = <(u.\nabla)u, u> - <(u.\nabla)u, u> = 0.
    #
    # More references and info on skew-symmetry can be found in
    #
    #     <http://www.wias-berlin.de/people/john/lectures_madrid_2012.pdf>,
    #     <http://calcul.math.cnrs.fr/Documents/Ecoles/CEMRACS2012/Julius_Reiss.pdf>.
    #
    # The first lecture is quite instructive and gives info on other
    # possibilties, e.g.,
    #
    #   * Rotational form
    #     <http://www.igpm.rwth-aachen.de/Download/reports/DROPS/IGPM193.pdf>
    #   * Divergence form
    #     This paper
    #     <http://www.cimec.org.ar/ojs/index.php/mc/article/viewFile/486/464>
    #     mentions 'divergence form', but it seems to be understood as another
    #     way of expressing the stress term mu\Delta(u).
    #
    # The different methods are numerically compared in
    #
    #     On the accuracy of the rotation form in simulations
    #     of the Navier-Stokes equations;
    #     Layton et al.;
    #
    #     <http://www.mathcs.emory.edu/~molshan/ftp/pub/RotationForm.pdf>
    #
    # In
    #
    #     Finite element methods
    #     for the incompressible Navier-Stokes equations;
    #     Ir. A. Segal;
    #     <http://ta.twi.tudelft.nl/users/vuik/burgers/fem_notes.pdf>;
    #
    # it is advised to use (u{k}.\nabla)u^{k+1} for the treatment of the
    # nonlinear term. In connection with the the div-stabilitation, this yields
    # unconditional stability of the scheme. On the other hand, and advantage
    # of treating the nonlinear term purely explicitly is that the resulting
    # problem would be symmetric and positive definite, qualifying for robust
    # AMG preconditioning.
    # One can also find advice on the boundary conditions for axisymmetric flow
    # here.
    #
    # For more information on stabilization techniques and general solution
    # recipes, check out
    #
    #     Finite Element Methods for Flow Problems;
    #     Jean Donea, Antonio Huerta.
    #
    # There are plenty of references in the book, e.g. to
    #
    #     Finite element stabilization parameters
    #     computed from element matrices and vectors;
    #     Tezduyar, Osawa;
    #     Comput. Methods Appl. Mech. Engrg. 190 (2000) 411-430;
    #     <http://www.tafsm.org/PUB_PRE/jALL/j89-CMAME-EBTau.pdf>
    #
    # where more details on SUPG are given.
    #
    return dot(f, v) * dx \
        - mu * inner(grad(u), grad(v)) * dx \
        - rho * 0.5 * (inner(grad(u)*u, v) - inner(grad(v)*u, u)) * dx
        #- rho*inner(grad(u)*u, v) * dx


def _rhs_strong(u, f, rho, mu):
    '''Right-hand side of the Navier--Stokes momentum equation in strong form.
    '''
    return f \
        - mu * div(grad(u)) \
        - rho * (grad(u)*u + 0.5*div(u)*u)


class PressureProjection(object):
    '''General pressure projection scheme as described in section 3.4 of

        An overview of projection methods for incompressible flows;
        Guermond, Miev, Shen;
        Comput. Methods Appl. Mech. Engrg. 195 (2006).
    '''
    def __init__(self,
                 W, P,
                 rho, mu,
                 theta,
                 stabilization=None
                 ):
        assert mu > 0.0
        # Only works for linear elements.
        if isinstance(rho, float):
            assert rho > 0.0
        else:
            assert rho.vector().min() > 0.0

        self.theta = theta
        self.W = W
        self.P = P
        self.rho = rho
        self.mu = mu
        self.stabilization = stabilization
        return

    def step(self,
             dt,
             u1, p1,
             u, p0,
             u_bcs, p_bcs,
             f0=None, f1=None,
             verbose=True,
             tol=1.0e-10
             ):
        # Some initial sanity checkups.
        assert dt > 0.0
        # Define trial and test functions
        ui = Function(self.W)
        v = TestFunction(self.W)
        # Create functions
        # Define coefficients
        k = Constant(dt)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Compute tentative velocity step:
        #
        #     F(u) = 0,
        #     F(u) := rho (U0 + (u.\nabla)u) - mu \div(\nabla u) - f = 0.
        #
        with Message('Computing tentative velocity'):
            # TODO higher-order scheme for time integration
            #
            # For higher-order schemes, see
            #
            #     A comparison of time-discretization/linearization approaches
            #     for the incompressible Navier-Stokes equations;
            #     Volker John, Gunar Matthies, Joachim Rang;
            #     Comput. Methods Appl. Mech. Engrg. 195 (2006) 5995-6010;
            #     <http://www.wias-berlin.de/people/john/ELECTRONIC_PAPERS/JMR06.CMAME.pdf>.
            #
            F1 = self.rho * inner((ui - u[0])/k, v) * dx

            if abs(self.theta) > DOLFIN_EPS:
                # Implicit terms.
                if f1 is None:
                    raise RuntimeError('Implicit schemes need right-hand side '
                                       'at target step (f1).')
                F1 -= self.theta * _rhs_weak(ui, v, f1, self.rho, self.mu)
            if abs(1.0 - self.theta) > DOLFIN_EPS:
                # Explicit terms.
                if f0 is None:
                    raise RuntimeError('Explicit schemes need right-hand side '
                                       'at current step (f0).')
                F1 -= (1.0 - self.theta) \
                    * _rhs_weak(u[0], v, f0, self.rho, self.mu)

            if p0:
                F1 += dot(grad(p0), v) * dx

            #if stabilization:
            #    tau = stab.supg2(V.mesh(),
            #                     u_1,
            #                     mu/rho,
            #                     V.ufl_element().degree()
            #                     )
            #    R = rho*(ui - u_1)/k
            #    if abs(theta) > DOLFIN_EPS:
            #        R -= theta * _rhs_strong(ui, f1, rho, mu)
            #    if abs(1.0-theta) > DOLFIN_EPS:
            #        R -= (1.0-theta) * _rhs_strong(u_1, f0, rho, mu)
            #    if p_1:
            #        R += grad(p_1)
            #    # TODO use u_1 or ui here?
            #    F1 += tau * dot(R, grad(v)*u_1) * dx

            # Get linearization and solve nonlinear system.
            # If the scheme is fully explicit (theta=0.0), then the system is
            # actually linear and only one Newton iteration is performed.
            J = derivative(F1, ui)

            # What is a good initial guess for the Newton solve?
            # Three choices come to mind:
            #
            #    (1) the previous solution u_1,
            #    (2) the intermediate solution from the previous step ui_1,
            #    (3) the solution of the semilinear system
            #        (u.\nabla(u) -> u_1.\nabla(u)).
            #
            # Numerical experiments with the Karman vortex street show that the
            # order of accuracy is (1), (3), (2). Typical norms would look like
            #
            #     ||u - u_1 || = 1.726432e-02
            #     ||u - ui_1|| = 2.720805e+00
            #     ||u - u_e || = 5.921522e-02
            #
            # Hence, use u_1 as initial guess.
            ui.assign(u[0])

            # Wrap the solution in a try-catch block to make sure we call end()
            # if necessary.
            #problem = NonlinearVariationalProblem(F1, ui, u_bcs, J)
            #solver  = NonlinearVariationalSolver(problem)
            solve(
                F1 == 0, ui,
                bcs=u_bcs,
                J=J,
                solver_parameters={
                    #'nonlinear_solver': 'snes',
                    'nonlinear_solver': 'newton',
                    'newton_solver': {
                        'maximum_iterations': 5,
                        'report': True,
                        'absolute_tolerance': tol,
                        'relative_tolerance': 0.0,
                        'linear_solver': 'iterative',
                        ## The nonlinear term makes the problem generally
                        ## nonsymmetric.
                        #'symmetric': False,
                        # If the nonsymmetry is too strong, e.g., if u_1 is
                        # large, then AMG preconditioning might not work very
                        # well.
                        'preconditioner': 'ilu',
                        #'preconditioner': 'hypre_amg',
                        'krylov_solver': {'relative_tolerance': tol,
                                          'absolute_tolerance': 0.0,
                                          'maximum_iterations': 1000,
                                          'monitor_convergence': verbose}
                        }})
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Message('Computing pressure correction'):
            #
            # The following is based on the update formula
            #
            #     rho/dt (u_{n+1}-u*) + \nabla phi = 0
            #
            # with
            #
            #     phi = (p_{n+1} - p*) + chi*mu*div(u*)
            #
            # and div(u_{n+1})=0. One derives
            #
            #   - \nabla^2 phi = rho/dt div(u_{n+1} - u*),
            #   - n.\nabla phi = rho/dt  n.(u_{n+1} - u*),
            #
            # In its weak form, this is
            #
            #     \int \grad(phi).\grad(q)
            #   = - rho/dt \int div(u*) q - rho/dt \int_Gamma n.(u_{n+1}-u*) q.
            #
            # If Dirichlet boundary conditions are applied to both u* and
            # u_{n+1} (the latter in the final step), the boundary integral
            # vanishes.
            #
            # Assume that on the boundary
            #   L2 -= inner(n, rho/k (u_bcs - ui)) * q * ds
            # is zero. This requires the boundary conditions to be set for
            # ui as well as u_final.
            # This creates some problems if the boundary conditions are
            # supposed to remain 'free' for the velocity, i.e., no Dirichlet
            # conditions in normal direction. In that case, one needs to
            # specify Dirichlet pressure conditions.
            #
            rotational_form = False
            self._pressure_poisson(p1, p0,
                                   self.mu, ui,
                                   divu=Constant(self.rho/dt)*div(ui),
                                   p_bcs=p_bcs,
                                   rotational_form=rotational_form,
                                   tol=tol,
                                   verbose=verbose
                                   )
            #plot(p - phi, title='p-phi')
            #plot(ui, title='u intermediate')
            #plot(f, title='f')
            ##plot(ui[1], title='u intermediate[1]')
            #plot(div(ui), title='div(u intermediate)')
            #plot(phi, title='phi')
            #interactive()
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Velocity correction.
        #   U = U0 - dt/rho \nabla p.
        with Message('Computing velocity correction'):
            u2 = TrialFunction(self.W)
            a3 = inner(u2, v) * dx
            if p0:
                phi = Function(self.P)
                phi.assign(p1)
                phi -= p0
            else:
                phi = p1
            if rotational_form:
                phi += self.mu * div(ui)
            L3 = inner(ui,  v) * dx \
                - k/self.rho * inner(grad(phi), v) * dx
            solve(a3 == L3, u1,
                  bcs=u_bcs,
                  solver_parameters={
                      'linear_solver': 'iterative',
                      'symmetric': True,
                      'preconditioner': 'jacobi',
                      'krylov_solver': {'relative_tolerance': tol,
                                        'absolute_tolerance': 0.0,
                                        'maximum_iterations': 100,
                                        'monitor_convergence': verbose}
                      })
        #u = project(ui - k/rho * grad(phi), V)
        #print '||u||_div = %e' % norm(u, 'Hdiv0')
        #uu = TrialFunction(Q)
        #vv = TestFunction(Q)
        #div_u = Function(Q)
        #solve(uu*vv*dx == div(u)*vv*dx, div_u,
        #      #bcs=DirichletBC(Q, 0.0, 'on_boundary')
        #      )
        #plot(div_u, title='div(u)')
        #interactive()
        return

    def _pressure_poisson(self,
                          p1, p0,
                          mu, ui,
                          divu,
                          p_bcs=None,
                          p_n=None,
                          rotational_form=False,
                          tol=1.0e-10,
                          verbose=True
                          ):
        '''Solve the pressure Poisson equation

            - \Delta phi = -div(u),
            boundary conditions,

        for

            \nabla p = u.
        '''
        P = p1.function_space()
        p = TrialFunction(P)
        q = TestFunction(P)

        a2 = dot(grad(p), grad(q)) * dx
        L2 = -divu * q * dx
        if p0:
            L2 += dot(grad(p0), grad(q)) * dx
        if p_n:
            n = FacetNormal(P.mesh())
            L2 += dot(n, p_n) * q * ds

        if rotational_form:
            L2 -= mu * dot(grad(div(ui)), grad(q)) * dx

        if p_bcs:
            solve(a2 == L2, p1,
                  bcs=p_bcs,
                  solver_parameters={
                      'linear_solver': 'iterative',
                      'symmetric': True,
                      'preconditioner': 'hypre_amg',
                      'krylov_solver': {'relative_tolerance': tol,
                                        'absolute_tolerance': 0.0,
                                        'maximum_iterations': 100,
                                        'monitor_convergence': verbose}
                  })
        else:
            # If we're dealing with a pure Neumann problem here (which is the
            # default case), this doesn't hurt CG if the system is consistent,
            # cf.
            #
            #    Iterative Krylov methods for large linear systems,
            #    Henk A. van der Vorst.
            #
            # And indeed, it is consistent: Note that
            #
            #    <1, rhs> = \sum_i 1 * \int div(u) v_i
            #             = 1 * \int div(u) \sum_i v_i
            #             = \int div(u).
            #
            # With the divergence theorem, we have
            #
            #    \int div(u) = \int_\Gamma n.u.
            #
            # The latter term is 0 iff inflow and outflow are exactly the same
            # at any given point in time. This corresponds with the
            # incompressibility of the liquid.
            #
            # In turn, this hints towards penetrable boundaries to require
            # Dirichlet conditions on the pressure.
            #
            A = assemble(a2)
            b = assemble(L2)
            #
            # In principle, the ILU preconditioner isn't advised here since it
            # might destroy the semidefiniteness needed for CG.
            #
            # The system is consistent, but the matrix has an eigenvalue 0.
            # This does not harm the convergence of CG, but when
            # preconditioning one has to take care that the preconditioner
            # preserves the kernel.  ILU might destroy this (and the
            # semidefiniteness). With AMG, the coarse grid solves cannot be LU
            # then, so try Jacobi here.
            # <http://lists.mcs.anl.gov/pipermail/petsc-users/2012-February/012139.html>
            #
            prec = PETScPreconditioner('hypre_amg')
            PETScOptions.set('pc_hypre_boomeramg_relax_type_coarse', 'jacobi')
            solver = PETScKrylovSolver('cg', prec)
            solver.parameters['absolute_tolerance'] = 0.0
            solver.parameters['relative_tolerance'] = tol
            solver.parameters['maximum_iterations'] = 100
            solver.parameters['monitor_convergence'] = verbose
            # Create solver and solve system
            A_petsc = as_backend_type(A)
            b_petsc = as_backend_type(b)
            p1_petsc = as_backend_type(p1.vector())
            solver.set_operator(A_petsc)
            try:
                solver.solve(p1_petsc, b_petsc)
            except RuntimeError as error:
                info('')
                # Check if the system is indeed consistent.
                #
                # If the right hand side is flawed (e.g., by round-off errors),
                # then it may have a component b1 in the direction of the null
                # space, orthogonal the image of the operator:
                #
                #     b = b0 + b1.
                #
                # When starting with initial guess x0=0, the minimal achievable
                # relative tolerance is then
                #
                #    min_rel_tol = ||b1|| / ||b||.
                #
                # If ||b|| is very small, which is the case when ui is almost
                # divergence-free, then min_rel_to may be larger than the
                # prescribed relative tolerance tol.
                #
                # Use this as a consistency check, i.e., bail out if
                #
                #     tol < min_rel_tol = ||b1|| / ||b||.
                #
                # For computing ||b1||, we use the fact that the null space is
                # one-dimensional, i.e.,  b1 = alpha e,  and
                #
                #     e.b = e.(b0 + b1) = e.b1 = alpha ||e||^2,
                #
                # so  alpha = e.b/||e||^2  and
                #
                #     ||b1|| = |alpha| ||e|| = e.b / ||e||
                #
                e = Function(P)
                e.interpolate(Constant(1.0))
                evec = e.vector()
                evec /= norm(evec)
                alpha = b.inner(evec)
                normB = norm(b)
                info('Linear system convergence failure.')
                info(error.message)
                message = ('Linear system not consistent! '
                           '<b,e> = %g, ||b|| = %g, <b,e>/||b|| = %e, tol = %e.') \
                           % (alpha, normB, alpha/normB, tol)
                info(message)
                if tol < abs(alpha) / normB:
                    info('\int div(u)  =  %e' % assemble(divu * dx))
                    #n = FacetNormal(Q.mesh())
                    #info('\int_Gamma n.u = %e' % assemble(dot(n, u)*ds))
                    #info('\int_Gamma u[0] = %e' % assemble(u[0]*ds))
                    #info('\int_Gamma u[1] = %e' % assemble(u[1]*ds))
                    ## Now plot the faulty u on a finer mesh (to resolve the
                    ## quadratic trial functions).
                    #fine_mesh = Q.mesh()
                    #for k in range(1):
                    #    fine_mesh = refine(fine_mesh)
                    #V1 = FunctionSpace(fine_mesh, 'CG', 1)
                    #W1 = V1*V1
                    #uplot = project(u, W1)
                    ##uplot = Function(W1)
                    ##uplot.interpolate(u)
                    #plot(uplot, title='u_tentative')
                    #plot(uplot[0], title='u_tentative[0]')
                    #plot(uplot[1], title='u_tentative[1]')
                    plot(divu, title='div(u_tentative)')
                    interactive()
                    exit()
                    raise RuntimeError(message)
                else:
                    exit()
                    raise RuntimeError('Linear system failed to converge.')
            except:
                exit()
        return


class IPCS(PressureProjection):
    '''
    Incremental pressure correction scheme.
    '''
    def __init__(self, W, P, rho, mu, theta,
                 stabilization=False
                 ):
        super(IPCS, self).__init__(W, P, rho, mu, theta,
                                   stabilization=stabilization
                                   )
        return


class AB2R():
    # AB2/TR method as described in 3.16.4 of
    #
    #     Incompressible flow and the finite element method;
    #     Volume 2: Isothermal laminar flow;
    #     P.M. Gresho, R.L. Sani.
    #
    # Here, the Navier-Stokes equation is written as
    #
    #     Mu' + (K+N(u)) u + Cp = f,
    #     C^T u = g.
    #
    # For incompressible Navier-Stokes,
    #
    #     rho (u' + u.nabla(u)) = - nabla(p) + mu Delta(u) + f,
    #     div(u) = 0,
    #
    # we have
    #
    #     M = rho,
    #     K = - mu \Delta,
    #     N(u) = rho * u.nabla(u),
    #     C = nabla,
    #     C^T = div,
    #     g = 0.
    #
    def __init__(self):
        return

    # Initial AB2/TR step.
    def ab2tr_step0(u0,
                    P,
                    f,  # right-hand side
                    rho,
                    mu,
                    dudt_bcs=[],
                    p_bcs=[],
                    eps=1.0e-4,  # relative error tolerance
                    verbose=True
                    ):
        # Make sure that the initial velocity is divergence-free.
        alpha = norm(u0, 'Hdiv0')
        if abs(alpha) > DOLFIN_EPS:
            warn('Initial velocity not divergence-free (||u||_div = %e).'
                 % alpha
                 )
        # Get the initial u0' and p0 by solving the linear equation system
        #
        #     [M   C] [u0']   [f0 - (K+N(u0)u0)]
        #     [C^T 0] [p0 ] = [ g0'            ],
        #
        # i.e.,
        #
        #     rho u0' + nabla(p0) = f0 + mu\Delta(u0) - rho u0.nabla(u0),
        #     div(u0')            = 0.
        #
        W = u0.function_space()
        WP = W*P

        # Translate the boundary conditions into product space. See
        # <http://fenicsproject.org/qa/703/boundary-conditions-in-product-space>.
        dudt_bcs_new = []
        for dudt_bc in dudt_bcs:
            dudt_bcs_new.append(DirichletBC(WP.sub(0),
                                            dudt_bc.value(),
                                            dudt_bc.user_sub_domain()))
        p_bcs_new = []
        for p_bc in p_bcs:
            p_bcs_new.append(DirichletBC(WP.sub(1),
                                         p_bc.value(),
                                         p_bc.user_sub_domain()))

        new_bcs = dudt_bcs_new + p_bcs_new

        (u, p) = TrialFunctions(WP)
        (v, q) = TestFunctions(WP)

        #a = rho * dot(u, v) * dx + dot(grad(p), v) * dx \
        a = rho * inner(u, v) * dx - p * div(v) * dx \
            - div(u) * q * dx
        L = _rhs_weak(u0, v, f, rho, mu)

        A, b = assemble_system(a, L, new_bcs)

        # Similar preconditioner as for the Stokes problem.
        # TODO implement something better!
        prec = rho * inner(u, v) * dx \
            - p*q*dx
        M, _ = assemble_system(prec, L, new_bcs)

        solver = KrylovSolver('gmres', 'amg')

        solver.parameters['monitor_convergence'] = verbose
        solver.parameters['report'] = verbose
        solver.parameters['absolute_tolerance'] = 0.0
        solver.parameters['relative_tolerance'] = 1.0e-6
        solver.parameters['maximum_iterations'] = 10000

        # Associate operator (A) and preconditioner matrix (M)
        solver.set_operators(A, M)
        #solver.set_operator(A)

        # Solve
        up = Function(WP)
        solver.solve(up.vector(), b)

        # Get sub-functions
        dudt0, p0 = up.split()

        # Choosing the first step size for the trapezoidal rule can be tricky.
        # Chapters 2.7.4a, 2.7.4e of the book
        #
        #     Incompressible flow and the finite element method,
        #     volume 1: advection-diffusion;
        #     P.M. Gresho, R.L. Sani,
        #
        # give some hints.
        #
        #     eps ... relative error tolerance
        #     tau ... estimate of the initial 'time constant'
        tau = None
        if tau:
            dt0 = tau * eps**(1.0/3.0)
        else:
            # Choose something 'reasonably small'.
            dt0 = 1.0e-3
        # Alternative:
        # Use a dissipative scheme like backward Euler or BDF2 for the first
        # couple of steps. This makes sure that noisy initial data is damped
        # out.
        return dudt0, p0, dt0

    def ab2tr_step(W, P, dt0, dt_1,
                   mu, rho,
                   u0, u_1, u_bcs,
                   dudt0, dudt_1, dudt_bcs,
                   p_1, p_bcs,
                   f0, f1,
                   tol=1.0e-12,
                   verbose=True
                   ):
        # General AB2/TR step.
        #
        # Steps are labeled in the following way:
        #
        #   * u_1: previous step.
        #   * u0:  current step.
        #   * u1:  next step.
        #
        # The same scheme applies to all other entities.
        #
        WP = W * P

        # Make sure the boundary conditions fit with the space.
        u_bcs_new = []
        for u_bc in u_bcs:
            u_bcs_new.append(DirichletBC(WP.sub(0),
                                         u_bc.value(),
                                         u_bc.user_sub_domain()))
        p_bcs_new = []
        for p_bc in p_bcs:
            p_bcs_new.append(DirichletBC(WP.sub(1),
                                         p_bc.value(),
                                         p_bc.user_sub_domain()))

        # Predict velocity.
        if dudt_1:
            u_pred = u0 \
                + 0.5*dt0*((2 + dt0/dt_1) * dudt0 - (dt0/dt_1) * dudt_1)
        else:
            # Simple linear extrapolation.
            u_pred = u0 + dt0 * dudt0

        uu = TrialFunctions(WP)
        vv = TestFunctions(WP)

        # Assign up[1] with u_pred and up[1] with p_1.
        # As of now (2013/09/05), there is no proper subfunction assignment in
        # Dolfin, cf.
        # <https://bitbucket.org/fenics-project/dolfin/issue/84/subfunction-assignment>.
        # Hence, we need to be creative here.
        # TODO proper subfunction assignment
        #
        # up1.assign(0, u_pred)
        # up1.assign(1, p_1)
        #
        up1 = Function(WP)
        a = (dot(uu[0],  vv[0]) + uu[1] * vv[1]) * dx
        L = dot(u_pred, vv[0]) * dx
        if p_1:
            L += p_1 * vv[1] * dx
        solve(a == L, up1,
              bcs=u_bcs_new + p_bcs_new
              )

        # Split up1 for easier access.
        # This is not as easy as it may seem at first, see
        # <http://fenicsproject.org/qa/1123/nonlinear-solves-with-mixed-function-spaces>.
        # Note in particular that
        #     u1, p1 = up1.split()
        # doesn't work here.
        #
        u1, p1 = split(up1)

        # Form the nonlinear equation system (3.16-235) in Gresho/Sani.
        # Left-hand side of the nonlinear equation system.
        F = 2.0/dt0 * rho * dot(u1, vv[0]) * dx \
            + mu * inner(grad(u1), grad(vv[0])) * dx \
            + rho * 0.5 * (inner(grad(u1)*u1, vv[0])
                           - inner(grad(vv[0]) * u1, u1)) * dx \
            + dot(grad(p1), vv[0]) * dx \
            + div(u1) * vv[1] * dx

        # Subtract the right-hand side.
        F -= dot(rho*(2.0/dt0*u0 + dudt0) + f1, vv[0]) * dx

        #J = derivative(F, up1)

        # Solve nonlinear system for u1, p1.
        solve(
            F == 0, up1,
            bcs=u_bcs_new + p_bcs_new,
            #J = J,
            solver_parameters={
              #'nonlinear_solver': 'snes',
              'nonlinear_solver': 'newton',
              'newton_solver': {'maximum_iterations': 5,
                                'report': True,
                                'absolute_tolerance': tol,
                                'relative_tolerance': 0.0
                                },
              'linear_solver': 'direct',
              #'linear_solver': 'iterative',
              ## The nonlinear term makes the problem
              ## generally nonsymmetric.
              #'symmetric': False,
              ## If the nonsymmetry is too strong, e.g., if
              ## u_1 is large, then AMG preconditioning
              ## might not work very well.
              #'preconditioner': 'ilu',
              ##'preconditioner': 'hypre_amg',
              #'krylov_solver': {'relative_tolerance': tol,
              #                  'absolute_tolerance': 0.0,
              #                  'maximum_iterations': 100,
              #                  'monitor_convergence': verbose}
              })

        ## Simpler access to the solution.
        #u1, p1 = up1.split()

        # Invert trapezoidal rule for next du/dt.
        dudt1 = 2 * (u1 - u0)/dt0 - dudt0

        # Get next dt.
        if dt_1:
            # Compute local trunction error (LTE) estimate.
            d = (u1 - u_pred) / (3*(1.0 + dt_1 / dt))
            # There are other ways of estimating the LTE norm.
            norm_d = numpy.sqrt(inner(d, d) / u_max**2)
            # Get next step size.
            dt1 = dt0 * (eps / norm_d)**(1.0/3.0)
        else:
            dt1 = dt0
        return u1, p1, dudt1, dt1
