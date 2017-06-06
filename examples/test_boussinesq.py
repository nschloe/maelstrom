#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
'''
Coupled solve of the Navier--Stokes and the heat equation
in cylindrical coordinates.
'''
from dolfin import (
    parameters, Constant, Function, XDMFFile, mpi_comm_world,
    SpatialCoordinate, DOLFIN_EPS, begin, end, dot, grad, pi, project, plot,
    norm, assemble, dx, TestFunction, TrialFunction, LUSolver, ds, as_vector,
    lhs, rhs
    )
import parabolic

import maelstrom.navier_stokes as cyl_ns

import problems


# We need to allow extrapolation here since otherwise, the equation systems
# for Maxwell cannot be constructed: They contain the velocity `u` (from
# Navier-Stokes) which is only defined on the workpiece subdomain.
# Cf. <https://answers.launchpad.net/dolfin/+question/210508>.
parameters['allow_extrapolation'] = True

GMSH_EPS = 1.0e-15


class Heat(object):
    def __init__(self, V, conv, kappa, rho, cp, dirichlet_bcs, neumann_bcs):
        # TODO stabilization
        # About stabilization for reaction-diffusion-convection:
        # http://www.ewi.tudelft.nl/fileadmin/Faculteit/EWI/Over_de_faculteit/Afdelingen/Applied_Mathematics/Rapporten/doc/06-03.pdf
        # http://www.xfem.rwth-aachen.de/Project/PaperDownload/Fries_ReviewStab.pdf
        #
        # R = u_t \
        #     + dot(u0, grad(trial)) \
        #     - 1.0/(rho(293.0)*cp) * div(kappa*grad(trial))
        # F -= R * dot(tau*u0, grad(v)) * dx
        #
        # Stabilization
        # tau = stab.supg2(
        #         mesh,
        #         u0,
        #         kappa/(rho(293.0)*cp),
        #         Q.ufl_element().degree()
        #         )
        self.V = V
        self.conv = conv
        self.kappa = kappa
        self.rho = rho
        self.cp = cp
        self.dirichlet_bcs = dirichlet_bcs
        self.neumann_bcs = neumann_bcs
        return

    def _get_system(self, alpha, beta, u, v):
        # If there are sharp temperature gradients, numerical oscillations may
        # occur. This happens because the resulting matrix is not an M-matrix,
        # caused by the fact that A1 puts positive elements in places other
        # than the main diagonal. To prevent that, it is suggested by
        # Großmann/Roos to use a vertex-centered discretization for the mass
        # matrix part.
        # Check
        # https://bitbucket.org/fenics-project/ffc/issues/145/uflacs-error-for-vertex-quadrature-scheme
        f1 = assemble(
              u * v * dx,
              form_compiler_parameters={
                  'quadrature_rule': 'vertex',
                  'representation': 'quadrature'
                  }
              )

        r = SpatialCoordinate(self.V.mesh())[0]

        rho_cp = self.rho * self.cp
        f2 = (
            - self.kappa * r * dot(grad(u), grad(v/rho_cp)) * 2*pi*dx
            - dot(self.conv, grad(u)) * v * 2*pi*r*dx
            )

        # Neumann boundary conditions
        for k, n_dot_grad_T in self.neumann_bcs.items():
            f2 -= r * self.kappa * n_dot_grad_T * v / rho_cp * 2*pi*ds(k)

        # # Add SUPG stabilization.
        # rho_cp = rho[wpi](background_temp)*cp[wpi]
        # k = kappa[wpi](background_temp)
        # Rdx = u_t * 2*pi*r*dx(wpi) \
        #     + dot(u_1, grad(trial)) * 2*pi*r*dx(wpi) \
        #     - 1.0/(rho_cp) * div(k*r*grad(trial)) * 2*pi*dx(wpi)
        # #F -= dot(tau*u_1, grad(v)) * Rdx
        # #F -= tau * inner(u_1, grad(v)) * 2*pi*r*dx(wpi)
        # #plot(tau, mesh=V.mesh(), title='u_tau')
        # #interactive()
        # #F -= tau * v * 2*pi*r*dx(wpi)
        # #F -= tau * Rdx

        f2l = assemble(lhs(f2))
        f2r = assemble(rhs(f2))
        return alpha * f1 + beta * f2l, f2r

    def eval_alpha_M_beta_F(self, alpha, beta, u, t):
        # Evaluate  alpha * M * u + beta * F(u, t).
        v = TestFunction(self.V)
        Au, b = self._get_system(alpha, beta, u, v)
        return Au + b

    def solve_alpha_M_beta_F(self, alpha, beta, b, t):
        # Solve  alpha * M * u + beta * F(u, t) = b  for u.
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        A, b = self._get_system(alpha, beta, u, v)

        for dbc in self.dirichlet_bcs:
            dbc.apply(A, b)

        # solver = KrylovSolver('gmres', 'ilu')
        # solver.parameters['relative_tolerance'] = 1.0e-13
        # solver.parameters['absolute_tolerance'] = 0.0
        # solver.parameters['maximum_iterations'] = 1000
        # solver.parameters['monitor_convergence'] = True

        # The Krylov solver doesn't converge
        solver = LUSolver()
        solver.set_operator(A)

        u = Function(self.V)
        solver.solve(u.vector(), b)
        return u


def test():
    # mesh, V, Q, u_bcs, p_bcs, heat_boundary, left_boundary, right_boundary, \
    #     lower_boundary, upper_boundary = _domain_ballintube()
    # mesh, subdomains, subdomain_materials, wpi, V, Q, P, u_bcs, p_bcs, \
    #     heater_boundary = _domain_peter()
    # mesh, subdomains, subdomain_materials, wpi, V, Q, P, u_bcs, p_bcs, \
    #     background_temp, heater_bcs = probs.crucible()

    problem = problems.Crucible()

    # dx = Measure('dx', subdomain_data=problem.subdomains)

    material = problem.subdomain_materials[problem.wpi]
    rho = material.density
    cp = material.specific_heat_capacity
    kappa = material.thermal_conductivity
    mu = material.dynamic_viscosity

    # Start time, end time, time step.
    t = 0.0
    T = 1200.0
    dt = 1.0e-4

    g = Constant((0.0, -9.81))

    # Initial states.
    u_1 = Function(problem.W, name='velocity')
    u_1.interpolate(Constant((0.0, 0.0, 0.0)))

    p_1 = Function(problem.P, name='pressure')
    p_1.interpolate(Constant(0.0))

    theta_1 = Function(problem.Q, name='temperature')
    theta_1.interpolate(Constant(problem.background_temp))

    with XDMFFile(mpi_comm_world(), 'boussinesq.xdmf') as xdmf_file:
        xdmf_file.parameters['flush_output'] = True
        xdmf_file.parameters['rewrite_function_mesh'] = False

        xdmf_file.write(u_1, t)
        xdmf_file.write(p_1, t)
        xdmf_file.write(theta_1, t)

        while t < T + DOLFIN_EPS:
            begin('Time step %e -> %e...' % (t, t+dt))

            begin('Computing heat...')
            # Stabilization in the workpiece.
            # rho_cp = rho[wpi](1550.0) * cp[wpi](1550.0)
            # tau = stab.supg2(
            #     Q.mesh(),  # TODO what to put here?
            #     u_1,
            #     k/rho_cp,
            #     Q.ufl_element().degree()
            #     )
            # u_tau = stab.supg(u_1,
            #                   k/rho_cp,
            #                   Q.ufl_element().degree()
            #                   )
            # Build right-hand side F of heat equation such that u' = F(u).

            # def weak_F(t, u_t, trial, v):
            #     # TODO reevaluate
            #     # Don't use zero() or 0 to avoid errors as described in
            #     # <https://bitbucket.org/fenics-project/dolfin/issue/44/assemble-0-vectors>.
            #     # Use Expression instead of Constant to work around the error
            #     # <https://bitbucket.org/fenics-project/dolfin/issue/38/constant-expressions-dont-use-the-cell>.
            #     # Also, explicitly both RHS and LHS to something that doesn't
            #     # evaluate to an empty form.
            #     F = trial \
            #         * Expression('0.0', cell=triangle) * v * 2*pi*r*dx(0) \
            #         + Expression('0.0', cell=triangle) * v * 2*pi*r*dx(0)
            #     for i in subdomain_indices:
            #         # Take all parameters at 1550K.
            #         rho_cp = rho[i](1550.0) * cp[i](1550.0)
            #         k = kappa[i](1550.0)
            #         F -= k * r * dot(grad(trial), grad(v/rho_cp)) * 2*pi*dx(i)
            #     # Add convection.
            #     F -= dot(u_1, grad(trial)) * v * 2*pi*r*dx(problem.wpi)
            #     # # Add SUPG stabilization.
            #     # rho_cp = rho[wpi](background_temp)*cp[wpi]
            #     # k = kappa[wpi](background_temp)
            #     # Rdx = u_t * 2*pi*r*dx(wpi) \
            #     #     + dot(u_1, grad(trial)) * 2*pi*r*dx(wpi) \
            #     #     - 1.0/(rho_cp) * div(k*r*grad(trial)) * 2*pi*dx(wpi)
            #     # #F -= dot(tau*u_1, grad(v)) * Rdx
            #     # #F -= tau * inner(u_1, grad(v)) * 2*pi*r*dx(wpi)
            #     # #plot(tau, mesh=V.mesh(), title='u_tau')
            #     # #interactive()
            #     # #F -= tau * v * 2*pi*r*dx(wpi)
            #     # #F -= tau * Rdx
            #     return F
            # theta = ts.implicit_euler_step(
            #     problem.Q,
            #     weak_F,
            #     theta_1,
            #     t, dt,
            #     sympy_bcs=problem.heater_bcs,
            #     tol=1.0e-12,
            #     lhs_multiplier=2*pi*r,
            #     verbose=False,
            #     form_compiler_parameters={
            #         'quadrature_rule': 'vertex',
            #         'quadrature_degree': 1,
            #         },
            #     )

            stepper = parabolic.ImplicitEuler(
                    Heat(
                        problem.Q,
                        # Only take the first two components of the convection
                        as_vector([u_1[0], u_1[1]]),
                        # Take all parameters at 1550K.
                        kappa, rho(1550.0), cp,
                        problem.theta_bcs_d,
                        # problem.theta_bcs_d_strict,
                        problem.theta_bcs_n
                        )
                    )
            theta = stepper.step(theta_1, t, dt)
            end()

            # Do one Navier-Stokes time step.
            begin('Computing flux and pressure...')
            try:
                u, p = cyl_ns.ipcs_step(
                    problem.W, problem.P, dt,
                    mu[problem.wpi](problem.background_temp),
                    rho[problem.wpi](problem.background_temp),
                    u_1, problem.u_bcs,
                    p_1, problem.p_bcs,
                    f0=rho(theta_1) * g,
                    f1=rho(theta) * g,
                    theta=1.0,
                    stabilization=True,
                    tol=1.0e-10
                    )
            except RuntimeError as e:
                print(e.message)
                print('Navier--Stokes solver failed to converge. '
                      'Decrease time step from %e to %e and try again.' %
                      (dt, 0.5*dt)
                      )
                dt *= 0.5
                end()
                end()
                end()
                continue
            end()

            # Assignments and plotting.
            theta_1.assign(theta)
            u_1.assign(u)
            p_1.assign(p)

            # Save
            xdmf_file.write(theta_1, t+dt)
            xdmf_file.write(u_1, t+dt)
            xdmf_file.write(p_1, t+dt)

            plot(theta_1, title='temperature', rescale=True)
            plot(u_1, title='velocity', rescale=True)
            plot(p_1, title='pressure', rescale=True)
            # interactive()
            t += dt

            # Time update.
            begin('Step size adaptation...')
            u1, u2 = u_1.split()
            unorm = project(
                    abs(u1) + abs(u2),
                    problem.P,
                    form_compiler_parameters={'quadrature_degree': 4}
                    )
            unorm = norm(unorm.vector(), 'linf')
            # print('||u||_inf = %e' % unorm)
            unorm = max(unorm, DOLFIN_EPS)
            # http://scicomp.stackexchange.com/questions/2927/estimating-the-courant-number-for-the-navier-stokes-equations-under-differing-re
            rho_mu = rho[problem.wpi](problem.background_temp) \
                / mu[problem.wpi](problem.background_temp)
            target_dt = min(
                0.5*problem.mesh.hmin()/unorm,
                0.5*problem.mesh.hmin()**2 * rho_mu
                )
            print('previous dt: %e' % dt)
            print('target dt:  %e' % target_dt)
            # alpha is the aggressiveness factor. The distance between the
            # current step size and the target step size is reduced by
            # |1-alpha|. Hence, if alpha==1 then dt_next==target_dt. Otherwise
            # target_dt is approached more slowly.
            alpha = 0.5
            dt_max = 1.0e1
            dt = min(dt_max,
                     # At most double the step size from step to step.
                     dt * min(2.0, 1.0 + alpha*(target_dt - dt)/dt)
                     )
            print('new dt:    %e' % dt)
            end()
            end()
    return


if __name__ == '__main__':
    test()
