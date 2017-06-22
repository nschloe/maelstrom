#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from __future__ import print_function

from dolfin import (
    XDMFFile, Measure, FunctionSpace, begin, end, SubMesh, project, Function,
    assemble, grad, as_vector, DOLFIN_EPS, info, mpi_comm_world, FiniteElement,
    SpatialCoordinate, VectorFunctionSpace, norm
    )
import numpy
from numpy import pi, sin, cos

import maelstrom.maxwell as cmx

import problems


# pylint: disable=too-many-branches
def test():
    problem = problems.Crucible()

    # The voltage is defined as
    #
    #     v(t) = Im(exp(i omega t) v)
    #          = Im(exp(i (omega t + arg(v)))) |v|
    #          = sin(omega t + arg(v)) |v|.
    #
    # Hence, for a lagging voltage, arg(v) needs to be negative.
    voltages = [
        38.0 * numpy.exp(-1j * 2*pi * 2 * 70.0/360.0),
        38.0 * numpy.exp(-1j * 2*pi * 1 * 70.0/360.0),
        38.0 * numpy.exp(-1j * 2*pi * 0 * 70.0/360.0),
        25.0 * numpy.exp(-1j * 2*pi * 0 * 70.0/360.0),
        25.0 * numpy.exp(-1j * 2*pi * 1 * 70.0/360.0)
        ]
    #
    # voltages = [0.0, 0.0, 0.0, 0.0, 0.0]
    #
    # voltages = [
    #     25.0 * numpy.exp(-1j * 2*pi * 2 * 70.0/360.0),
    #     25.0 * numpy.exp(-1j * 2*pi * 1 * 70.0/360.0),
    #     25.0 * numpy.exp(-1j * 2*pi * 0 * 70.0/360.0),
    #     38.0 * numpy.exp(-1j * 2*pi * 0 * 70.0/360.0),
    #     38.0 * numpy.exp(-1j * 2*pi * 1 * 70.0/360.0)
    #     ]
    #
    # voltages = [
    #     38.0 * numpy.exp(+1j * 2*pi * 2 * 70.0/360.0),
    #     38.0 * numpy.exp(+1j * 2*pi * 1 * 70.0/360.0),
    #     38.0 * numpy.exp(+1j * 2*pi * 0 * 70.0/360.0),
    #     25.0 * numpy.exp(+1j * 2*pi * 0 * 70.0/360.0),
    #     25.0 * numpy.exp(+1j * 2*pi * 1 * 70.0/360.0)
    #     ]

    info('Input voltages:')
    info('%r' % voltages)

    # Merge coil rings with voltages.
    coils = []
    for coil_domain, voltage in zip(problem.coil_domains, voltages):
        coils.append({
            'rings': coil_domain,
            'c_type': 'voltage',
            'c_value': voltage
            })

    subdomain_indices = problem.subdomain_materials.keys()

    # Build subdomain parameter dictionaries.
    mu = {}
    sigma = {}
    for i in subdomain_indices:
        # Take all parameters at background_temp.
        # background_temp = 1500.0
        material = problem.subdomain_materials[i]
        mu[i] = material.magnetic_permeability
        sigma[i] = material.electrical_conductivity

    dx = Measure('dx')(subdomain_data=problem.subdomains)
    # boundaries = mesh.domains().facet_domains()
    # ds = Measure('ds')(subdomain_data=problem.subdomains)

    # Function space for Maxwell.
    V = FunctionSpace(problem.mesh, 'CG', 1)

    omega = 240

    # TODO when projected onto submesh, the time harmonic solver bails out
    # V_submesh = FunctionSpace(problem.submesh_workpiece, 'CG', 2)
    # u_1 = Function(V_submesh * V_submesh)
    # u_1.vector().zero()
    # conv = {problem.wpi: u_1}

    conv = {}

    Phi, voltages = cmx.compute_potential(
            coils,
            V,
            dx,
            mu, sigma, omega,
            convections=conv
            )

    # # show current in the first ring of the first coil
    # ii = coils[0]['rings'][0]
    # submesh_coil = SubMesh(mesh, subdomains, ii)
    # V1 = FunctionSpace(submesh_coil, 'CG', ii)

    # #File('phi.xdmf') << project(as_vector((Phi_r, Phi_i)), V*V)
    # from dolfin import plot
    # plot(Phi[0], title='Re(Phi)')
    # plot(Phi[1], title='Im(Phi)')
    # plot(project(Phi_r, V1), title='Re(Phi)')
    # plot(project(Phi_i, V1), title='Im(Phi)')
    # interactive()

    check_currents = False
    if check_currents:
        r = SpatialCoordinate(problem.mesh)[0]
        begin('Currents computed after the fact:')
        k = 0
        with XDMFFile('currents.xdmf') as xdmf_file:
            for coil in coils:
                for ii in coil['rings']:
                    J_r = sigma[ii] * (
                        voltages[k].real/(2*pi*r) + omega * Phi[1]
                        )
                    J_i = sigma[ii] * (
                        voltages[k].imag/(2*pi*r) - omega * Phi[0]
                        )
                    alpha = assemble(J_r * dx(ii))
                    beta = assemble(J_i * dx(ii))
                    info('J = {:e} + i {:e}'.format(alpha, beta))
                    info(
                        '|J|/sqrt(2) = {:e}'.format(
                            numpy.sqrt(0.5 * (alpha**2 + beta**2))
                        ))
                    submesh = SubMesh(problem.mesh, problem.subdomains, ii)
                    V1 = FunctionSpace(submesh, 'CG', 1)
                    # Those projections may take *very* long.
                    # TODO find out why
                    j_v1 = [
                        project(J_r, V1),
                        project(J_i, V1)
                        ]
                    # plot(j_v1[0], title='j_r')
                    # plot(j_v1[1], title='j_i')
                    # interactive()
                    current = project(as_vector(j_v1), V1*V1)
                    current.rename('j{}'.format(ii), 'current {}'.format(ii))
                    xdmf_file.write(current)
                    k += 1
        end()

    filename = './maxwell.xdmf'
    with XDMFFile(mpi_comm_world(), filename) as xdmf_file:
        xdmf_file.parameters['flush_output'] = True
        xdmf_file.parameters['rewrite_function_mesh'] = False

        # Store phi
        info('Writing out Phi to %s...' % filename)
        phi = Function(V, name='phi')
        Phi0 = project(Phi[0], V)
        Phi1 = project(Phi[1], V)
        for t in numpy.linspace(0.0, 2*pi/omega, num=100, endpoint=False):
            # Im(Phi * exp(i*omega*t))
            phi.vector().zero()
            phi.vector().axpy(sin(omega*t), Phi0.vector())
            phi.vector().axpy(cos(omega*t), Phi1.vector())
            xdmf_file.write(phi, t)

        # Show the resulting magnetic field
        #
        #   B_r = -dphi/dz,
        #   B_z = 1/r d(rphi)/dr.
        #
        r = SpatialCoordinate(problem.mesh)[0]
        g = 1.0/r * grad(r*Phi[0])

        V_element = FiniteElement('CG', V.mesh().ufl_cell(), 1)
        VV = FunctionSpace(V.mesh(), V_element * V_element)

        B_r = project(as_vector((-g[1], g[0])), VV)
        g = 1/r * grad(r*Phi[1])
        B_i = project(as_vector((-g[1], g[0])), VV)
        info('Writing out B to %s...' % filename)
        B = Function(VV)
        B.rename('B', 'magnetic field')
        if abs(omega) < DOLFIN_EPS:
            B.assign(B_r)
            xdmf_file.write(B)
            # plot(B_r, title='Re(B)')
            # plot(B_i, title='Im(B)')
            # interactive()
        else:
            # Write those out to a file.
            lspace = numpy.linspace(0.0, 2*pi/omega, num=100, endpoint=False)
            for t in lspace:
                # Im(B * exp(i*omega*t))
                B.vector().zero()
                B.vector().axpy(sin(omega*t), B_r.vector())
                B.vector().axpy(cos(omega*t), B_i.vector())
                xdmf_file.write(B, t)

    # Store Lorentz force and Joule heat source in file
    # Get resulting Lorentz force.
    lorentz_wpi = cmx.compute_lorentz(Phi, omega, sigma[problem.wpi])

    # Show the Lorentz force in the workpiece.
    # W_element = VectorElement('CG', submesh_workpiece.ufl_cell(), 1)
    # First project onto the entire mesh, then onto the submesh; see bug
    # <https://bitbucket.org/fenics-project/dolfin/issues/869/projecting-grad-onto-submesh-error>.
    W = VectorFunctionSpace(problem.mesh, 'CG', 1)
    pl = project(lorentz_wpi, W)
    W2 = VectorFunctionSpace(problem.submesh_workpiece, 'CG', 1)
    lorentz_fun = project(pl, W2)
    lorentz_fun.rename('F_L', 'Lorentz force')

    assert abs(norm(lorentz_fun, 'L2') - 0.8417945622831131) < 1.0e-3

    # plot(lfun, title='Lorentz force')
    # interactive()

    joule = cmx.compute_joule(
            Phi, voltages,
            omega, sigma, mu,
            subdomain_indices=[problem.wpi]
            )
    V2 = FunctionSpace(problem.submesh_workpiece, 'CG', 1)
    jp = project(joule[problem.wpi], V2)
    jp.rename('s', 'Joule heat source')

    assert abs(norm(jp, 'L2') - 32.232276325879475) < 1.0e-3

    # plot(jp, title='heat source')
    # interactive()

    filename = './lorentz-joule.xdmf'
    info('Writing out Lorentz force and Joule heat source to {}...'.format(
        filename
        ))
    with XDMFFile(mpi_comm_world(), filename) as xdmf_file:
        # xdmf_file.parameters['flush_output'] = True
        # xdmf_file.parameters['rewrite_function_mesh'] = False
        xdmf_file.write(jp, 0.0)
        xdmf_file.write(lorentz_fun, 0.0)

    # # For the lulz: solve heat equation with the Joule source.
    # u = TrialFunction(V)
    # v = TestFunction(V)
    # a = zero() * dx(0)
    # r = SpatialCoordinate(problem.mesh)[0]
    # # v/r doesn't hurt: hom. dirichlet boundary for r=0.
    # for i in subdomain_indices:
    #     a += dot(kappa[i] * r * grad(u), grad(v/r)) * dx(i)
    # sol = Function(V)
    # class OuterBoundary(SubDomain):
    #     def inside(self, x, on_boundary):
    #         return on_boundary and abs(x[0]) > DOLFIN_EPS
    # outer_boundary = OuterBoundary()
    # bcs = DirichletBC(V, background_temp, outer_boundary)
    # solve(a == joule, sol, bcs=bcs)
    # plot(sol)
    # interactive()
    return


if __name__ == '__main__':
    test()
