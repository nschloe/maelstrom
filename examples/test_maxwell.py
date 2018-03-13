#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from __future__ import print_function

from dolfin import (
    XDMFFile, Measure, FunctionSpace, SubMesh, project, Function, info,
    VectorFunctionSpace, norm, Constant, plot, SpatialCoordinate, grad,
    FiniteElement, DOLFIN_EPS, as_vector
    )
import matplotlib.pyplot as plt
import numpy
from numpy import pi
from numpy import sin, cos

import maelstrom.maxwell as cmx
from maelstrom.message import Message

import problems


def test(show=False):
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

    lorentz, joule, Phi = get_lorentz_joule(problem, voltages, show=show)

    # Some assertions
    ref = 1.4627674791126285e-05
    assert abs(norm(Phi[0], 'L2') - ref) < 1.0e-3 * ref
    ref = 3.161363929287592e-05
    assert abs(norm(Phi[1], 'L2') - ref) < 1.0e-3 * ref
    #
    ref = 12.115309575057681
    assert abs(norm(lorentz, 'L2') - ref) < 1.0e-3 * ref
    #
    ref = 1406.336109054347
    V = FunctionSpace(problem.submesh_workpiece, 'CG', 1)
    jp = project(joule, V)
    jp.rename('s', 'Joule heat source')
    assert abs(norm(jp, 'L2') - ref) < 1.0e-3 * ref

    # check_currents = False
    # if check_currents:
    #     r = SpatialCoordinate(problem.mesh)[0]
    #     begin('Currents computed after the fact:')
    #     k = 0
    #     with XDMFFile('currents.xdmf') as xdmf_file:
    #         for coil in coils:
    #             for ii in coil['rings']:
    #                 J_r = sigma[ii] * (
    #                     voltages[k].real/(2*pi*r) + problem.omega * Phi[1]
    #                     )
    #                 J_i = sigma[ii] * (
    #                     voltages[k].imag/(2*pi*r) - problem.omega * Phi[0]
    #                     )
    #                 alpha = assemble(J_r * dx(ii))
    #                 beta = assemble(J_i * dx(ii))
    #                 info('J = {:e} + i {:e}'.format(alpha, beta))
    #                 info(
    #                     '|J|/sqrt(2) = {:e}'.format(
    #                         numpy.sqrt(0.5 * (alpha**2 + beta**2))
    #                     ))
    #                 submesh = SubMesh(problem.mesh, problem.subdomains, ii)
    #                 V1 = FunctionSpace(submesh, 'CG', 1)
    #                 # Those projections may take *very* long.
    #                 # TODO find out why
    #                 j_v1 = [
    #                     project(J_r, V1),
    #                     project(J_i, V1)
    #                     ]
    #                 # show=Trueplot(j_v1[0], title='j_r')
    #                 # plot(j_v1[1], title='j_i')
    #                 current = project(as_vector(j_v1), V1*V1)
    #                 current.rename('j{}'.format(ii), 'current {}'.format(ii))
    #                 xdmf_file.write(current)
    #                 k += 1
    #     end()

    filename = './maxwell.xdmf'
    with XDMFFile(filename) as xdmf_file:
        xdmf_file.parameters['flush_output'] = True
        xdmf_file.parameters['rewrite_function_mesh'] = False

        # Store phi
        info('Writing out Phi to {}...'.format(filename))
        V = FunctionSpace(problem.mesh, 'CG', 1)
        phi = Function(V, name='phi')
        Phi0 = project(Phi[0], V)
        Phi1 = project(Phi[1], V)
        omega = problem.omega
        for t in numpy.linspace(0.0, 2*pi/omega, num=100, endpoint=False):
            # Im(Phi * exp(i*omega*t))
            phi.vector().zero()
            phi.vector().axpy(sin(problem.omega*t), Phi0.vector())
            phi.vector().axpy(cos(problem.omega*t), Phi1.vector())
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
        if abs(problem.omega) < DOLFIN_EPS:
            B.assign(B_r)
            xdmf_file.write(B)
            # plot(B_r, title='Re(B)')
            # plot(B_i, title='Im(B)')
        else:
            # Write those out to a file.
            lspace = numpy.linspace(
                0.0, 2*pi/problem.omega, num=100, endpoint=False
                )
            for t in lspace:
                # Im(B * exp(i*omega*t))
                B.vector().zero()
                B.vector().axpy(sin(problem.omega*t), B_r.vector())
                B.vector().axpy(cos(problem.omega*t), B_i.vector())
                xdmf_file.write(B, t)

    filename = './lorentz-joule.xdmf'
    info('Writing out Lorentz force and Joule heat source to {}...'.format(
        filename
        ))
    with XDMFFile(filename) as xdmf_file:
        xdmf_file.write(lorentz, 0.0)
        # xdmf_file.write(jp, 0.0)

    return


def get_lorentz_joule(problem, input_voltages, show=False):
    submesh_workpiece = problem.W.mesh()

    subdomain_indices = problem.subdomain_materials.keys()

    info('Input voltages:')
    info(repr(input_voltages))

    if input_voltages is None:
        return None, Constant(0.0)

    # Merge coil rings with voltages.
    coils = [
        {'rings': coil_domain, 'c_type': 'voltage', 'c_value': voltage}
        for coil_domain, voltage in zip(problem.coil_domains, input_voltages)
        ]
    # Build subdomain parameter dictionaries for Maxwell
    mu_const = {
        i: problem.subdomain_materials[i].magnetic_permeability
        for i in subdomain_indices
        }
    sigma_const = {
        i: problem.subdomain_materials[i].electrical_conductivity
        for i in subdomain_indices
        }

    # Function space for magnetic scalar potential, Lorentz force etc.
    V = FunctionSpace(problem.mesh, 'CG', 1)
    # Compute the magnetic field.
    # The Maxwell equations depend on two parameters that change during the
    # computation: (a) the temperature, and (b) the velocity field u0. We
    # assume though that changes in either of the two will only marginally
    # influence the magnetic field. Consequently, we precompute all associated
    # values.
    dx_subdomains = Measure('dx', subdomain_data=problem.subdomains)
    with Message('Computing magnetic field...'):
        Phi, voltages = cmx.compute_potential(
            coils,
            V,
            dx_subdomains,
            mu_const, sigma_const, problem.omega,
            convections={}
            # io_submesh=submesh_workpiece
            )
        # Get resulting Lorentz force.
        lorentz = cmx.compute_lorentz(
            Phi, problem.omega, sigma_const[problem.wpi]
            )

        # Show the Lorentz force in the workpiece.
        # W_element = VectorElement('CG', submesh_workpiece.ufl_cell(), 1)
        # First project onto the entire mesh, then onto the submesh; see bug
        # <https://bitbucket.org/fenics-project/dolfin/issues/869/projecting-grad-onto-submesh-error>.
        W = VectorFunctionSpace(problem.mesh, 'CG', 1)
        pl = project(lorentz, W)
        W2 = VectorFunctionSpace(submesh_workpiece, 'CG', 1)
        pl = project(pl, W2)
        pl.rename('Lorentz force', 'Lorentz force')
        with XDMFFile(submesh_workpiece.mpi_comm(), 'lorentz.xdmf') as f:
            f.parameters['flush_output'] = True
            f.write(pl)

        if show:
            tri = plot(pl, title='Lorentz force')
            plt.colorbar(tri)
            plt.show()

        # Get Joule heat source.
        joule = cmx.compute_joule(
            Phi, voltages,
            problem.omega, sigma_const, mu_const,
            subdomain_indices
            )

        if show:
            # Show Joule heat source.
            submesh = SubMesh(problem.mesh, problem.subdomains, problem.wpi)
            W_submesh = FunctionSpace(submesh, 'CG', 1)
            jp = Function(W_submesh, name='Joule heat source')
            jp.assign(project(joule[problem.wpi], W_submesh))
            tri = plot(jp)
            plt.title('Joule heat source')
            plt.colorbar(tri)
            plt.show()

        joule_wpi = joule[problem.wpi]

    # To work around bug
    # <https://bitbucket.org/fenics-project/dolfin/issues/869/projecting-grad-onto-submesh-error>.
    # return the projection `pl` and not `lorentz` itself.
    # TODO remove this workaround
    return pl, joule_wpi, Phi


if __name__ == '__main__':
    test(show=True)
