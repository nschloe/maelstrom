# -*- coding: utf-8 -*-
#
#  Copyright (c) 2012--2014, Nico Schlömer, <nico.schloemer@gmail.com>
#  All rights reserved.
#
#  This file is part of Maelstrom.
#
#  Maelstrom is free software: you can redistribute it and/or modify it under
#  the terms of the GNU General Public License as published by the Free
#  Software Foundation, either version 3 of the License, or (at your option)
#  any later version.
#
#  Maelstrom is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#  FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
#  details.
#
#  You should have received a copy of the GNU General Public License along with
#  Maelstrom.  If not, see <http://www.gnu.org/licenses/>.
#

DEBUG = False


class CrucibleProblem():

    def __init__(self):
        import os
        from dolfin import Mesh, MeshFunction, SubMesh, SubDomain, \
            FacetFunction, DirichletBC, dot, grad, FunctionSpace, \
            MixedFunctionSpace, Expression, FacetNormal, pi, Function, \
            Constant, TestFunction, MPI, mpi_comm_world, File
        import numpy
        import warnings

        from maelstrom import heat_cylindrical as cyl_heat
        from maelstrom import materials_database as md

        GMSH_EPS = 1.0e-15

        current_path = os.path.dirname(os.path.realpath(__file__))
        base = os.path.join(
            current_path,
            '../../meshes/2d/crucible-with-coils'
            )
        self.mesh = Mesh(base + '.xml')
        self.subdomains = MeshFunction('size_t',
                                       self.mesh,
                                       base + '_physical_region.xml'
                                       )

        self.subdomain_materials = {
            1: md.get_material('porcelain'),
            2: md.get_material('argon'),
            3: md.get_material('GaAs (solid)'),
            4: md.get_material('GaAs (liquid)'),
            27: md.get_material('air')
            }

        # coils
        for k in range(5, 27):
            self.subdomain_materials[k] = md.get_material('graphite EK90')

        # Define the subdomains which together form a single coil.
        self.coil_domains = [
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23],
            [24, 25, 26]
            ]

        self.wpi = 4
        # http://fenicsproject.org/qa/2026/submesh-workaround-for-parallel-computation
        submesh_parallel_bug_fixed = False
        if submesh_parallel_bug_fixed:
            submesh_workpiece = SubMesh(self.mesh, self.subdomains, self.wpi)
        else:
            # To get the mesh in parallel, we need to read it in from a file.
            # Writing out can only happen in serial mode, though. :/
            base = os.path.join(current_path,
                                '../../meshes/2d/crucible-with-coils-submesh'
                                )
            filename = base + '.xml'
            if not os.path.isfile(filename):
                warnings.warn(
                    'Submesh file \'%s\' does not exist. Creating... '
                    % filename
                    )
                if MPI.size(mpi_comm_world()) > 1:
                    raise RuntimeError(
                        'Can only write submesh in serial mode.'
                        )
                submesh_workpiece = \
                    SubMesh(self.mesh, self.subdomains, self.wpi)
                output_stream = File(filename)
                output_stream << submesh_workpiece
            # Read the mesh
            submesh_workpiece = Mesh(base + '.xml')

        coords = submesh_workpiece.coordinates()
        ymin = min(coords[:, 1])
        ymax = max(coords[:, 1])

        # Find the top right point.
        k = numpy.argmax(numpy.sum(coords, 1))
        topright = coords[k, :]

        # Initialize mesh function for boundary domains
        class Left(SubDomain):
            def inside(self, x, on_boundary):
                # Explicitly exclude the lowest and the highest point of the
                # symmetry axis.
                # It is necessary for the consistency of the pressure-Poisson
                # system in the Navier-Stokes solver that the velocity is
                # exactly 0 at the boundary r>0. Hence, at the corner points
                # (r=0, melt-crucible, melt-crystal) we must enforce u=0
                # already and cannot have a component in z-direction.
                return on_boundary \
                    and x[0] < GMSH_EPS \
                    and x[1] < ymax - GMSH_EPS \
                    and x[1] > ymin + GMSH_EPS

        class Crucible(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary \
                    and ((x[0] > GMSH_EPS and x[1] < ymax - GMSH_EPS)
                         or (x[0] > topright[0] - GMSH_EPS
                             and x[1] > topright[1] - GMSH_EPS)
                         or (x[0] < GMSH_EPS and x[1] < ymin + GMSH_EPS)
                         )

        # At the top right part (boundary melt--gas), slip is allowed, so only
        # n.u=0 is enforced. Very weirdly, the PPE is consistent if and only if
        # the end points of UpperRight are in UpperRight. This contrasts
        # Left(), where the end points must NOT belong to Left().  Judging from
        # the experiments, these settings do the right thing.
        # TODO try to better understand the PPE system/dolfin's boundary
        # settings
        class Upper(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary \
                    and x[1] > ymax - GMSH_EPS

        class UpperRight(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary \
                    and x[1] > ymax - GMSH_EPS \
                    and x[0] > 0.038 - GMSH_EPS

        # The crystal boundary is taken to reach up to 0.038 where the
        # Dirichlet boundary data is about the melting point of the crystal,
        # 1511K. This setting gives pretty acceptable results when there is no
        # convection except the one induced by buoyancy. Is there is any more
        # stirring going on, though, the end point of the crystal with its
        # fixed temperature of 1511K might be the hottest point globally. This
        # looks rather unphysical.
        # TODO check out alternatives
        class UpperLeft(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary \
                    and x[1] > ymax - GMSH_EPS \
                    and x[0] < 0.038 + GMSH_EPS

        left = Left()
        crucible = Crucible()
        upper_left = UpperLeft()
        upper_right = UpperRight()

        self.wp_boundaries = FacetFunction('size_t', submesh_workpiece)
        self.wp_boundaries.set_all(0)
        left.mark(self.wp_boundaries, 1)
        crucible.mark(self.wp_boundaries, 2)
        upper_right.mark(self.wp_boundaries, 3)
        upper_left.mark(self.wp_boundaries, 4)

        if DEBUG:
            from dolfin import plot, interactive
            plot(self.wp_boundaries, title='Boundaries')
            interactive()

        submesh_boundary_indices = {
            'left': 1,
            'crucible': 2,
            'upper right': 3,
            'upper left': 4
            }

        # Boundary conditions for the velocity.
        #
        # [1] Incompressible flow and the finite element method; volume two;
        #     Isothermal Laminar Flow;
        #     P.M. Gresho, R.L. Sani;
        #
        # For the choice of function space, [1] says:
        #     "In 2D, the triangular elements P_2^+P_1 and P_2^+P_{-1} are very
        #      good [...]. [...] If you wish to avoid bubble functions on
        #      triangular elements, P_2P_1 is not bad, and P_2(P_1+P_0) is even
        #      better [...]."
        #
        # It turns out that adding the bubble space significantly hampers the
        # convergence of the Stokes solver and also considerably increases the
        # time it takes to construct the Jacobian matrix of the Navier--Stokes
        # problem if no optimization is applied.
        V = FunctionSpace(submesh_workpiece, 'CG', 2)
        with_bubbles = False
        if with_bubbles:
            V += FunctionSpace(submesh_workpiece, 'B', 3)
        self.W = MixedFunctionSpace([V, V, V])

        self.u_bcs = [
            DirichletBC(self.W,
                        Expression(
                            ('0.0', '0.0', '-2*pi*x[0] * 5.0/60.0'),
                            degree=1
                            ),
                        #(0.0, 0.0, 0.0),
                        crucible),
            DirichletBC(self.W.sub(0), 0.0, left),
            DirichletBC(self.W.sub(2), 0.0, left),
            # Make sure that u[2] is 0 at r=0.
            DirichletBC(self.W,
                        Expression(
                            ('0.0', '0.0', '2*pi*x[0] * 5.0/60.0'),
                            degree=1
                            ),
                        upper_left),
            DirichletBC(self.W.sub(1), 0.0, upper_right),
            ]
        self.p_bcs = []

        self.P = FunctionSpace(submesh_workpiece, 'CG', 1)

        # Boundary conditions for heat equation.
        self.Q = FunctionSpace(submesh_workpiece, 'CG', 2)
        # Dirichlet.
        # This is a bit of a tough call since the boundary conditions need to
        # be read from a Tecplot file here.
        import tecplot_reader
        filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'data/crucible-boundary.dat'
            )
        data = tecplot_reader.read(filename)
        RZ = numpy.c_[data['ZONE T']['node data']['r'],
                      data['ZONE T']['node data']['z']
                      ]
        T_vals = data['ZONE T']['node data']['temp. [K]']

        class TecplotDirichletBC(Expression):
            # TODO specify degree
            def eval(self, value, x):
                # Find on which edge x sits, and raise exception if it doesn't.
                edge_found = False
                for edge in data['ZONE T']['element data']:
                    # Given a point X and an edge X0--X1,
                    #
                    #     (1 - theta) X0 + theta X1,
                    #
                    # the minimum distance is assumed for
                    #
                    #    argmin_theta ||(1-theta) X0  + theta X1 - X||^2
                    #    = <X1 - X0, X - X0> / ||X1 - X0||^2.
                    #
                    # If the distance is 0 and 0<=theta<=1, we found the edge.
                    #
                    # Note that edges are 1-based in Tecplot.
                    X0 = RZ[edge[0] - 1]
                    X1 = RZ[edge[1] - 1]
                    theta = numpy.dot(X1-X0, x-X0) / numpy.dot(X1-X0, X1-X0)
                    diff = (1.0-theta)*X0 + theta*X1 - x
                    if numpy.dot(diff, diff) < 1.0e-10 and \
                            0.0 <= theta and theta <= 1.0:
                        # Linear interpolation of the temperature value.
                        value[0] = (1.0-theta) * T_vals[edge[0]-1] \
                                 + theta       * T_vals[edge[1]-1]
                        edge_found = True
                        break
                # This class is supposed to be used for Dirichlet boundary
                # conditions. For some reason, FEniCS also evaluates
                # DirichletBC objects at coordinates which do not sit on the
                # boundary, see
                # <http://fenicsproject.org/qa/1033/dirichletbc-expressions-evaluated-away-from-the-boundary>.
                # The assigned values have no meaning though, so not assigning
                # values[0] here is okay.
                #
                #from matplotlib import pyplot as pp
                #pp.plot(x[0], x[1], 'xg')
                if not edge_found:
                    value[0] = 0.0
                    warnings.warn('Coordinate (%e, %e) doesn\'t sit on edge.'
                                  % (x[0], x[1]))
                    #pp.plot(RZ[:, 0], RZ[:, 1], '.k')
                    #pp.plot(x[0], x[1], 'xr')
                    #pp.show()
                    #raise RuntimeError('Input coordinate '
                    #                   '%r is not on boundary.' % x)
                return

        tecplot_dbc = TecplotDirichletBC()
        self.theta_bcs_d = [
            DirichletBC(self.Q, tecplot_dbc, upper_left)
            ]
        theta_bcs_d_strict = [
            DirichletBC(self.Q, tecplot_dbc, upper_right),
            DirichletBC(self.Q, tecplot_dbc, crucible),
            DirichletBC(self.Q, tecplot_dbc, upper_left)
            ]

        # Neumann
        dTdr_vals = data['ZONE T']['node data']['dTempdx [K/m]']
        dTdz_vals = data['ZONE T']['node data']['dTempdz [K/m]']

        class TecplotNeumannBC(Expression):
            # TODO specify degree
            def eval(self, value, x):
                # Same problem as above: This expression is not only evaluated
                # at boundaries.
                for edge in data['ZONE T']['element data']:
                    X0 = RZ[edge[0] - 1]
                    X1 = RZ[edge[1] - 1]
                    theta = numpy.dot(X1-X0, x-X0) / numpy.dot(X1-X0, X1-X0)
                    dist = numpy.linalg.norm((1-theta)*X0 + theta*X1 - x)
                    if dist < 1.0e-5 and 0.0 <= theta and theta <= 1.0:
                        value[0] = (1-theta) * dTdr_vals[edge[0]-1] \
                            + theta * dTdr_vals[edge[1]-1]
                        value[1] = (1-theta) * dTdz_vals[edge[0]-1] \
                            + theta * dTdz_vals[edge[1]-1]
                        break
                return

            def value_shape(self):
                return (2,)

        tecplot_nbc = TecplotNeumannBC()
        n = FacetNormal(self.Q.mesh())
        self.theta_bcs_n = {
            submesh_boundary_indices['upper right']: dot(n, tecplot_nbc),
            submesh_boundary_indices['crucible']: dot(n, tecplot_nbc)
            }
        self.theta_bcs_r = {}

        # It seems that the boundary conditions from above are inconsistent in
        # that solving with Dirichlet overall and mixed Dirichlet-Neumann give
        # different results; the value *cannot* correspond to one solution.
        # From looking at the solutions, the pure Dirichlet setting appears
        # correct, so extract the Neumann values directly from that solution.
        zeta = TestFunction(self.Q)

        theta_reference = Function(self.Q, name='temperature (Dirichlet)')
        theta_reference.vector()[:] = 0.0

        # Solve the *quasilinear* PDE (coefficients may depend on theta).
        # This is to avoid setting a fixed temperature for the coefficients.

        # Get material parameters
        wp_material = self.subdomain_materials[self.wpi]
        if isinstance(wp_material.specific_heat_capacity, float):
            cp = wp_material.specific_heat_capacity
        else:
            cp = wp_material.specific_heat_capacity(theta_reference)
        if isinstance(wp_material.density, float):
            rho = wp_material.density
        else:
            rho = wp_material.density(theta_reference)
        if isinstance(wp_material.thermal_conductivity, float):
            k = wp_material.thermal_conductivity
        else:
            k = wp_material.thermal_conductivity(theta_reference)

        reference_problem = cyl_heat.HeatCylindrical(
            self.Q, theta_reference,
            zeta,
            b=Constant((0.0, 0.0, 0.0)),
            kappa=k,
            rho=rho,
            cp=cp,
            source=Constant(0.0),
            dirichlet_bcs=theta_bcs_d_strict
            )

        from dolfin import solve
        solve(reference_problem.F0 == 0,
              theta_reference,
              bcs=theta_bcs_d_strict
              )

        # Create equivalent boundary conditions from theta_reference. This
        # makes sure that the potentially expensive Expression evaluation in
        # theta_bcs_* is replaced by something reasonably cheap.
        for k, bc in enumerate(self.theta_bcs_d):
            self.theta_bcs_d[k] = DirichletBC(bc.function_space(),
                                              theta_reference,
                                              bc.domain_args[0]
                                              )
        # Adapt Neumann conditions.
        n = FacetNormal(self.Q.mesh())
        for k in self.theta_bcs_n:
            self.theta_bcs_n[k] = dot(n, grad(theta_reference))

        if DEBUG:
            # Solve the heat equation with the mixed Dirichlet-Neumann
            # boundary conditions and compare it to the Dirichlet-only
            # solution.
            theta_new = Function(
                self.Q,
                name='temperature (Neumann + Dirichlet)'
                )
            from dolfin import Measure
            ds_workpiece = Measure('ds')[self.wp_boundaries]
            problem_new = cyl_heat.HeatCylindrical(
                self.Q, theta_new,
                zeta,
                b=Constant((0.0, 0.0, 0.0)),
                kappa=k,
                rho=rho,
                cp=cp,
                source=Constant(0.0),
                dirichlet_bcs=self.theta_bcs_d,
                neumann_bcs=self.theta_bcs_n,
                ds=ds_workpiece
                )

            from dolfin import solve
            solve(problem_new.F0 == 0,
                  theta_new,
                  bcs=problem_new.dirichlet_bcs
                  )
            from dolfin import plot, interactive, errornorm
            print('||theta_new - theta_ref|| = %e'
                  % errornorm(theta_new, theta_reference)
                  )
            plot(theta_reference)
            plot(theta_new)
            plot(
                theta_reference - theta_new,
                title='theta_ref - theta_new'
                )
            interactive()

        #omega = 2 * pi * 10.0e3
        self.omega = 2 * pi * 300.0

        return
