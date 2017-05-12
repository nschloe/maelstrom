# -*- coding: utf-8 -*-
#
from . import meshes

from dolfin import (
    Mesh, FunctionSpace, SubDomain, DirichletBC, FiniteElement
    )
import meshio


GMSH_EPS = 1.0e-15


class Ball_in_tube(object):
    def __init__(self):
        # https://fenicsproject.org/qa/12891/initialize-mesh-from-vertices-connectivities-at-once
        points, cells, point_data, cell_data, _ = \
            meshes.ball_in_tube_cyl.generate()
        tmp_filename = 'test.xml'
        meshio.write(
                tmp_filename, points, cells, cell_data=cell_data,
                file_format='dolfin-xml'
                )
        self.mesh = Mesh('test.xml')

        V0_element = FiniteElement('CG', self.mesh.ufl_cell(), 2)
        V1_element = FiniteElement('B', self.mesh.ufl_cell(), 3)
        self.W = FunctionSpace(self.mesh, V0_element * V1_element)

        self.P = FunctionSpace(self.mesh, 'CG', 1)

        # Define mesh and boundaries.
        class LeftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0] < GMSH_EPS
        left_boundary = LeftBoundary()

        class RightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0] > 1.0 - GMSH_EPS
        right_boundary = RightBoundary()

        class LowerBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[1] < GMSH_EPS
        lower_boundary = LowerBoundary()

        class UpperBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[1] > 5.0-GMSH_EPS

        class CoilBoundary(SubDomain):
            def inside(self, x, on_boundary):
                # One has to pay a little bit of attention when defining the
                # coil boundary; it's easy to miss the edges closest to x[0]=0.
                return on_boundary \
                    and x[1] > 1.0-GMSH_EPS and x[1] < 2.0+GMSH_EPS \
                    and x[0] < 1.0-GMSH_EPS
        coil_boundary = CoilBoundary()

        self.u_bcs = [
            DirichletBC(self.W, (0.0, 0.0), right_boundary),
            DirichletBC(self.W.sub(0), 0.0, left_boundary),
            DirichletBC(self.W, (0.0, 0.0), lower_boundary),
            DirichletBC(self.W, (0.0, 0.0), coil_boundary)
            ]
        self.p_bcs = []
        # self.p_bcs = [DirichletBC(Q, 0.0, upper_boundary)]
        return
