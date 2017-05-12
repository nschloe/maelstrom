# -*- coding: utf-8 -*-
#
from dolfin import (
    UnitSquareMesh, SubDomain, FunctionSpace, DirichletBC, Constant,
    FiniteElement, MixedElement, SpatialCoordinate
    )


class Rotating_lid():

    def __init__(self):
        self.mesh = UnitSquareMesh(20, 20, 'left/right')

        eps = 1.0e-15

        class LeftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0] < eps

        class RightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0] > 1.0 - eps

        class LowerBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[1] < eps

        class UpperBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[1] > 1.0 - eps

        class RestrictedUpperBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[1] > 1.0-eps \
                    and eps < x[0] and x[0] < 0.5-eps

        left = LeftBoundary()
        right = RightBoundary()
        lower = LowerBoundary()
        upper = UpperBoundary()
        restricted_upper = RestrictedUpperBoundary()

        V_element = FiniteElement('CG', self.mesh.ufl_cell(), 2)

        self.W = FunctionSpace(self.mesh, MixedElement(3*[V_element]))

        x0 = SpatialCoordinate(self.mesh)[0]
        self.u_bcs = [
                DirichletBC(self.W.sub(0), 0.0, left),
                DirichletBC(self.W.sub(2), 0.0, left),
                DirichletBC(self.W, Constant((0.0, 0.0, 0.0)), right),
                DirichletBC(self.W.sub(0), 0.0, upper),
                DirichletBC(self.W.sub(1), 0.0, upper),
                DirichletBC(self.W.sub(2), x0, restricted_upper),
                DirichletBC(self.W, Constant((0.0, 0.0, 0.0)), lower),
                ]

        self.P = FunctionSpace(self.mesh, 'CG', 1)
        self.p_bcs = []
        return
