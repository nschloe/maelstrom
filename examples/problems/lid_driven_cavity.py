# -*- coding: utf-8 -*-
#
from dolfin import (
    UnitSquareMesh, SubDomain, FunctionSpace, DirichletBC, Constant,
    FiniteElement, DOLFIN_EPS
    )


class Lid_driven_cavity(object):
    def __init__(self):
        n = 40
        self.mesh = UnitSquareMesh(n, n, 'crossed')

        # Define mesh and boundaries.
        class LeftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0] < DOLFIN_EPS

        class RightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0] > 1.0-DOLFIN_EPS

        class LowerBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[1] < DOLFIN_EPS

        class UpperBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[1] > 1.0-DOLFIN_EPS

        class RestrictedUpperBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[1] > 1.0-DOLFIN_EPS \
                    and DOLFIN_EPS < x[0] and x[0] < 0.5-DOLFIN_EPS

        left = LeftBoundary()
        right = RightBoundary()
        lower = LowerBoundary()
        upper = UpperBoundary()
        # restricted_upper = RestrictedUpperBoundary()

        # Be particularly careful with the boundary conditions.
        # The main problem here is that the PPE system is consistent if and
        # only if
        #
        #     \int_\Omega div(u) = \int_\Gamma n.u = 0.
        #
        # This is exactly and even pointwise fulfilled for the continuous
        # problem.  In the discrete case, we can have to make sure that n.u is
        # 0 all along the boundary.
        # In the lid-driven cavity problem, of particular interest are the
        # corner points at the lid. One has to assert that the z-component of u
        # is 0 all across the lid, and the x-component of u is 0 everywhere but
        # the lid.  Since u is L2-"continuous", the lid condition on u_x must
        # not be enforced in the corner points. The u_y component must be
        # enforced all over the lid, including the end points.
        V_element = FiniteElement('CG', self.mesh.ufl_cell(), 2)
        self.W = FunctionSpace(self.mesh, V_element * V_element)

        self.u_bcs = [
            DirichletBC(self.W, (0.0, 0.0), left),
            DirichletBC(self.W, (0.0, 0.0), right),
            # DirichletBC(self.W.sub(0), Expression('x[0]'), restricted_upper),
            DirichletBC(self.W, (0.0, 0.0), lower),
            DirichletBC(self.W.sub(0), Constant('1.0'), upper),
            DirichletBC(self.W.sub(1), 0.0, upper),
            # DirichletBC(self.W.sub(0), Constant('-1.0'), lower),
            # DirichletBC(self.W.sub(1), 0.0, lower),
            # DirichletBC(self.W.sub(1), Constant('1.0'), left),
            # DirichletBC(self.W.sub(0), 0.0, left),
            # DirichletBC(self.W.sub(1), Constant('-1.0'), right),
            # DirichletBC(self.W.sub(0), 0.0, right),
            ]
        self.P = FunctionSpace(self.mesh, 'CG', 1)
        self.p_bcs = []
        return
