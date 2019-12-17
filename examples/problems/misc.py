# -*- coding: utf-8 -*-
#
"""
Collection of (test) problems.
"""
import numpy
from dolfin import (
    DOLFIN_EPS,
    CellFunction,
    Constant,
    DirichletBC,
    Expression,
    FunctionSpace,
    Mesh,
    MeshFunction,
    MixedFunctionSpace,
    RectangleMesh,
    SubDomain,
    SubMesh,
    UnitSquareMesh,
    VectorFunctionSpace,
    between,
    near,
    pi,
)

GMSH_EPS = 1.0e-15


def crucible_without_coils():
    base = "../meshes/2d/crucible"
    mesh = Mesh(base + ".xml")
    subdomains = MeshFunction("size_t", mesh, base + "_physical_region.xml")
    workpiece_index = 4
    subdomain_materials = {
        1: "SiC",
        2: "boron trioxide",
        3: "GaAs (solid)",
        4: "GaAs (liquid)",
    }

    submesh_workpiece = SubMesh(mesh, subdomains, workpiece_index)
    V = VectorFunctionSpace(submesh_workpiece, "CG", 2)
    P = FunctionSpace(submesh_workpiece, "CG", 1)

    Q = FunctionSpace(mesh, "CG", 2)

    # Define mesh and boundaries.
    class LeftBoundary(SubDomain):
        def inside(self, x, on_boundary):
            # Be especially careful in the corners when defining the r=0 axis:
            # For the PPE to be consistent, we need to have n.u=0 *everywhere*
            # on the non-r=0 boundaries.
            return (
                on_boundary
                and x[0] < GMSH_EPS
                and 0.366 + GMSH_EPS < x[1]
                and x[1] < 0.411 - GMSH_EPS
            )

    left_boundary = LeftBoundary()

    class SurfaceBoundary(SubDomain):
        def inside(self, x, on_boundary):
            # Make sure to catch the entire surface, so don't be too
            # restrictive about x[0].
            return (
                on_boundary
                and x[1] > 0.38 - GMSH_EPS
                and x[0] > 0.04 - GMSH_EPS
                and x[0] < 0.07 + GMSH_EPS
            )

    class OtherBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return (
                on_boundary
                and not left_boundary.inside(x, on_boundary)
                # and not surface_boundary.inside(x, on_boundary)
            )

    other_boundary = OtherBoundary()

    # Boundary conditions for the velocity.
    u_bcs = [
        DirichletBC(V, (0.0, 0.0), other_boundary),
        DirichletBC(V.sub(0), 0.0, left_boundary),
        # DirichletBC(V.sub(1), 0.0, surface_boundary),
    ]
    # u_bcs = [DirichletBC(V, (0.0, 0.0), 'on_boundary')]
    p_bcs = []

    class HeaterBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (
                (x[1] < 0.38 and GMSH_EPS < x[0] and x[0] < 0.075 + GMSH_EPS)
                or x[0] < GMSH_EPS
                and x[1] < 0.365 + GMSH_EPS
            )

    heater_boundary = HeaterBoundary()

    background_temp = 1400.0

    # Heat with 1580K at r=0, 1590K at r=0.075.
    heater_bcs = [(heater_boundary, "1580.0 + 133.0 * x[0]")]

    return (
        mesh,
        subdomains,
        subdomain_materials,
        workpiece_index,
        V,
        Q,
        P,
        u_bcs,
        p_bcs,
        background_temp,
        heater_bcs,
    )


def ball_in_tube():
    mesh = Mesh("../meshes/2d/ball-in-tube-cylindrical-coarse4.xml")
    V0 = FunctionSpace(mesh, "CG", 2)
    V1 = FunctionSpace(mesh, "B", 3)
    V = V0 + V1
    W = MixedFunctionSpace([V, V])

    P = FunctionSpace(mesh, "CG", 1)

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
            return on_boundary and x[1] > 5.0 - GMSH_EPS

    class CoilBoundary(SubDomain):
        def inside(self, x, on_boundary):
            # One has to pay a little bit of attention when defining the
            # coil boundary; it's easy to miss the edges closest to x[0]=0.
            return (
                on_boundary
                and x[1] > 1.0 - GMSH_EPS
                and x[1] < 2.0 + GMSH_EPS
                and x[0] < 1.0 - GMSH_EPS
            )

    coil_boundary = CoilBoundary()

    u_bcs = [
        DirichletBC(W, (0.0, 0.0), right_boundary),
        DirichletBC(W.sub(0), 0.0, left_boundary),
        DirichletBC(W, (0.0, 0.0), lower_boundary),
        DirichletBC(W, (0.0, 0.0), coil_boundary),
    ]
    p_bcs = []
    # p_bcs = [DirichletBC(Q, 0.0, upper_boundary)]
    return mesh, W, P, u_bcs, p_bcs


def rotating_lid():
    mesh = UnitSquareMesh(20, 20, "left/right")

    # Define mesh and boundaries.
    class LeftBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] < GMSH_EPS

    class RightBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] > 1.0 - GMSH_EPS

    class LowerBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] < GMSH_EPS

    class UpperBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] > 1.0 - DOLFIN_EPS

    class RestrictedUpperBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return (
                on_boundary
                and x[1] > 1.0 - DOLFIN_EPS
                and DOLFIN_EPS < x[0]
                and x[0] < 0.5 - DOLFIN_EPS
            )

    left = LeftBoundary()
    right = RightBoundary()
    lower = LowerBoundary()
    upper = UpperBoundary()
    restricted_upper = RestrictedUpperBoundary()

    V = FunctionSpace(mesh, "CG", 2)

    W = MixedFunctionSpace([V, V, V])
    u_bcs = [
        DirichletBC(W.sub(0), 0.0, left),
        DirichletBC(W.sub(2), 0.0, left),
        DirichletBC(W, Constant((0.0, 0.0, 0.0)), right),
        DirichletBC(W.sub(0), 0.0, upper),
        DirichletBC(W.sub(1), 0.0, upper),
        DirichletBC(W.sub(2), Expression("x[0]", degree=1), restricted_upper),
        DirichletBC(W, Constant((0.0, 0.0, 0.0)), lower),
    ]

    P = FunctionSpace(mesh, "CG", 1)
    p_bcs = []
    return mesh, W, P, u_bcs, p_bcs


def lid_driven_cavity():
    n = 40
    mesh = RectangleMesh(0.0, 0.0, 1.0, 1.0, n, n, "crossed")

    # Define mesh and boundaries.
    class LeftBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] < DOLFIN_EPS

    class RightBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] > 1.0 - DOLFIN_EPS

    class LowerBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] < DOLFIN_EPS

    class UpperBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] > 1.0 - DOLFIN_EPS

    class RestrictedUpperBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return (
                on_boundary
                and x[1] > 1.0 - DOLFIN_EPS
                and DOLFIN_EPS < x[0]
                and x[0] < 0.5 - DOLFIN_EPS
            )

    left = LeftBoundary()
    right = RightBoundary()
    lower = LowerBoundary()
    upper = UpperBoundary()
    # restricted_upper = RestrictedUpperBoundary()

    V = FunctionSpace(mesh, "CG", 2)

    # Be particularly careful with the boundary conditions.
    # The main problem here is that the PPE system is consistent if and only
    # if
    #
    #     \int_\Omega div(u) = \int_\Gamma n.u = 0.
    #
    # This is exactly and even pointwise fulfilled for the continuous problem.
    # In the discrete case, we can have to make sure that n.u is 0 all along
    # the boundary.
    # In the lid-driven cavity problem, of particular interest are the corner
    # points at the lid. One has to assert that the z-component of u is 0 all
    # across the lid, and the x-component of u is 0 everywhere but the lid.
    # Since u is L2-"continuous", the lid condition on u_x must not be enforced
    # in the corner points. The u_y component must be enforced all over the
    # lid, including the end points.
    W = MixedFunctionSpace([V, V])
    u_bcs = [
        DirichletBC(W, (0.0, 0.0), left),
        DirichletBC(W, (0.0, 0.0), right),
        # DirichletBC(W.sub(0), Expression('x[0]'), restricted_upper),
        DirichletBC(W, (0.0, 0.0), lower),
        DirichletBC(W.sub(0), Constant("1.0"), upper),
        DirichletBC(W.sub(1), 0.0, upper),
        # DirichletBC(W.sub(0), Constant('-1.0'), lower),
        # DirichletBC(W.sub(1), 0.0, lower),
        # DirichletBC(W.sub(1), Constant('1.0'), left),
        # DirichletBC(W.sub(0), 0.0, left),
        # DirichletBC(W.sub(1), Constant('-1.0'), right),
        # DirichletBC(W.sub(0), 0.0, right),
    ]
    P = FunctionSpace(mesh, "CG", 1)
    p_bcs = []
    return mesh, W, P, u_bcs, p_bcs


def peter():
    base = "../meshes/2d/peter"
    # base = '../meshes/2d/peter-fine'
    mesh = Mesh(base + ".xml")

    subdomains = MeshFunction("size_t", mesh, base + "_physical_region.xml")

    workpiece_index = 3
    subdomain_materials = {1: "SiC", 2: "carbon steel", 3: "GaAs (liquid)"}

    submesh_workpiece = SubMesh(mesh, subdomains, workpiece_index)
    W = VectorFunctionSpace(submesh_workpiece, "CG", 2)
    Q = FunctionSpace(submesh_workpiece, "CG", 2)

    # Define melt boundaries.
    class LeftBoundary(SubDomain):
        def inside(self, x, on_boundary):
            # Be especially careful in the corners when defining the r=0 axis:
            # For the PPE to be consistent, we need to have n.u=0 *everywhere*
            # on the non-(r=0) boundaries.
            return (
                on_boundary
                and x[0] < GMSH_EPS
                and 0.005 + GMSH_EPS < x[1]
                and x[1] < 0.050 - GMSH_EPS
            )

    class SurfaceBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return (
                on_boundary
                and 0.055 - GMSH_EPS < x[1]
                and 0.030 - GMSH_EPS < x[0]
                and x[0] < 0.075 + GMSH_EPS
            )

    class StampBoundary(SubDomain):
        def inside(self, x, on_boundary):
            # Well well. If we don't include the endpoints here, the PPE turns
            # out inconsistent. This is weird since the endpoints should be
            # picked up by RightBoundary and StampBoundary.
            return (
                on_boundary
                and 0.050 - GMSH_EPS < x[1]
                and x[1] < 0.055 + GMSH_EPS
                and x[0] < 0.030 + GMSH_EPS
            )

    class LowerBoundary(SubDomain):
        def inside(self, x, on_boundary):
            # Danger, danger.
            # If the rightmost point (0.075, 0.010) is excluded, the PPE will
            # end up inconsistent.
            # This is actually weird since it should be picked up by
            # RightBoundary.
            return on_boundary and (
                x[1] < 0.005 + GMSH_EPS
                or (x[0] > 0.070 - GMSH_EPS and x[1] < 0.010 + GMSH_EPS)
            )

    class RightBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] > 0.075 - GMSH_EPS

    left_boundary = LeftBoundary()
    lower_boundary = LowerBoundary()
    right_boundary = RightBoundary()
    surface_boundary = SurfaceBoundary()
    stamp_boundary = StampBoundary()

    # Define workpiece boundaries.
    wp_boundaries = MeshFunction(
        "size_t", submesh_workpiece, self.submesh_workpiece.topology().dim() - 1
    )
    wp_boundaries.set_all(0)
    left_boundary.mark(wp_boundaries, 1)
    lower_boundary.mark(wp_boundaries, 2)
    right_boundary.mark(wp_boundaries, 3)
    surface_boundary.mark(wp_boundaries, 4)
    stamp_boundary.mark(wp_boundaries, 5)

    # For local use only:
    wp_boundary_indices = {"left": 1, "lower": 2, "right": 3, "surface": 4, "stamp": 5}

    # Boundary conditions for the velocity.
    u_bcs = [
        DirichletBC(W, (0.0, 0.0), stamp_boundary),
        DirichletBC(W, (0.0, 0.0), right_boundary),
        DirichletBC(W, (0.0, 0.0), lower_boundary),
        DirichletBC(W.sub(0), 0.0, left_boundary),
        DirichletBC(W.sub(1), 0.0, surface_boundary),
    ]
    p_bcs = []

    # Boundary conditions for the heat equation.
    # Dirichlet
    theta_bcs_d = []
    # Neumann
    theta_bcs_n = {}
    # Robin, i.e.,
    #
    #    -dtheta/dn = alpha (theta - theta0)
    #
    # (with alpha>0 to preserve coercivity of the scheme).
    #
    theta_bcs_r = {
        wp_boundary_indices["stamp"]: (100.0, 1500.0),
        wp_boundary_indices["surface"]: (100.0, 1550.0),
        wp_boundary_indices["right"]: (300.0, Expression("200*x[0] + 1600", degree=1)),
        wp_boundary_indices["lower"]: (
            300.0,
            Expression("-1200*x[1] + 1614", degree=1),
        ),
    }
    return (
        mesh,
        subdomains,
        subdomain_materials,
        workpiece_index,
        wp_boundaries,
        W,
        u_bcs,
        p_bcs,
        Q,
        theta_bcs_d,
        theta_bcs_n,
        theta_bcs_r,
    )


def pot():
    # Define mesh and boundaries.
    mesh = RectangleMesh(0.0, 0.0, 0.4, 0.1, 120, 30, "left/right")

    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)

    class LeftBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] < GMSH_EPS

    left_boundary = LeftBoundary()

    class RightBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] > 0.4 - GMSH_EPS

    right_boundary = RightBoundary()

    class LowerBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] < GMSH_EPS

    lower_boundary = LowerBoundary()

    class UpperBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] > 0.1 - GMSH_EPS

    upper_boundary = UpperBoundary()

    # Boundary conditions for the velocity.
    u_bcs = [
        DirichletBC(V, (0.0, 0.0), lower_boundary),
        DirichletBC(V.sub(1), 0.0, upper_boundary),
        DirichletBC(V.sub(0), 0.0, right_boundary),
        DirichletBC(V.sub(0), 0.0, left_boundary),
    ]
    p_bcs = []
    return mesh, V, Q, u_bcs, p_bcs, [lower_boundary], [right_boundary, left_boundary]


def cavity():
    # Define mesh and boundaries.
    mesh = UnitSquareMesh(32, 32, "left/right")

    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)

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
            return on_boundary and x[1] > 1.0 - GMSH_EPS

    upper_boundary = UpperBoundary()

    # Boundary conditions for the velocity.
    u_bcs = [
        DirichletBC(V, (0.0, 0.0), lower_boundary),
        DirichletBC(V, (0.0, 0.0), upper_boundary),
        DirichletBC(V, (0.0, 0.0), right_boundary),
        DirichletBC(V, (0.0, 0.0), left_boundary),
    ]
    p_bcs = []
    return (
        mesh,
        V,
        Q,
        u_bcs,
        p_bcs,
        [right_boundary],
        [upper_boundary, left_boundary, lower_boundary],
    )


def coil():
    mesh = Mesh("../meshes/2d/coil-in-box.xml")
    V = VectorFunctionSpace(mesh, "CG", 2)
    P = FunctionSpace(mesh, "CG", 1)
    Q = FunctionSpace(mesh, "CG", 1)

    # Define mesh and boundaries.
    class LeftBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] < GMSH_EPS

    left_boundary = LeftBoundary()

    class RightBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] > 0.6 - GMSH_EPS

    right_boundary = RightBoundary()

    class LowerBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] < GMSH_EPS

    lower_boundary = LowerBoundary()

    class UpperBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] > 2.0 - GMSH_EPS

    upper_boundary = UpperBoundary()

    class CoilBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return (
                on_boundary
                and x[0] > GMSH_EPS
                and x[0] < 0.6 - GMSH_EPS
                and x[1] > GMSH_EPS
                and x[1] < 2.0 - GMSH_EPS
            )

    coil_boundary = CoilBoundary()

    # Boundary conditions for the velocity.
    u_bcs = DirichletBC(V, (0.0, 0.0), "on_boundary")
    p_bcs = []
    # u_bcs = [DirichletBC(V.sub(0), 0.0, right_boundary),
    #          DirichletBC(V.sub(0), 0.0, left_boundary),
    #          DirichletBC(V.sub(1), 0.0, lower_boundary),
    #          DirichletBC(V.sub(1), 0.0, upper_boundary),
    #          DirichletBC(V, (0.0, 0.0), coil_boundary)
    #          ]
    # p_bcs = []
    # u_bcs = [DirichletBC(V, (0.0, 0.0), right_boundary),
    #          DirichletBC(V, (0.0, 0.0), left_boundary),
    #          DirichletBC(V, (0.0, 0.0), lower_boundary),
    #          DirichletBC(V, (0.0, 0.0), coil_boundary)
    #          ]
    # p_bcs = DirichletBC(Q, 0.0, upper_boundary)
    return (
        mesh,
        V,
        P,
        Q,
        u_bcs,
        p_bcs,
        [coil_boundary],
        [left_boundary, right_boundary, lower_boundary, upper_boundary],
    )


def square_with_obstacle():
    # Create classes for defining parts of the boundaries and the interior
    # of the domain
    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0.0)

    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 1.0)

    class Bottom(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 0.0)

    class Top(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 1.0)

    class Obstacle(SubDomain):
        def inside(self, x, on_boundary):
            return between(x[1], (0.5, 0.7)) and between(x[0], (0.2, 1.0))

    # Initialize sub-domain instances
    left = Left()
    top = Top()
    right = Right()
    bottom = Bottom()
    obstacle = Obstacle()

    # Define mesh
    mesh = UnitSquareMesh(100, 100, "crossed")

    # Initialize mesh function for interior domains
    domains = CellFunction("size_t", mesh)
    domains.set_all(0)
    obstacle.mark(domains, 1)

    # Initialize mesh function for boundary domains
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    left.mark(boundaries, 1)
    top.mark(boundaries, 2)
    right.mark(boundaries, 3)
    bottom.mark(boundaries, 4)

    boundary_indices = {"left": 1, "top": 2, "right": 3, "bottom": 4}
    f = Constant(0.0)
    theta0 = Constant(293.0)
    return mesh, f, boundaries, boundary_indices, theta0


def coil_in_box():
    mesh = Mesh("../meshes/2d/coil-in-box.xml")

    f = Constant(0.0)

    # Define mesh and boundaries.
    class LeftBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] < GMSH_EPS

    left_boundary = LeftBoundary()

    class RightBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] > 2.5 - GMSH_EPS

    right_boundary = RightBoundary()

    class LowerBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] < GMSH_EPS

    lower_boundary = LowerBoundary()

    class UpperBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] > 0.4 - GMSH_EPS

    upper_boundary = UpperBoundary()

    class CoilBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return (
                on_boundary
                and x[0] > GMSH_EPS
                and x[0] < 2.5 - GMSH_EPS
                and x[1] > GMSH_EPS
                and x[1] < 0.4 - GMSH_EPS
            )

    coil_boundary = CoilBoundary()

    # heater_temp = 380.0
    # room_temp = 293.0
    # bcs = [(coil_boundary, heater_temp),
    #        (left_boundary, room_temp),
    #        (right_boundary, room_temp),
    #        (upper_boundary, room_temp),
    #        (lower_boundary, room_temp)
    #        ]

    boundaries = {}
    boundaries["left"] = left_boundary
    boundaries["right"] = right_boundary
    boundaries["upper"] = upper_boundary
    boundaries["lower"] = lower_boundary
    boundaries["coil"] = coil_boundary

    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    left_boundary.mark(boundaries, 1)
    right_boundary.mark(boundaries, 2)
    upper_boundary.mark(boundaries, 3)
    lower_boundary.mark(boundaries, 4)
    coil_boundary.mark(boundaries, 5)

    boundary_indices = {"left": 1, "right": 2, "top": 3, "bottom": 4, "coil": 5}
    theta0 = Constant(293.0)
    return mesh, f, boundaries, boundary_indices, theta0


def karman():
    # mesh = Mesh('../meshes/2d/karman.xml')
    mesh = Mesh("../meshes/2d/karman-coarse2.xml")

    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)

    # Define mesh and boundaries.
    class LeftBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] < DOLFIN_EPS

    left_boundary = LeftBoundary()

    class RightBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] > 2.5 - DOLFIN_EPS

    class LowerBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] < DOLFIN_EPS

    lower_boundary = LowerBoundary()

    class UpperBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] > 0.4 - DOLFIN_EPS

    upper_boundary = UpperBoundary()

    class ObstacleBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return (
                on_boundary
                and x[0] > DOLFIN_EPS
                and x[0] < 2.5 - DOLFIN_EPS
                and x[1] > DOLFIN_EPS
                and x[1] < 0.4 - DOLFIN_EPS
            )

    obstacle_boundary = ObstacleBoundary()

    # Boundary conditions for the velocity.
    # Proper inflow and outflow conditions are a matter of voodoo. See for
    # example Gresho/Sani, or
    #
    #     Boundary conditions for open boundaries
    #     for the incompressible Navier-Stokes equation;
    #     B.C.V. Johansson;
    #     J. Comp. Phys. 105, 233-251 (1993).
    #
    # The latter in particularly suggest for the inflow:
    #
    #     u = u0,
    #     d^r v / dx^r = v_r,
    #     div(u) = 0,
    #
    # where u and v are the velocities in normal and tangential directions,
    # respectively, and r\in{0,1,2}. The setting r=0 essentially means to set
    # (u,v) statically at the left boundary, r=1 means to set u and control
    # dv/dn, which is what we do here (namely implicitly by dv/dn=0).
    # At the outflow,
    #
    #     d^j u / dx^j = 0,
    #     d^q v / dx^q = 0,
    #     p = p0,
    #
    # is suggested with j=q+1. Choosing q=0, j=1 means setting the tangential
    # component of the outflow to 0, and letting the normal component du/dn=0
    # (again, this is achieved implicitly by the weak formulation).
    #
    inflow = Expression("100*x[1]*(0.4-x[1])", degree=2)
    u_bcs = [
        DirichletBC(V, (0.0, 0.0), upper_boundary),
        DirichletBC(V, (0.0, 0.0), lower_boundary),
        DirichletBC(V, (0.0, 0.0), obstacle_boundary),
        DirichletBC(V.sub(0), inflow, left_boundary),
        # DirichletBC(V.sub(1), 0.0, right_boundary),
    ]
    dudt_bcs = [
        DirichletBC(V, (0.0, 0.0), upper_boundary),
        DirichletBC(V, (0.0, 0.0), lower_boundary),
        DirichletBC(V, (0.0, 0.0), obstacle_boundary),
        DirichletBC(V.sub(0), 0.0, left_boundary),
        # DirichletBC(V.sub(1), 0.0, right_boundary),
    ]
    # If there is a penetration boundary (i.e., n.u!=0), then the pressure must
    # be set at the boundary to make sure that the Navier-Stokes problem
    # remains consistent.
    # It is not quite clear now where exactly to set the pressure to 0. Inlet,
    # outlet, some other place? The PPE system is consistent in all cases.
    # TODO find out more about it
    # p_bcs = [DirichletBC(Q, 0.0, right_boundary)]
    p_bcs = []
    return mesh, V, Q, u_bcs, dudt_bcs, p_bcs


def currentloop():
    base = "../meshes/2d/circle-in-halfcircle"
    mesh = Mesh(base + ".xml")
    subdomains = MeshFunction("size_t", mesh, base + "_physical_region.xml")
    subdomain_materials = {1: "copper", 2: "air"}
    coils = [{"rings": [1], "c_type": "voltage", "c_value": 230.0 * numpy.sqrt(2.0)}]
    wpi = None
    omega = 1.0e3
    # According to
    # <http://hyperphysics.phy-astr.gsu.edu/hbase/magnetic/curloo.html>,
    # the magnitude of the magnetic field is
    #
    #     |B| = mu0*I / (2*r0)
    #         = mu0*V / (2*r0*R).
    #
    # at the center of the loop. With a copper wire, this is
    #
    #     |B| = pi * 4e-7 * V / (2*r0*1.535356e-08)
    #         ~ 40.9 * V/r0.
    #         = 13.30350e3
    #
    return mesh, subdomains, coils, wpi, subdomain_materials, omega


def pons():
    base = "../meshes/2d/pons"
    mesh = Mesh(base + ".xml")
    subdomains = MeshFunction("size_t", mesh, base + "_physical_region.xml")
    subdomain_materials = {19: "graphite", 20: "SiC", 21: "air"}
    for k in range(1, 19, 2):
        subdomain_materials[k] = "air"
    for k in range(2, 19, 2):
        subdomain_materials[k] = "copper"
    coils = [
        {
            "rings": range(2, 19, 2),
            # 'rings': range(10,19,2),
            # 'rings': [10],
            "c_type": "voltage",
            "c_value": 230.0 * numpy.sqrt(2.0),
        }
    ]
    wpi = 20
    omega = 2 * pi * 10e3
    return mesh, subdomains, coils, wpi, subdomain_materials, omega


def generic():
    base = "../meshes/2d/circles2d-boundary"
    mesh = Mesh(base + ".xml")
    subdomains = MeshFunction("size_t", mesh, base + "_physical_region.xml")
    subdomain_materials = {1: "graphite", 2: "GaAs (liquid)", 3: "air"}
    coils = [
        {"rings": [1], "c_type": "voltage", "c_value": 20.0}  # 230.0*numpy.sqrt(2.0)
    ]
    wpi = 2
    # omega = 2 * pi * 10.0e3
    omega = 2 * pi * 300.0
    return mesh, subdomains, coils, wpi, subdomain_materials, omega
