"""
Half-circle in a box.
If rotated around x=0, this geometry corresponds to a ball in a cylindrical
tube.
"""
import pygmsh


def _define():
    geom = pygmsh.built_in.Geometry()

    z = 0
    lcar = 4.0e-2

    xmin = 0.0
    xmax = 1.0
    ymin = 0.0
    ymax = 5.0
    ball_radius = 0.5
    ball_y = 1.5

    # Define the points for the double rectangle.
    points = [
        geom.add_point([xmin, ymin, z], lcar),
        geom.add_point([xmax, ymin, z], lcar),
        geom.add_point([xmax, ymax, z], lcar),
        geom.add_point([xmin, ymax, z], lcar),
        #
        geom.add_point([xmin, ball_y + ball_radius, z], lcar),
        geom.add_point([xmin, ball_y, z], lcar),
        geom.add_point([xmin + ball_radius, ball_y, z], lcar),
        geom.add_point([xmin, ball_y - ball_radius, z], lcar),
    ]
    lines = [
        geom.add_line(points[0], points[1]),
        geom.add_line(points[1], points[2]),
        geom.add_line(points[2], points[3]),
        geom.add_line(points[3], points[4]),
        geom.add_circle_arc(points[4], points[5], points[6]),
        geom.add_circle_arc(points[6], points[5], points[7]),
        geom.add_line(points[7], points[0]),
    ]

    ll = geom.add_line_loop(lines)
    geom.add_plane_surface(ll)

    return geom


def generate():
    return pygmsh.generate_mesh(_define())


if __name__ == "__main__":
    import meshio

    points, cells, point_data, cell_data, _ = generate()
    meshio.write_points_cells(
        "ball-in-tube.vtu", points, cells, point_data=point_data, cell_data=cell_data
    )
