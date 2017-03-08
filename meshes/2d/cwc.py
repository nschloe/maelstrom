# import numpy
import pygmsh


def generate():
    geom = pygmsh.Geometry()

    z = 0.0

    lcar_base = 1.0e-2
    lcar_far = 10*lcar_base
    lcar_coil = lcar_base
    lcar_gas = 4 * lcar_base
    lcar_crucible = 1.0e-1

    # # omega: current frequency (for boundary layer width)
    # omega = 2 * numpy.pi * 300.0
    # # omega = 2 * numpy.pi * 10.0e3;
    # mu0 = numpy.pi * 4e-7

    # symmetry axis
    x0 = 0.0

    # upper end of the elliptic bend
    y0 = 0.38826701

    # Crucible
    crucible_wall_width = 0.002

    # The assumption that the outer outline of the crucible is defined by the
    # ellipse
    #
    #     ((x-x0)/(crucible_radius-x0))^2 + ((y-y0)/(crucible_bottom-y0))^2 = 1
    #
    # leads to a geometry which isn't true to the original from [1].
    # Hence, build some hand-crafted B-splines.
    #
    crucible_radius = 0.076 + crucible_wall_width
    # upper end of the elliptic bend
    y0 = 0.38826701
    # lower end
    crucible_bottom = 0.365917 - crucible_wall_width
    # upper crucible end
    crucible_top = 0.500
    # liquid surface level
    melt_level = 0.411
    # shielding gas surface level
    gas_level = 0.426

    # Outer ellipse.
    # B-Spline.
    tp1 = geom.add_point([x0, crucible_bottom, z], lcar_crucible)
    tp2 = geom.add_point([crucible_radius, y0+0.005, z], lcar_crucible)
    tp2_spline0 = geom.add_point([0.02, crucible_bottom, z], lcar_crucible)
    tp2_spline1 = geom.add_point(
            [crucible_radius, crucible_bottom+0.005, z],
            lcar_crucible
            )
    tc1 = geom.add_bspline([tp1, tp2_spline0, tp2_spline1, tp2])

    # Inner ellipse
    # Linear line segments (data from Tecplot)
    inner_crucible_start = geom.add_point([0.0, 0.365917, z], lcar_crucible)
    tp101 = geom.add_point([0.00642931, 0.365989, z], lcar_crucible)
    tp102 = geom.add_point([0.0128557, 0.36619601, z], lcar_crucible)
    tp103 = geom.add_point([0.0192762, 0.366539, z], lcar_crucible)
    tp104 = geom.add_point([0.0256882, 0.367017, z], lcar_crucible)
    tp105 = geom.add_point([0.0320886, 0.36762899, z], lcar_crucible)
    tp106 = geom.add_point([0.0384747, 0.368377, z], lcar_crucible)
    tp107 = geom.add_point([0.0448436, 0.369259, z], lcar_crucible)
    tp108 = geom.add_point([0.0511925, 0.370276, z], lcar_crucible)
    tp109 = geom.add_point([0.0633982, 0.37261799, z], lcar_crucible)
    tp110 = geom.add_point([0.0681625, 0.374695, z], lcar_crucible)
    tp111 = geom.add_point([0.0736436, 0.37980801, z], lcar_crucible)

    # The Tecplot data seems inaccurate here in that the x-value of this point
    # and the top-right point of the melt do not coincide. Fix this manually.
    inner_crucible_end = geom.add_point([0.0760625, y0, z], lcar_crucible)
    # inner_crucible_end = geom.add_point([0.076, y0, z], lcar_crucible)

    cl100 = geom.add_line(inner_crucible_start, tp101)
    cl101 = geom.add_line(tp101, tp102)
    cl102 = geom.add_line(tp102, tp103)
    cl103 = geom.add_line(tp103, tp104)
    cl104 = geom.add_line(tp104, tp105)
    cl105 = geom.add_line(tp105, tp106)
    cl106 = geom.add_line(tp106, tp107)
    cl107 = geom.add_line(tp107, tp108)
    cl108 = geom.add_line(tp108, tp109)
    cl109 = geom.add_line(tp109, tp110)
    cl110 = geom.add_line(tp110, tp111)
    cl111 = geom.add_line(tp111, inner_crucible_end)

    # Extend and close.
    tp5 = geom.add_point([crucible_radius, crucible_top, z], lcar_crucible)
    tp6 = geom.add_point(
        [crucible_radius - crucible_wall_width, crucible_top, z],
        lcar_crucible
        )
    tp7 = geom.add_point(
            [crucible_radius - crucible_wall_width, melt_level, z],
            lcar_crucible
            )
    tp71 = geom.add_point(
            [crucible_radius - crucible_wall_width, gas_level, z],
            lcar_crucible
            )

    cl1 = geom.add_line(tp2, tp5)
    cl2 = geom.add_line(tp5, tp6)
    cl3 = geom.add_line(tp6, tp71)
    cl31 = geom.add_line(tp71, tp7)
    cl4 = geom.add_line(tp7, inner_crucible_end)
    cl5 = geom.add_line(tp1, inner_crucible_start)

    # Define crucible surface.
    ll1 = geom.add_line_loop([
            tc1, cl1, cl2, cl3, cl31, cl4,
            # '-' + tc2,
            '-' + cl111, '-' + cl110, '-' + cl109, '-' + cl108,
            '-' + cl107, '-' + cl106, '-' + cl105,
            '-' + cl104, '-' + cl103,
            '-' + cl102, '-' + cl101,
            '-' + cl100,
            '-' + cl5
            ])
    surf = geom.add_plane_surface(ll1)
    geom.add_physical_surface(surf, 'crucible')

    # gas above the melt
    x2 = 0.038
    tp8 = geom.add_point([x2, melt_level, z], lcar_gas)
    tp9 = geom.add_point([x2, gas_level, z], lcar_gas)
    #
    cl6 = geom.add_line(tp71, tp9)
    cl7 = geom.add_line(tp9, tp8)
    cl8 = geom.add_line(tp8, tp7)
    ll2 = geom.add_line_loop([cl6, cl7, cl8, '-' + cl31])
    #
    boron = geom.add_plane_surface(ll2)
    geom.add_physical_surface(boron, 'gas')

    # the crystal
    tp10 = geom.add_point([x0, melt_level, z], lcar_gas)
    tp11 = geom.add_point([x0, crucible_top, z], lcar_gas)
    tp11a = geom.add_point([x0+0.01, crucible_top, z], lcar_gas)
    tp11b = geom.add_point([x0+0.01, gas_level+0.025, z], lcar_gas)
    #
    cl9 = geom.add_line(tp9, tp11b)
    cl9a = geom.add_line(tp11b, tp11a)
    cl9b = geom.add_line(tp11a, tp11)
    cl10 = geom.add_line(tp11, tp10)
    cl11 = geom.add_line(tp10, tp8)
    #
    ll3 = geom.add_line_loop([cl9, cl9a, cl9b, cl10, cl11, '-' + cl7])
    #
    surf = geom.add_plane_surface(ll3)
    geom.add_physical_surface(surf, 'crystal')

    # the melt
    cl12 = geom.add_line(tp10, inner_crucible_start)
    ll4 = geom.add_line_loop([
        cl12,
        # tc2,
        cl100, cl101, cl102, cl103, cl104, cl105, cl106, cl107,
        cl108, cl109, cl110, cl111,
        '-' + cl4, '-' + cl8, '-' + cl11
        ])
    surf = geom.add_plane_surface(ll4)
    geom.add_physical_surface(surf, 'melt')

    line_loops = []
    # Coils to the right.
    xmin = 0.092
    xmax = xmin + 0.015
    ymin = 0.266
    ymax = ymin + 0.010
    for k in range(15):
        ymin += 0.0132
        ymax += 0.0132
        surf, ll = geom.add_rectangle(xmin, xmax, ymin, ymax, z, lcar_coil)
        line_loops.append(ll)
        geom.add_physical_surface(surf, 'coil right %d' % k)

    # coils at the bottom
    xmin = 0.031
    xmax = xmin + 0.005
    ymin = 0.33
    ymax = ymin + 0.024
    for k in range(7):
        surf, ll = geom.add_rectangle(xmin, xmax, ymin, ymax, z, lcar_coil)
        line_loops.append(ll)
        geom.add_physical_surface(surf, 'coil bottom %d' % k)
        xmin += 0.008
        xmax += 0.008

    # Hold-all domain.
    r = 1.0
    # Draw first quarter-circle.
    tp20 = geom.add_point([x0, y0, z], lcar_far)
    tp21 = geom.add_point([x0, y0-r, z], lcar_far)
    tp22 = geom.add_point([x0+r, y0, z], lcar_far)
    tp23 = geom.add_point([x0, y0+r, z], lcar_far)

    # Build circle from arcs.
    cc1 = geom.add_circle_sector([tp21, tp20, tp22])
    cc2 = geom.add_circle_sector([tp22, tp20, tp23])

    # # Connecting lines.
    # cc00 = geom.add_line(tp23, tp21)
    cl20 = geom.add_line(tp1, tp21)
    cl24 = geom.add_line(tp23, tp11)

    ll = geom.add_line_loop([
        cl20,
        cc1,
        cc2,
        cl24,
        '-' + cl9b,
        '-' + cl9a,
        '-' + cl9,
        '-' + cl6,
        '-' + cl3,
        '-' + cl2,
        '-' + cl1,
        '-' + tc1,
        ])

    # prepend the line loop
    line_loops.insert(0, ll)

    pl = geom.add_plane_surface(line_loops)
    geom.add_physical_surface(pl, 'air')

    # background_field_id = newf;
    # Field[background_field_id] = Min;
    # Field[background_field_id].FieldsList = {thefields[]};
    # Background Field = background_field_id;

    # Mesh.CharacteristicLengthExtendFromBoundary = 0;

    # Decrease the precision of 1D integration. If set to default, mesh
    # generation may take very long.
    # Mesh.LcIntegrationPrecision = 1.0e-3;

    return geom


if __name__ == '__main__':
    import meshio
    points, cells = pygmsh.generate_mesh(generate())
    meshio.write('out.vtu', points, cells)
