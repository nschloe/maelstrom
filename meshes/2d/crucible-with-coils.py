# -*- coding: utf-8 -*-
'''
Creates a crucible similar to what is described in

  Mathematical Modeling of Czochralski Type Growth Processes for
  Semiconductor Bulk Single Crystals,
  http://dx.doi.org/10.1007/s00032-012-0184-9
'''
import python4gmsh as p4g
from numpy import pi, sqrt, exp, log


def _main():
    args = _parse_args()

    z = 0.0

    lcar_base = 1.0e-1
    #lcar_far = 10 * lcar_base
    lcar_coil = lcar_base
    lcar_gas = 4 * lcar_base
    lcar_crucible = 1.0e-1

    # omega: current frequency (for boundary layer width)
    omega = 2 * pi * 300.0
    #omega = 2 * pi * 10.0e3
    mu0 = pi * 4.0e-7

    # symmetry axis
    x0 = 0.0

    # Construct the crucible.
    crucible_wall_width = 0.002

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

    #p1 = p4g.Point([x0, y0, z], lcar_crucible)
    # -------------------------------------------------------------------------
    # The assumption that the outer outline of the crucible is defined by the
    # ellipse
    #
    #     ((x-x0)/(crucible_radius-x0))^2 + ((y-y0)/(crucible_bottom-y0))^2 = 1
    #
    # leads to a geometry which isn't true to the original from [1].
    # Hence, build some hand-crafted B-splines.
    # Outer boundary: B-spline
    tp1 = p4g.Point([x0, crucible_bottom, z], lcar_crucible)
    tp2 = p4g.Point([crucible_radius, y0+0.005, z], lcar_crucible)
    tp2_spline0 = p4g.Point([0.02, crucible_bottom, z], lcar_crucible)
    tp2_spline1 = p4g.Point([crucible_radius, crucible_bottom+0.005, z],
                            lcar_crucible)
    # Ellipse(tc1) = {tp1, center, tp2, tp2}
    tc1 = p4g.BSpline([tp1, tp2_spline0, tp2_spline1, tp2])
    # -------------------------------------------------------------------------
    # Inner boundary: line segments (data from Tecplot)
    inner_crucible_start = p4g.Point([0.0, 0.365917, z], lcar_crucible)
    tp101 = p4g.Point([0.00642931, 0.365989, z], lcar_crucible)
    tp102 = p4g.Point([0.0128557, 0.36619601, z], lcar_crucible)
    tp103 = p4g.Point([0.0192762, 0.366539, z], lcar_crucible)
    tp104 = p4g.Point([0.0256882, 0.367017, z], lcar_crucible)
    tp105 = p4g.Point([0.0320886, 0.36762899, z], lcar_crucible)
    tp106 = p4g.Point([0.0384747, 0.368377, z], lcar_crucible)
    tp107 = p4g.Point([0.0448436, 0.369259, z], lcar_crucible)
    tp108 = p4g.Point([0.0511925, 0.370276, z], lcar_crucible)
    tp109 = p4g.Point([0.0633982, 0.37261799, z], lcar_crucible)
    tp110 = p4g.Point([0.0681625, 0.374695, z], lcar_crucible)
    tp111 = p4g.Point([0.0736436, 0.37980801, z], lcar_crucible)
    # The Tecplot data seems inaccurate here in that the x-value of this point
    # and the top-right point of the melt do not coincide. Fix this manually.
    inner_crucible_end = p4g.Point([0.0760625, y0, z], lcar_crucible)
    # inner_crucible_end = p4g.Point([0.076, y0, z], lcar_crucible)

    cl100 = p4g.Line(inner_crucible_start, tp101)
    cl101 = p4g.Line(tp101, tp102)
    cl102 = p4g.Line(tp102, tp103)
    cl103 = p4g.Line(tp103, tp104)
    cl104 = p4g.Line(tp104, tp105)
    cl105 = p4g.Line(tp105, tp106)
    cl106 = p4g.Line(tp106, tp107)
    cl107 = p4g.Line(tp107, tp108)
    cl108 = p4g.Line(tp108, tp109)
    cl109 = p4g.Line(tp109, tp110)
    cl110 = p4g.Line(tp110, tp111)
    cl111 = p4g.Line(tp111, inner_crucible_end)

    # Extend and close
    tp5 = p4g.Point([crucible_radius, crucible_top, z], lcar_crucible)
    tp6 = p4g.Point([crucible_radius - crucible_wall_width, crucible_top, z],
                    lcar_crucible)
    tp7 = p4g.Point([crucible_radius - crucible_wall_width, melt_level, z],
                    lcar_crucible)
    tp71 = p4g.Point([crucible_radius - crucible_wall_width, gas_level, z],
                     lcar_crucible)
    cl1 = p4g.Line(tp2, tp5)
    cl2 = p4g.Line(tp5, tp6)
    cl3 = p4g.Line(tp6, tp71)
    cl31 = p4g.Line(tp71, tp7)
    cl4 = p4g.Line(tp7, inner_crucible_end)
    cl5 = p4g.Line(tp1, inner_crucible_start)
    ll1 = p4g.LineLoop([tc1, cl1, cl2, cl3, cl31, cl4,
                        #'-'+tc2,
                        '-'+cl111, '-'+cl110, '-'+cl109, '-'+cl108, '-'+cl107,
                        '-'+cl106, '-'+cl105,
                        '-'+cl104, '-'+cl103, '-'+cl102, '-'+cl101, '-'+cl100,
                        '-'+cl5]
                       )
    crucible = p4g.PlaneSurface(ll1)
    p4g.PhysicalSurface(crucible, 'crucible')
    # -------------------------------------------------------------------------
    # The gas
    x2 = 0.038
    tp8 = p4g.Point([x2, melt_level, z], lcar_gas)
    tp9 = p4g.Point([x2, gas_level, z], lcar_gas)
    cl6 = p4g.Line(tp71, tp9)
    cl7 = p4g.Line(tp9, tp8)
    cl8 = p4g.Line(tp8, tp7)
    ll2 = p4g.LineLoop([cl6, cl7, cl8, '-'+cl31])
    boron = p4g.PlaneSurface(ll2)
    p4g.PhysicalSurface(boron, 'gas')
    # -------------------------------------------------------------------------
    # The crystal
    tp10 = p4g.Point([x0, melt_level, z], lcar_gas)
    tp11 = p4g.Point([x0, crucible_top, z], lcar_gas)
    tp11a = p4g.Point([x0+0.01, crucible_top, z], lcar_gas)
    tp11b = p4g.Point([x0+0.01, gas_level+0.025, z], lcar_gas)
    cl9 = p4g.Line(tp9, tp11b)
    cl9a = p4g.Line(tp11b, tp11a)
    cl9b = p4g.Line(tp11a, tp11)
    cl10 = p4g.Line(tp11, tp10)
    cl11 = p4g.Line(tp10, tp8)
    ll3 = p4g.LineLoop([cl9, cl9a, cl9b, cl10, cl11, '-'+cl7])
    crystal = p4g.PlaneSurface(ll3)
    p4g.PhysicalSurface(crystal, 'crystal')
    # -------------------------------------------------------------------------
    # The melt.
    cl12 = p4g.Line(tp10, inner_crucible_start)
    ll4 = p4g.LineLoop([cl12,
                        #tc2,
                        cl100, cl101, cl102, cl103, cl104, cl105, cl106, cl107,
                        cl108, cl109, cl110, cl111,
                        '-'+cl4, '-'+cl8, '-'+cl11]
                       )
    melt = p4g.PlaneSurface(ll4)
    p4g.PhysicalSurface(melt, 'melt')
    # -------------------------------------------------------------------------
    # Refinement around the boundaries.
    # The boundary width of the melt is determined by diffusion coefficients of
    # three PDEs:
    #   * Maxwell
    #   * Navier-Stokes
    #   * heat equation.
    #
    T = 1511.0
    #rho_gaas = 7.33e3 - 1.07 * T
    #cp_gaas = 0.434e3
    reynolds = 6.197e3
    prandtl = 0.068
    mu_gaas = mu0*(1.0 - 0.85e-10)
    sigma_gaas = 7.9e5
    #kappa_gaas = 17.8
    # Boundary layer widths
    w_maxwell = sqrt(2.0/(mu_gaas*sigma_gaas*omega))
    w_heat = sqrt(1.0/(reynolds*prandtl))
    w_navier = sqrt(1.0/reynolds)
    print('Melt boundary layer widths:')
    print('    Maxwell: %g' % w_maxwell)
    print('    heat eq: %g' % w_heat)
    print('    Navier:  %g' % w_navier)

    w_b0 = min(w_maxwell, w_heat, w_navier)

    # Fit 2*k elements normal into the boundary layer.
    k = 10
    lcar_b = min(lcar_crucible, w_b0/k)

    #boundary_layer = p4g.BoundaryLayer(
    #    edges_list=[cl100, cl101, cl102, cl103,
    #                cl104, cl105, cl106, cl107,
    #                cl108, cl109, cl110, cl111,
    #                #tc2,
    #                cl4, cl8, cl11],
    #    anisomax=100.0,
    #    hfar=lcar_far,
    #    hwall_n=lcar_b,
    #    hwall_t=lcar_b,
    #    thickness=w_b0
    #    )
    #s += 1
    #thefields[s] = b_id
    # -------------------------------------------------------------------------
    # Coils.
    # For layer-adapted meshes for reaction-diffusion problems, check out
    #
    # [1] Layer-adapted meshes for reaction-convection-diffusion problems
    #     T. Linß
    #     <https://www.springer.com/mathematics/computational+science+%26+engineering/book/978-3-642-05133-3>.
    #
    # The PDE for the potential \phi is of reaction-diffusion type
    #
    #     -eps^2 \Delta u + u = f
    #
    # with singularity parameter
    #
    #     eps^2 = 1 / (mu*sigma*omega)
    #
    # mu: magnetic permeability
    mu_graphite = mu0 * 0.999984
    # sigma: electrical conductivity
    T = 1511.0
    sigma_graphite = 1.0e6 / (28.9 - 18.8 * exp(-(log(T/1023.0)/2.37)**2))

    # It exhibits layers where \phi behaves like exp(-x/eps). This also
    # corresponds with the skin effect
    # (https://en.wikipedia.org/wiki/Skin_effect) layer width
    #
    #     \delta = \sqrt{2/(mu*sigma*omega)}.
    #
    w_b0 = sqrt(2.0/(mu_graphite*sigma_graphite*omega))
    # Fit 2*k elements normal into the boundary layer.
    k = 50
    lcar_b = min(lcar_coil, w_b0/k)

    print("lcar_coil: %g" % lcar_coil)
    print("lcar boundary: %g" % lcar_b)
    print("Coil boundary layer width: %f" % w_b0)

    # Coils to the right of the crucible
    xmin = 0.092
    xmax = xmin + 0.015
    ymin = 0.266
    ymax = ymin + 0.010
    for k in range(15):
        p4g.add_rectangle(xmin, xmax, ymin, ymax, z, lcar_coil)
        ymin += 0.0132
        ymax += 0.0132

    # Coils at the bottom of the crucible
    xmin = 0.031
    xmax = xmin + 0.005
    ymin = 0.33
    ymax = ymin + 0.024
    for k in range(7):
        p4g.add_rectangle(xmin, xmax, ymin, ymax, z, lcar_coil)
        xmin += 0.008
        xmax += 0.008
    # -------------------------------------------------------------------------
    ## Hold-all domain.
    #r = 1.0
    ## Draw first quarter-circle.
    #tp20 = p4g.Point([x0, y0, z], lcar_far)
    #tp21 = p4g.Point([x0, y0-r, z], lcar_far)
    #tp22 = p4g.Point([x0+r, y0, z], lcar_far)
    #tp23 = p4g.Point([x0, y0+r, z], lcar_far)

    ## Build circle from arcs.
    #cc1 = p4g.Circle([tp21, tp20, tp22])
    #cc2 = p4g.Circle([tp22, tp20, tp23])
    ## Connecting lines.
    #cl20 = p4g.Line(tp1, tp21)
    #cl24 = p4g.Line(tp23, tp11)

    #t = 0
    #theloops[t] = newc
    #Line Loop(theloops[t]) = {
    #    cl20, cc1, cc2, cl24, -cl9b, -cl9a, -cl9, -cl6, -cl3, -cl2, -cl1, -tc1
    #    }

    ## Rectangle.
    #yl = 0.20
    #yu = 0.60
    #xu = 0.25
    #
    #// outer corners
    #tp20 = newp
    #Point(tp20) = {x0, yl, z, lcar_far}
    #tp21 = newp
    #Point(tp21) = {xu, yl, z, lcar_far}
    #tp22 = newp
    #Point(tp22) = {xu, yu, z, lcar_far}
    #tp23 = newp
    #Point(tp23) = {x0, yu, z, lcar_far}
    #
    #// lines
    #cl20 = newc
    #Line(cl20) = {tp1,tp20}
    #cl21 = newc
    #Line(cl21) = {tp20,tp21}
    #cl22 = newc
    #Line(cl22) = {tp21,tp22}
    #cl23 = newc
    #Line(cl23) = {tp22,tp23}
    #cl24 = newc
    #Line(cl24) = {tp23,tp11}
    #
    #t = 0
    #theloops[t] = newreg
    #Line Loop(theloops[t]) = {
    #    cl20, cl21, cl22, cl23, cl24, -cl9b, -cl9a,
    #    -cl9, -cl6, -cl3, -cl2, -cl1, -tc1
    #    }

    #holdall = p4g.PlaneSurface(theloops[])
    #p4g.PhysicalSurface(holdall, 'air')
    # -------------------------------------------------------------------------
    # Get the code
    f = open(args.filename, 'w')
    f.write(p4g.get_code())
    f.close()
    return


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Crucible domain.')
    parser.add_argument('filename',
                        type=str,
                        help='output GEO file name'
                        )
    return parser.parse_args()


if __name__ == '__main__':
    _main()
