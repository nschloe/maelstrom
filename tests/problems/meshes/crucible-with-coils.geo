// Domain as presented in
//
// [1] Mathematical modeling of Czochralski-type growth processes for
//     semicondictor bulk single crystals;
//     W. Dreyer, P.-E. Druet, O. Klein, J. Sprekels.
//
// ----------------------------------------------------------------------------
Function MyCircle
  // Given a zshift and two radii rad and xshift, and a zshift, this
  // creates a torus parallel to the x-y-plane.
  // The points:
  cp1 = newp;
  Point(cp1) = {xshift,yshift,z,lcar};
  cp2 = newp;
  Point(cp2) = {rad+xshift,yshift,z,lcar};
  cp3 = newp;
  Point(cp3) = {xshift,yshift+rad,z,lcar};
  cp4 = newp;
  Point(cp4) = {xshift,yshift-rad,z,lcar};
  cp5 = newp;
  Point(cp5) = {-rad+xshift,yshift,z,lcar};

  // Build circle from arcs.
  cc1 = newc;
  Circle(cc1) = {cp2,cp1,cp3};
  cc2 = newc;
  Circle(cc2) = {cp3,cp1,cp5};
  cc3 = newc;
  Circle(cc3) = {cp5,cp1,cp4};
  cc4 = newc;
  Circle(cc4) = {cp4,cp1,cp2};

  theloops[t] = newc;
  Line Loop(theloops[t]) = {cc1,cc2,cc3,cc4};

  surf1 = newc;
  Plane Surface(surf1) = {theloops[t]};
  // Name the circles.
  Physical Surface(Sprintf("coil %g", t)) = surf1;

  // Refinement around the boundaries.
  b_id = newf;
  Field[b_id] = BoundaryLayer;
  Field[b_id].EdgesList = {cc1,cc2,cc3,cc4};
  Field[b_id].hfar = lcar_far;
  Field[b_id].hwall_n = lcar_b;
  Field[b_id].hwall_t = lcar_b;
  Field[b_id].ratio = 1.1;
  Field[b_id].thickness = w_b0;
  Field[b_id].AnisoMax = 100.0;
  thefields[s] = b_id;
Return
// ----------------------------------------------------------------------------
Function Rectangle
  // Points.
  cp1r = newp;
  Point(cp1r) = {xmin,ymin,z,lcar_coil};
  cp2r = newp;
  Point(cp2r) = {xmax,ymin,z,lcar_coil};
  cp3r = newp;
  Point(cp3r) = {xmax,ymax,z,lcar_coil};
  cp4r = newp;
  Point(cp4r) = {xmin,ymax,z,lcar_coil};

  // Lines.
  cl1r = newc;
  Line(cl1r) = {cp1r,cp2r};
  cl2r = newc;
  Line(cl2r) = {cp2r,cp3r};
  cl3r = newc;
  Line(cl3r) = {cp3r,cp4r};
  cl4r = newc;
  Line(cl4r) = {cp4r,cp1r};

  theloops[t] = newreg;
  Line Loop(theloops[t]) = {cl1r,cl2r,cl3r,cl4r};
  coil = news;
  Plane Surface(coil) = {theloops[t]};

  // We can't use %d here. -- Gmsh can't handle it properly. :/
  Physical Surface(Sprintf("coil %f", t)) = coil;

  // Refinement around the boundaries.
  b_id = newf;
  Field[b_id] = BoundaryLayer;
  Field[b_id].EdgesList = {cl1r,cl2r,cl3r,cl4r};
  Field[b_id].hfar = lcar_far;
  Field[b_id].hwall_n = lcar_b;
  Field[b_id].hwall_t = lcar_b;
  Field[b_id].ratio = 1.1;
  Field[b_id].thickness = w_b0;
  Field[b_id].AnisoMax = 100.0;
  thefields[s] = b_id;
Return
// ----------------------------------------------------------------------------
z = 0.0;

lcar_base = 1.0e-1;
lcar_far = 10*lcar_base;
lcar_coil = lcar_base;
lcar_gas = 4 * lcar_base;
lcar_crucible = 1.0e-1;

// Counter for the background field.
s = 0;
// Counter for the line loops.
t = 0;

// omega: current frequency (for boundary layer width)
omega = 2 * Pi * 300.0;
//omega = 2 * Pi * 10.0e3;
mu0 = Pi * 4e-7;

// symmetry axis
x0 = 0.0;
// ----------------------------------------------------------------------------
// Crucible
crucible_wall_width = 0.002;

// The assumption that the outer outline of the crucible is defined by the
// ellipse
//
//     ((x-x0)/(crucible_radius-x0))^2 + ((y-y0)/(crucible_bottom-y0))^2 = 1
//
// leads to a geometry which isn't true to the original from [1].
// Hence, build some hand-crafted B-splines.
//
crucible_radius = 0.076 + crucible_wall_width;
// upper end of the elliptic bend
y0 = 0.38826701;
// lower end
crucible_bottom = 0.365917 - crucible_wall_width;
// upper crucible end
crucible_top = 0.500;
// liquid surface level
melt_level = 0.411;
// shielding gas surface level
gas_level = 0.426;

center = newp;
Point(center) = {x0, y0, z, lcar_crucible};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Outer ellipse.

// B-Spline.
tp1 = newp;
Point(tp1) = {x0, crucible_bottom, z, lcar_crucible};
tp2 = newp;
Point(tp2) = {crucible_radius, y0+0.005, z, lcar_crucible};

tp2_spline0 = newp;
Point(tp2_spline0) = {0.02, crucible_bottom, z, lcar_crucible};
tp2_spline1 = newp;
Point(tp2_spline1) = {crucible_radius, crucible_bottom+0.005, z, lcar_crucible};

tc1 = newc;
//Ellipse(tc1) = {tp1, center, tp2, tp2};
BSpline(tc1) = {tp1, tp2_spline0, tp2_spline1, tp2};

Printf("tc1: %g", tc1);
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Inner ellipse.

//// B-Spline.
//inner_crucible_start = newp;
//Point(inner_crucible_start) = {x0, crucible_bottom+crucible_wall_width, z,
//lcar_crucible};
//inner_crucible_end = newp;
//Point(inner_crucible_end) = {crucible_radius-crucible_wall_width, y0, z,
//lcar_crucible};
//
//tp4_spline0 = newp;
//Point(tp4_spline0) = {0.022, crucible_bottom+crucible_wall_width, z,
//lcar_crucible};
//tp4_spline1 = newp;
//Point(tp4_spline1) = {crucible_radius-crucible_wall_width,
//crucible_bottom+crucible_wall_width+0.005, z, lcar_crucible};
//
//tc2 = newc;
////Ellipse(tc2) = {inner_crucible_start, center, inner_crucible_end, inner_crucible_end};
//BSpline(tc2) = {inner_crucible_start, tp4_spline0, tp4_spline1, inner_crucible_end};
//Printf("tc2: %g", tc2);

// Linear line segments.
// Data from Tecplot.
inner_crucible_start = newp;
Point(inner_crucible_start) = {0.0, 0.365917, z, lcar_crucible};
tp101 = newp;
Point(tp101) = {0.00642931, 0.365989, z, lcar_crucible};
tp102 = newp;
Point(tp102) = {0.0128557, 0.36619601, z, lcar_crucible};
tp103 = newp;
Point(tp103) = {0.0192762, 0.366539, z, lcar_crucible};
tp104 = newp;
Point(tp104) = {0.0256882, 0.367017, z, lcar_crucible};
tp105 = newp;
Point(tp105) = {0.0320886, 0.36762899, z, lcar_crucible};
tp106 = newp;
Point(tp106) = {0.0384747, 0.368377, z, lcar_crucible};
tp107 = newp;
Point(tp107) = {0.0448436, 0.369259, z, lcar_crucible};
tp108 = newp;
Point(tp108) = {0.0511925, 0.370276, z, lcar_crucible};
tp109 = newp;
Point(tp109) = {0.0633982, 0.37261799, z, lcar_crucible};
tp110 = newp;
Point(tp110) = {0.0681625, 0.374695, z, lcar_crucible};
tp111 = newp;
Point(tp111) = {0.0736436, 0.37980801, z, lcar_crucible};
// The Tecplot data seems inaccurate here in that the x-value of this point
// and the top-right point of the melt do not coincide. Fix  this manually.
inner_crucible_end = newp;
Point(inner_crucible_end) = {0.0760625, y0, z, lcar_crucible};
//Point(inner_crucible_end) = {0.076, y0, z, lcar_crucible};

cl100 = newc;
Line(cl100) = {inner_crucible_start,tp101};
cl101 = newc;
Line(cl101) = {tp101,tp102};
cl102 = newc;
Line(cl102) = {tp102,tp103};
cl103 = newc;
Line(cl103) = {tp103,tp104};
cl104 = newc;
Line(cl104) = {tp104,tp105};
cl105 = newc;
Line(cl105) = {tp105,tp106};
cl106 = newc;
Line(cl106) = {tp106,tp107};
cl107 = newc;
Line(cl107) = {tp107,tp108};
cl108 = newc;
Line(cl108) = {tp108,tp109};
cl109 = newc;
Line(cl109) = {tp109,tp110};
cl110 = newc;
Line(cl110) = {tp110,tp111};
cl111 = newc;
Line(cl111) = {tp111,inner_crucible_end};

Printf("cl107: %g", cl107);

//tc2 = newc;
//Compound Line(tc2) = {cl100, cl101, cl102, cl103, cl104, cl105, cl106, cl107,
//                      cl108, cl109, cl110, cl111};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Extend and close.
tp5 = newp;
Point(tp5) = {crucible_radius, crucible_top, z, lcar_crucible};
tp6 = newp;
Point(tp6) = {crucible_radius - crucible_wall_width, crucible_top, z, lcar_crucible};
tp7 = newp;
Point(tp7) = {crucible_radius - crucible_wall_width, melt_level, z, lcar_crucible};
tp71 = newp;
Point(tp71) = {crucible_radius - crucible_wall_width, gas_level, z, lcar_crucible};
cl1 = newc;
Line(cl1) = {tp2,tp5};
cl2 = newc;
Line(cl2) = {tp5,tp6};
cl3 = newc;
Line(cl3) = {tp6,tp71};
cl31 = newc;
Line(cl31) = {tp71,tp7};
cl4 = newc;
Line(cl4) = {tp7,inner_crucible_end};
cl5 = newc;
Line(cl5) = {tp1,inner_crucible_start};
Printf("cl4: %g", cl4);

// Define crucible surface.
ll1 = newreg;
Line Loop(ll1) = {tc1, cl1, cl2, cl3, cl31, cl4,
                  //-tc2,
                  -cl111, -cl110, -cl109, -cl108, -cl107, -cl106, -cl105,
                  -cl104, -cl103, -cl102, -cl101, -cl100,
                  -cl5};
crucible = news;
Plane Surface(crucible) = {ll1};
//Physical Surface(Sprintf("crucible")) = crucible;
// ----------------------------------------------------------------------------
//// Crucible holder.
//crucible_holder_top = 0.49;
//ch1 = newp;
//Point(ch1) = {0.4*crucible_radius, crucible_bottom, z, lcar};
//ch2 = newp;
//Point(ch2) = {0.9*crucible_radius, crucible_bottom+0.007, z, lcar};
//
////ch2 = newp;
////Point(ch2) = {crucible_radius, crucible_holder_top, z, lcar};
//
//chl1 = newc;
//Line(chl1) = {tp1, ch1};
//chl2 = newc;
//Line(chl2) = {ch1, ch2};
// ----------------------------------------------------------------------------
// The gas.
x2 = 0.038;
tp8 = newp;
Point(tp8) = {x2, melt_level, z, lcar_gas};
tp9 = newp;
Point(tp9) = {x2, gas_level, z, lcar_gas};
cl6 = newc;
Line(cl6) = {tp71,tp9};
cl7 = newc;
Line(cl7) = {tp9,tp8};
cl8 = newc;
Line(cl8) = {tp8,tp7};

ll2 = newreg;
Line Loop(ll2) = {cl6, cl7, cl8, -cl31};
boron = news;
Plane Surface(boron) = {ll2};
Physical Surface(Sprintf("gas")) = boron;
// -----------------------------------------------------------------------------
// The crystal.
tp10 = newp;
Point(tp10) = {x0, melt_level, z, lcar_gas};
tp11 = newp;
Point(tp11) = {x0, crucible_top, z, lcar_gas};
tp11a = newp;
Point(tp11a) = {x0+0.01, crucible_top, z, lcar_gas};
tp11b = newp;
Point(tp11b) = {x0+0.01, gas_level+0.025, z, lcar_gas};

cl9 = newc;
Line(cl9) = {tp9,tp11b};
cl9a = newc;
Line(cl9a) = {tp11b,tp11a};
cl9b = newc;
Line(cl9b) = {tp11a,tp11};
cl10 = newc;
Line(cl10) = {tp11,tp10};
cl11 = newc;
Line(cl11) = {tp10,tp8};

ll3 = newreg;
Line Loop(ll3) = {cl9, cl9a, cl9b, cl10, cl11, -cl7};

crystal = news;
Plane Surface(crystal) = {ll3};
Physical Surface(Sprintf("crystal")) = crystal;
// ----------------------------------------------------------------------------
// The melt.
cl12 = newc;
Line(cl12) = {tp10,inner_crucible_start};
ll4 = newreg;
Line Loop(ll4) = {cl12,
                  //tc2,
                  cl100, cl101, cl102, cl103, cl104, cl105, cl106, cl107,
                  cl108, cl109, cl110, cl111,
                  -cl4, -cl8, -cl11};
melt = news;
Plane Surface(melt) = {ll4};
Physical Surface(Sprintf("melt")) = melt;

// Refinement around the boundaries.
// The boundary width of the melt is determined by diffusion coefficients of
// three PDEs:
//   * Maxwell;
//   * Navier-Stokes;
//   * heat equation.
//
T = 1511.0;
rho_gaas = 7.33e3 - 1.07 * T;
cp_gaas = 0.434e3;
reynolds = 6.197e3;
prandtl = 0.068;
mu_gaas = mu0*(1.0 - 0.85e-10);
sigma_gaas = 7.9e5;
kappa_gaas = 17.8;
// Boundary layer widths
w_maxwell = Sqrt(2.0/(mu_gaas*sigma_gaas*omega));
w_heat = Sqrt(1.0/(reynolds*prandtl));
w_navier = Sqrt(1.0/reynolds);
Printf("Melt boundary layer widths:");
Printf("    Maxwell: %f", w_maxwell);
Printf("    heat eq: %f", w_heat);
Printf("    Navier:  %f", w_navier);

// w_b0 = Min(w_maxwell, w_heat, w_navier);
w_b0 = w_maxwell;
w_b0 = (w_heat < w_b0) ? w_heat : w_b0;
w_b0 = (w_navier < w_b0) ? w_navier : w_b0;

// Fit 2*k elements normal into the boundary layer.
k = 10;
// Min(lcar_crucible, w_b0/k):
lcar_b = (lcar_crucible < w_b0/k) ? lcar_crucible : w_b0/k;
b_id = newf;
Field[b_id] = BoundaryLayer;
Field[b_id].EdgesList = {cl100, cl101, cl102, cl103, cl104, cl105, cl106,
                         cl107, cl108, cl109, cl110, cl111,
                         //tc2,
                         cl4,cl8,cl11};
Field[b_id].hfar = lcar_far;
Field[b_id].hwall_n = lcar_b;
Field[b_id].hwall_t = lcar_b;
Field[b_id].ratio = 1.1;
Field[b_id].thickness = w_b0;
Field[b_id].AnisoMax = 100.0;
s += 1;
thefields[s] = b_id;
// ----------------------------------------------------------------------------
// Coils.
// For layer-adapted meshes for reaction-diffusion problems, check out
//
// [1] Layer-adapted meshes for reaction-convection-diffusion problems;
//     T. Linß;
//     <https://www.springer.com/mathematics/computational+science+%26+engineering/book/978-3-642-05133-3>.
//
// The PDE for the potential \phi is of reaction-diffusion type
//
//     -eps^2 \Delta u + u = f
//
// with singularity parameter
//
//     eps^2 = 1 / (mu*sigma*omega)
//

// mu: magnetic permeability
mu_graphite = mu0 * 0.999984;
// sigma: electrical conductivity
T = 1511.0;
sigma_graphite = 1e6 / (28.9 - 18.8 * Exp(-(Log(T/1023.0)/2.37)^2));

// It exhibits layers where \phi behaves like exp(-x/eps). This also corresponds
// with the skin effect (https://en.wikipedia.org/wiki/Skin_effect) layer width
//
//     \delta = \sqrt{2/(mu*sigma*omega)}.
//
w_b0 = Sqrt(2.0/(mu_graphite*sigma_graphite*omega));
// Fit 2*k elements normal into the boundary layer.
k = 50;
// Min(lcar_coils, w_b0/k)
lcar_b = (lcar_coil < w_b0/k) ? lcar_coil : w_b0/k;

Printf("lcar_coil: %f", lcar_coil);
Printf("lcar boundary: %f", lcar_b);
Printf("Coil boundary layer width: %f", w_b0);

// Coils to the right.
xmin = 0.092;
xmax = xmin + 0.015;
ymin = 0.266;
ymax = ymin + 0.010;
For k In {1:15}
    s += 1;
    t += 1;
    ymin += 0.0132;
    ymax += 0.0132;
    Call Rectangle;
EndFor

//rad = 0.004;
//xshift = 0.09;
//yshift = 0.268;
//For k In {1:15}
//    s += 1;
//    t += 1;
//    yshift += 0.013;
//    Call MyCircle;
//EndFor

// Coils at the bottom.
xmin = 0.031;
xmax = xmin + 0.005;
ymin = 0.33;
ymax = ymin + 0.024;

For k In {1:7}
    s += 1;
    t += 1;
    Call Rectangle;
    xmin += 0.008;
    xmax += 0.008;
EndFor
// ----------------------------------------------------------------------------
// Hold-all domain.
r = 1.0;
// Draw first quarter-circle.
tp20 = newp;
Point(tp20) = {x0, y0, z, lcar_far};
tp21 = newp;
Point(tp21) = {x0, y0-r, z, lcar_far};
tp22 = newp;
Point(tp22) = {x0+r, y0, z, lcar_far};
tp23 = newp;
Point(tp23) = {x0, y0+r, z, lcar_far};

// Build circle from arcs.
cc1 = newc;
Circle(cc1) = {tp21,tp20,tp22};
cc2 = newc;
Circle(cc2) = {tp22,tp20,tp23};
// Connecting lines.
cl20 = newc;
Line(cl20) = {tp1,tp21};
cl24 = newc;
Line(cl24) = {tp23,tp11};

t = 0;
theloops[t] = newc;
Line Loop(theloops[t]) = {cl20,cc1,cc2,cl24,-cl9b,-cl9a,-cl9,-cl6,-cl3,-cl2,-cl1,-tc1};

// Rectangle.
//yl = 0.20;
//yu = 0.60;
//xu = 0.25;
//
//// outer corners
//tp20 = newp;
//Point(tp20) = {x0, yl, z, lcar_far};
//tp21 = newp;
//Point(tp21) = {xu, yl, z, lcar_far};
//tp22 = newp;
//Point(tp22) = {xu, yu, z, lcar_far};
//tp23 = newp;
//Point(tp23) = {x0, yu, z, lcar_far};
//
//// lines
//cl20 = newc;
//Line(cl20) = {tp1,tp20};
//cl21 = newc;
//Line(cl21) = {tp20,tp21};
//cl22 = newc;
//Line(cl22) = {tp21,tp22};
//cl23 = newc;
//Line(cl23) = {tp22,tp23};
//cl24 = newc;
//Line(cl24) = {tp23,tp11};
//
//t = 0;
//theloops[t] = newreg;
//Line Loop(theloops[t]) = {cl20,cl21,cl22,cl23,cl24,-cl9b,-cl9a,-cl9,-cl6,-cl3,-cl2,-cl1,-tc1};

holdall = news;
Plane Surface(holdall) = {theloops[]};
Physical Surface(Sprintf("air")) = holdall;
// ----------------------------------------------------------------------------
// Finally, let's use the minimum of all the fields as the background
// mesh field.
background_field_id = newf;
Field[background_field_id] = Min;
Field[background_field_id].FieldsList = {thefields[]};
Background Field = background_field_id;

Mesh.CharacteristicLengthExtendFromBoundary = 0;

// Decrease the precision of 1D integration. If set to default, mesh generation
// may take very long.
Mesh.LcIntegrationPrecision = 1.0e-3;
// ----------------------------------------------------------------------------
