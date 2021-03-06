Function MyCircle
  // Given a zshift and two radii rad and xshift, and a zshift, this
  // creates a torus parallel to the x-y-plane.
  // The points:
  tp1 = newp;
  Point(tp1) = {xshift,yshift,z,lcar};
  tp2 = newp;
  Point(tp2) = {rad+xshift,yshift,z,lcar};
  tp3 = newp;
  Point(tp3) = {xshift,yshift+rad,z,lcar};
  tp4 = newp;
  Point(tp4) = {xshift,yshift-rad,z,lcar};
  tp5 = newp;
  Point(tp5) = {-rad+xshift,yshift,z,lcar};

  // Build circle from arcs.
  tc1 = newc;
  Circle(tc1) = {tp2,tp1,tp3};
  tc2 = newc;
  Circle(tc2) = {tp3,tp1,tp5};
  tc3 = newc;
  Circle(tc3) = {tp5,tp1,tp4};
  tc4 = newc;
  Circle(tc4) = {tp4,tp1,tp2};

  theloops[t] = newc;
  Line Loop(theloops[t]) = {tc1,tc2,tc3,tc4};
  surf1 = newc;
  Plane Surface(surf1) = {theloops[t]};
  // Name the circles.
  Physical Surface(Sprintf("coil %g", t)) = surf1;

  // Refinement around the boundaries.
  attractor_id = newf;
  Field[attractor_id] = Attractor;
  Field[attractor_id].NNodesByEdge = 10000;
  Field[attractor_id].EdgesList = {tc1, tc2, tc3, tc4};

  // For Bakhvalov meshes on the 1D unit interval, the monitoring function is
  //
  //     M_{Ba}(x) = max{1, K * beta * eps^{-1} * exp(-beta*s/(sigma*eps))}
  //
  // where K, beta, sigma are mesh parameters. To retrieve a Bakhvalov mesh
  // with N grid points, equidistribute the function, i.e., the grid points
  // x_{i} fulfill
  //
  //    x_0 = 0
  //    \int_{x_i}^{x_{i+1}} M_{Ba}(x) dx = I/N
  //
  // where
  //
  //    I = \int_0^1 M_{Ba}(x) dx.
  //
  // In the boundary layer, i.e., where
  //
  //    1 < K * beta * eps^{-1} * exp(-beta*s/(sigma*eps)),
  //
  // the width of the interval following the point x_i is
  //
  //    h_1(x_i) = sigma*eps/beta \
  //             * log(1 / (1 - I/N * (K*sigma)^{-1} * exp(beta*x_i/(sigma*eps)))).
  //
  // This expression has a singularity where
  // I/N == K*sigma*exp(-beta*x_i/(sigma*eps)) and isn't defined beyond this
  // point.
  // Alternatively, the width h(x) can be characterized by the equation
  //
  //    I/N = \int_{x_0+h/2}^{x_0+h/2} M_{Ba}(x) dx,
  //
  // which leads to
  //
  //    h_2(x_0) = 2 * sigma*eps/beta \
  //             * asinh(0.5 * I/N * (K*sigma)^{-1} * exp(beta*x_0/(sigma*eps)))
  //
  // which is accurate at midpoints x_0 of actual intervals.
  // Both of the formulas nearly coincide for small x, and can naturally be
  // extended to the full interval by admitting all values of x instead of
  // just start- or midpoints of intervals. The quotient I/N coincides with
  // the width H of an interval in non-boundary regions.
  // The expression
  //
  //    h_3(x) = eps/beta * H * K^{-1} * exp(beta*x / (sigma*eps))
  //
  // also coincides at the boundary (TODO: proof).
  // Somewhat inaccurately, one can thus define a Bakhvalov mesh by its
  // interval with
  //
  //    h_1 = min{H, sigma*eps/beta \
  //                 * log(1 / (1 - H * (K*sigma)^{-1} * exp(beta*x/(sigma*eps))))}, or
  //    h_2 = min{H, 2 * sigma*eps/beta \
  //                 * asinh(0.5 * H * (K*sigma)^{-1} * exp(beta*x/(sigma*eps)))
  //             }, or
  //    h_3 = min{H, eps/beta * H * K^{-1} * exp(beta*x / (sigma*eps))}.
  //
  // For the sake of simplicity, use h_3 for the definition of the edge width
  // for 2D.
  //
  // Unfortunately, Gmsh doesn't create good quality meshes if lcar changes too
  // quickly. In particular, this is the case for the above expressions if eps
  // somewhat small.
  // As a workaround, one can replace exp() in h_3 by its Taylor-approximation
  // cut off at a certain term. Since T(exp(x))<=exp(x), the resulting mesh
  // will rather be finer than coarser.
  // For now, take the linear approximation 1+x, i.e.,
  //
  //     h_3' = H * min{1, K^{-1} * (eps/beta + x/sigma)}.
  //
  refinement_id = newf;
  Field[refinement_id] = MathEval;
  //Field[refinement_id].F = Sprintf("Min(%g, %g/%g * %g / %g * Exp(%g*F%g / (%g*%g)))",
  //                                 lcar, eps, beta, lcar, K, beta, attractor_id, sigma, eps);
  Field[refinement_id].F = Sprintf("Min(%g, %g/%g * %g / %g * (1.0 + %g*F%g/(%g*%g)))",
                                   lcar, eps, beta, lcar, K,
                                   beta, attractor_id, sigma, eps
                                   );

  thefields[t] = refinement_id;
Return

// For layer-adapted meshes for reaction-diffusion problems, check out
// https://www.springer.com/mathematics/computational+science+%26+engineering/book/978-3-642-05133-3
// (Layer-adapted meshes for reaction-convection-diffusion problems (Linß)).
//
// The PDE for the potential \phi is of reaction-diffusion type
//
//     -eps^2 \Delta u + u = f
//
// with singularity parameter
//
//     eps^2 = 1 / (mu*sigma*omega)
//
// It exhibits layers where \phi behaves like exp(-x/eps). This also corresponds
// with the skin effect (https://en.wikipedia.org/wiki/Skin_effect) layer width
//
//     \delta = \sqrt{2/(mu*sigma*omega)}.
//

z = 0;
lcar = 0.2;

// Set the Shishkin mesh parameters (width of boundary layer and lcar).
// Shishkin says (cf. http://www.math.ualberta.ca/ijnam/Volume-7-2010/No-3-10/2010-03-01.pdf),
// figure 1:
//   * width of the boundary layer: w = eps * C_sigma * log(1/h)
//   * element size in the boundary layer: lcar_b = w*h / 2
// Taking C_sigma=1 and
// assuming an eps (approx. omega^{-1} in induction heating context) of 1e-3,
// this yields
//   * w = - 1e-3 * log(lcar)
//   * lcar_b = -1e-3 * lcar*log(lcar)

// mu: magnetic permeability
mu0 = Pi * 4e-7;
mu_copper = mu0 * 0.999994;
mu_silver = mu0 * 0.999998;
// sigma: electrical conductivity
sigma_copper = 5.96e7;
sigma_silver = 6.30e7;

// omega: current frequency
omega = 10e3;


K = 1.0;
beta = 1.0;
sigma = 1.0;

eps = Sqrt(1.0 / (mu_copper*sigma_copper*omega));

// Circle (coil).
t = 1;
rad = 0.1;
xshift = 1.5;
yshift = 0.0;
Call MyCircle;

// The most convenient thing would be to define the workpiece
// and then treat it as a whole in the domain just like the circles above.
// However, one edge of the workpiece coincides with an edge of the hold all
// domain. Gmsh's meshing algorithm cannot deal with this situation yet.
// Hence, define the lines and points such as to create workpiece and hold-all
// separately.
z = 0;
lcar_air = lcar;
t = 2;
xmin = 0.0;
xmax = 3.0;
ymin = -2.0;
ymax= 2.0;
xmax2 = 1.0;
ymin2 = -1.0;
ymax2 = 1.0;

// Define the points for the half-circular workpiece.
R = 1.0;
cp1 = newp;
Point(cp1) = {xmin,ymin,z,lcar_air};
cp2 = newp;
Point(cp2) = {xmax,ymin,z,lcar_air};
cp3 = newp;
Point(cp3) = {xmax,ymax,z,lcar_air};
cp4 = newp;
Point(cp4) = {xmin,ymax,z,lcar_air};
cp5 = newp;
Point(cp5) = {xmin,-R,z,lcar};
cp6 = newp;
Point(cp6) = {R,0.0,z,lcar};
cp7 = newp;
Point(cp7) = {xmin,R,z,lcar};
cp8 = newp;
Point(cp8) = {0.0,0.0,z,lcar};

// Lines.
cl1 = newc;
Line(cl1) = {cp1,cp2};
cl2 = newc;
Line(cl2) = {cp2,cp3};
cl3 = newc;
Line(cl3) = {cp3,cp4};
cl4 = newc;
Line(cl4) = {cp4,cp7};
cl5 = newc;
Circle(cl5) = {cp7,cp8,cp6};
cl6 = newc;
Circle(cl6) = {cp6,cp8,cp5};
cl7 = newc;
Line(cl7) = {cp5,cp1};
cl8 = newc;
Line(cl8) = {cp5,cp7};

// Workpiece.
loop = newreg;
Line Loop(loop) = {cl5,cl6,cl8};
wp = news;
Plane Surface(wp) = {loop};
Physical Surface(Sprintf("workpiece")) = wp;

// Hold all domain.
t = 0;
theloops[t] = newreg;
Line Loop(theloops[t]) = {cl1,cl2,cl3,cl4,cl5,cl6,cl7};
air = news;
Plane Surface(air) = {theloops[]};
Physical Surface(Sprintf("air")) = air;

// Make elements match at interface boundaries.
// Without 'Coherence', the elements may still match, but the
// edges may be registered twice. This causes trouble with
// Extrude.
Coherence;

// Refine the mesh along the conductor boundaries
// using Attractor (Field[1]) and Threshold (Field[2]).
attractor_id = newf;
Field[attractor_id] = Attractor;
Field[attractor_id].NNodesByEdge = 10000;
Field[attractor_id].EdgesList = {cl5, cl6};

// See comments above.
eps = Sqrt(1.0 / (mu_silver*sigma_silver*omega));

refinement_id = newf;
Field[refinement_id] = MathEval;
//Field[refinement_id].F = Sprintf("Min(%g, %g/%g * %g / %g * Exp(%g*F%g / (%g*%g)))",
//                                 lcar, eps, beta, lcar, K, beta, attractor_id, sigma, eps);
//Field[refinement_id].F = Sprintf("Min(%g, %g/%g * %g / %g * (1.0 + %g*F%g/(%g*%g)))",
//                                 lcar, eps, beta, lcar, K,
//                                 beta, attractor_id, sigma, eps
//                                 );
Field[refinement_id].F = Sprintf("Min(%g, %g/%g * %g / %g * (1.0 + %g*F%g/(%g*%g) + 0.5 * (%g*F%g/(%g*%g))^2))",
                                 lcar, eps, beta, lcar, K,
                                 beta, attractor_id, sigma, eps,
                                 beta, attractor_id, sigma, eps
                                 );

// Finally, let's use the minimum of all the fields as the background
// mesh field.
background_field_id = newf;
Field[background_field_id] = Min;
Field[background_field_id].FieldsList = {thefields[], refinement_id};
Background Field = background_field_id;

Mesh.CharacteristicLengthExtendFromBoundary= 0;
