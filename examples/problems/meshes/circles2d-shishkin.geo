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
  Field[attractor_id].NNodesByEdge = 1000;
  Field[attractor_id].EdgesList = {tc1, tc2, tc3, tc4};

  refinement_id = newf;
  Field[refinement_id] = Threshold;
  Field[refinement_id].IField = attractor_id;
  Field[refinement_id].LcMin = lcar_b;
  Field[refinement_id].LcMax = lcar;
  Field[refinement_id].DistMin = w_b0;
  Field[refinement_id].DistMax = w_b1;

  thefields[t] = refinement_id;
Return

Function SetBoundaryLayerParameters
  // This function computes the necessary parameters for boundary layers.

  // For 1D problems on the unit interval, the Shishkin mesh is characterized
  // by the PDE parameters eps, beta, and the mesh parameters q, sigma, N;
  // N is the number of grid points, q the portion of grid points in the
  // boundary layer. The width of the boundary layer is
  //    tau = min{q, sigma*eps/beta * log(N)},
  // The minimum is taken to make sure that the mesh isn't coarser at the
  // boundary than in in non-boundary regions.
  // The mesh widths (in the boundary region and far from it) are
  //
  //     w_b = 1 / N
  //     w_f = 1 / N
  //
  // if tau == q, and
  //
  //     w_b = sigma*eps / (q*beta) * log(N)/N
  //     w_f = (1-sigma*eps / beta*log(N)) / ((1-q) * N)
  //
  // otherwise.
  // For higher dimensional PDEs, the parameter N is not useful. Instead, the
  // mesh with w_f typically specified (edge length in the far field, "lcar").
  // The relation N=N(w_f), is given by the above equations, but cannot be
  // solved exactly. As a workaround, we use the Pade(1,1)-approximation to
  // log(1+N),
  //
  //     P(1,1)(log(1+x)) = x / (1+x/2).
  //
  // The yields (with w_f=lcar)
  //
  N = (1/(2*beta*lcar*(1-q))) * (beta - 2*eps*sigma - beta*lcar*(1-q) + Sqrt(4*beta*lcar*(1-q)*(beta + 2*eps*sigma) + (-2*eps*sigma + beta * (1-(1-q)*lcar))^2));
  //
  // Hence
  //
  alpha = sigma*eps / (q*beta) * Log(N)/N;
  lcar_b = (lcar < alpha) ? lcar : alpha; // Min(lcar, alpha)
  //
  // and
  //
  w_b0 = sigma*eps/beta * Log(N);
  //
  // Note that we don't need to take a Min() for the boundary width since the
  // mesh width is already capped by lcar_b.
  //
  //
  // LcMax -                         /------------------
  //                               /
  //                             /
  //                           /
  // LcMin -o----------------/
  //        |                |       |
  //     Attractor       DistMin   DistMax
  //
  // Don't choose w_b1 too small. The size of the triangles needs to have
  // enough room to gradually increase from lcar_b to lcar. If w_b1 is chosen
  // to small, the resulting mesh may look really funny around the boundaries.
  m = 0.5;
  w_b1 = w_b0 + (lcar-lcar_b) / m;
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
//

// For Shishkin meshes on general domains with smooth boundaries, check
//
//     Uniform Approximation of Singularly Perturbed Reaction-Diffusion
//     Problems by the Finite Element Method on a Shishkin Mesh;
//     Christos Xenophontos, Scott R. Fulton;
//     <http://www2.ucy.ac.cy/~xenophon/pubs/cx_sf.pdf>.
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
mu_graphite = mu0 * 0.999984;
// sigma: electrical conductivity
sigma_copper = 5.96e7;
sigma_silver = 6.30e7;
sigma_graphite = 1.0e5;

// omega: current frequency
omega = 2 * Pi * 10e3;

eps_copper = Sqrt(1.0 / (mu_copper*sigma_copper*omega));
eps_graphite = Sqrt(1.0 / (mu_graphite*sigma_graphite*omega));

eps = eps_copper;
q = 0.5;
beta = 1.0;
sigma = 1.0;
Call SetBoundaryLayerParameters; // sets w_b0, w_b1, lcar_b

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
eps = eps_graphite;
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

// -----------------------------------------------------------------------------
// Refine the mesh along the conductor boundaries.
attractor_id = newf;
Field[attractor_id] = Attractor;
Field[attractor_id].NNodesByEdge = 10000;
Field[attractor_id].EdgesList = {cl5, cl6};

// See comments above.
eps = eps_graphite;
// q, beta, sigma are set above.
Call SetBoundaryLayerParameters;

refinement_id = newf;
Field[refinement_id] = Threshold;
Field[refinement_id].IField = attractor_id;
Field[refinement_id].LcMin = lcar_b;
Field[refinement_id].LcMax = lcar;
Field[refinement_id].DistMin = w_b0;
Field[refinement_id].DistMax = w_b1;
// -----------------------------------------------------------------------------
// Finally, let's use the minimum of all the fields as the background
// mesh field.
background_field_id = newf;
Field[background_field_id] = Min;
Field[background_field_id].FieldsList = {thefields[], refinement_id};
Background Field = background_field_id;

Mesh.CharacteristicLengthExtendFromBoundary= 0;
// -----------------------------------------------------------------------------
