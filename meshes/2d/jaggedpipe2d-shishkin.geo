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

Function BoundaryLayer
  // This function computes the necessary parameters for boundary layers.

  // For 1D problems on the unit interval, the Shishkin mesh is characterized
  // by the PDE parameters eps, beta, and the mesh parameters q, sigma, N;
  // N is the number of grid points, q the portion of grid points in the boundary
  // layer. The width of the boundary layer is
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
  // solved exactly. As an workaround, we use the Pade(1,1)-approximation to
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
  // Note that we don't need to take a Min() for the boundary width since the mesh
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


width0 = 0.4;
width1 = 0.4;
x0 = 1.4;
x1 = 0.8;
h0_left = 0.8;
h0_right = 1.2;
h1_left = 1.2;
h1_right = 0.8;

z = 0;
lcar = 0.02;



//eps = Sqrt(1.0 / (mu_copper*sigma_copper*omega));
//q = 0.5;
//beta = 1.0;
//sigma = 1.0;
//Call BoundaryLayer; // sets w_b0, w_b1, lcar_b


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
xmax = 2.0;
ymin = -1.0;
ymax= 1.0;

cp1 = newp;
Point(cp1) = {xmin,ymin,z,lcar_air};
cp2 = newp;
Point(cp2) = {xmax,ymin,z,lcar_air};
cp3 = newp;
Point(cp3) = {xmax,ymax,z,lcar_air};
cp4 = newp;
Point(cp4) = {xmin,ymax,z,lcar_air};
cp5 = newp;
Point(cp5) = {x0-width0/2,ymin,z,lcar};
cp6 = newp;
Point(cp6) = {x0+width0/2,ymin,z,lcar};
cp7 = newp;
Point(cp7) = {x0-width0/2,ymin+h0_left,z,lcar};
cp8 = newp;
Point(cp8) = {x0+width0/2,ymin+h0_right,z,lcar};
cp9 = newp;
Point(cp9) = {x1-width1/2,ymax-h1_left,z,lcar};
cp10 = newp;
Point(cp10) = {x1+width1/2,ymax-h1_right,z,lcar};
cp11 = newp;
Point(cp11) = {x1-width1/2,ymax,z,lcar};
cp12 = newp;
Point(cp12) = {x1+width1/2,ymax,z,lcar};


// Lines.
cl1 = newc;
Line(cl1) = {cp1,cp5};
cl2 = newc;
Line(cl2) = {cp5,cp6};
cl3 = newc;
Line(cl3) = {cp6,cp2};
cl4 = newc;
Line(cl4) = {cp2,cp3};
cl5 = newc;
Line(cl5) = {cp3,cp12};
cl6 = newc;
Line(cl6) = {cp12,cp11};
cl7 = newc;
Line(cl7) = {cp11,cp4};
cl8 = newc;
Line(cl8) = {cp4,cp1};
// Pipe lines:
cl9 = newc;
Line(cl9) = {cp5,cp7};
cl10 = newc;
Line(cl10) = {cp7,cp9};
cl11 = newc;
Line(cl11) = {cp9,cp11};
cl12 = newc;
Line(cl12) = {cp6,cp8};
cl13 = newc;
Line(cl13) = {cp8,cp10};
cl14 = newc;
Line(cl14) = {cp10,cp12};


// Pipe:
loop = newreg;
Line Loop(loop) = {cl2,cl12,cl13,cl14,cl6,-cl11,-cl10,-cl9};
wp = news;
Plane Surface(wp) = {loop};
Physical Surface(Sprintf("pipe")) = wp;

//// Area left of pipe:
//loop = newreg;
//Line Loop(loop) = {cl1,cl9,cl10,cl11,cl7,cl8};
//wp = news;
//Plane Surface(wp) = {loop};
//Physical Surface(Sprintf("air")) = wp;
//
//// Area right of pipe:
//loop = newreg;
//Line Loop(loop) = {cl3,cl4,cl5,-cl14,-cl13,-cl12};
//wp = news;
//Plane Surface(wp) = {loop};
//Physical Surface(Sprintf("air")) = wp;

//// Hold all domain.
//t = 0;
//theloops[t] = newreg;
//Line Loop(theloops[t]) = {cl1,cl2,cl3,cl4,cl5,cl6,cl7};
//air = news;
//Plane Surface(air) = {theloops[]};
//Physical Surface(Sprintf("air")) = air;

// Make elements match at interface boundaries.
// Without 'Coherence', the elements may still match, but the
// edges may be registered twice. This causes trouble with
// Extrude.
Coherence;

//// Refine the mesh along the conductor boundaries
//// using Attractor (Field[1]) and Threshold (Field[2]).
//attractor_id = newf;
//Field[attractor_id] = Attractor;
//Field[attractor_id].NNodesByEdge = 10000;
//Field[attractor_id].EdgesList = {cl5, cl6};
//
//// See comments above.
//eps = Sqrt(1.0 / (mu_silver*sigma_silver*omega));
//// q, beta, sigma are set above.
//Call BoundaryLayer;
//
//// We then define a Threshold field, which uses the return value of
//// the Attractor Field[1] in order to define a simple change in
//// element size around the attractors (i.e., around point 5 and line
//// 1)
////
//refinement_id = newf;
//Field[refinement_id] = Threshold;
//Field[refinement_id].IField = attractor_id;
//Field[refinement_id].LcMin = lcar_b;
//Field[refinement_id].LcMax = lcar;
//Field[refinement_id].DistMin = w_b0;
//Field[refinement_id].DistMax = w_b1;
//
//// Finally, let's use the minimum of all the fields as the background
//// mesh field.
//background_field_id = newf;
//Field[background_field_id] = Min;
//Field[background_field_id].FieldsList = {thefields[], refinement_id};
//Background Field = background_field_id;
//
//Mesh.CharacteristicLengthExtendFromBoundary= 0;
