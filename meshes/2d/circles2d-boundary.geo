// ----------------------------------------------------------------------------
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
  b_id = newf;
  Field[b_id] = BoundaryLayer;
  Field[b_id].EdgesList = {tc1,tc2,tc3,tc4};
  Field[b_id].hfar = lcar_far;
  Field[b_id].hwall_n = lcar_b;
  Field[b_id].hwall_t = lcar_b;
  Field[b_id].ratio = 1.1;
  Field[b_id].thickness = w_b0;
  Field[b_id].AnisoMax = 100.0;
  thefields[t] = b_id;
Return
// ----------------------------------------------------------------------------

// Scaling factor: radius of the center half-circle.
alpha = 0.08;

z = 0;
lcar = alpha * 1.0;
// Field and loop index.
t = 0;
// Elements in the boundary layer.
k = 2;

// For the boundary layer:
omega = 2 * Pi * 10.0e3;
//omega = 2 * Pi * 300.0;
mu0 = Pi * 4e-7;

// Boundary layer parameters.
mu_copper = mu0 * 0.999994;
sigma_copper = 5.96e7;
w_b0 = Sqrt(2.0/(mu_copper*sigma_copper*omega));
Printf("w_b0 = %f", w_b0);
// Fit 2*k elements normal into the boundary layer.
lcar_b = (lcar < w_b0/k) ? lcar : w_b0/k; // Min(lcar, w_b0/k)
lcar_far = lcar;

// Circle (coil).
t += 1;
rad = alpha*0.1;
xshift = alpha*1.5;
yshift = alpha*0.0;
Call MyCircle;
// ----------------------------------------------------------------------------
// Workpiece.
// The most convenient thing would be to define the workpiece
// and then treat it as a whole in the domain just like the circles above.
// However, one edge of the workpiece coincides with an edge of the hold all
// domain. Gmsh's meshing algorithm cannot deal with this situation yet.
// Hence, define the lines and points such as to create workpiece and hold-all
// separately.
z = 0;
lcar_air = lcar;
t = 2;
xmin = alpha*0.0;
xmax = alpha*3.0;
ymin = -2.0 * alpha;
ymax= 2.0 * alpha;
xmax2 = 1.0 * alpha;
ymin2 = -1.0 * alpha;
ymax2 = 1.0 * alpha;

// Define the points for the half-circular workpiece.
R = 1.0 * alpha;
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

// Refinement around the boundaries.
b_id = newf;
Field[b_id] = BoundaryLayer;
Field[b_id].EdgesList = {cl5,cl6};
Field[b_id].hfar = lcar_far;
Field[b_id].hwall_n = lcar_b;
Field[b_id].hwall_t = lcar_b;
Field[b_id].ratio = 1.1;
Field[b_id].thickness = w_b0;
Field[b_id].AnisoMax = 100.0;
t += 1;
thefields[t] = b_id;
// ----------------------------------------------------------------------------
// Hold all domain.
t = 0;
theloops[t] = newreg;
Line Loop(theloops[t]) = {cl1,cl2,cl3,cl4,cl5,cl6,cl7};
air = news;
Plane Surface(air) = {theloops[]};
Physical Surface(Sprintf("air")) = air;
// ----------------------------------------------------------------------------
// Make elements match at interface boundaries.
// Without 'Coherence', the elements may still match, but the
// edges may be registered twice. This causes trouble with
// Extrude.
Coherence;
// ----------------------------------------------------------------------------
// Finally, let's use the minimum of all the fields as the background
// mesh field.
background_field_id = newf;
Field[background_field_id] = Min;
Field[background_field_id].FieldsList = {thefields[]};
Background Field = background_field_id;

Mesh.CharacteristicLengthExtendFromBoundary= 0;

// Decrease the precision of 1D integration. If set to default, mesh generation
// may take very long.
Mesh.LcIntegrationPrecision = 1.0e-3 * alpha;
// ----------------------------------------------------------------------------
