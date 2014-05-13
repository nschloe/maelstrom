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

  theloops[t] = newreg;
  Line Loop(theloops[t]) = {tc1,tc2,tc3,tc4};
  surf1 = news;
  Plane Surface(surf1) = {theloops[t]};

  // Name the circles.
  Physical Surface(Sprintf("coil %g", t)) = surf1;

  // Refinement around the boundaries.
  attractor_id = newf;
  Field[attractor_id] = Attractor;
  Field[attractor_id].NNodesByEdge = 100;
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

z = 0;
lcar = 0.1;
lcar_air = 0.1;
lcar_b = 0.01;
eps = 1e-2;
lcar_b = -eps * lcar * Log(lcar);
// The width of the boundary layer has to have a minimum value; otherwise
// Gmsh will come up with a weird mesh around the boundaries.
//w_b = -eps * Log(lcar);
w_b0 = 0.01;
w_b1 = 20 * w_b0;

// Circles (coils).
t = 1;
rad = 0.1;
xshift = 0.8;
yshift = 0.3;
Call MyCircle;
t = 2;
yshift = 0.0;
Call MyCircle;
t = 3;
yshift = -0.3;
Call MyCircle;

// The most convenient thing would be to define the workpiece as a rectangle
// and then treat it as a whole in the domain just like the circles above.
// However, one edge of the workpiece coincides with an edge of the hold all
// domain. Gmsh's meshing algorithm cannot deal with this situation yet.
// Hence, define the lines and points such as to create workpiece and hold-all
// separately.
t = 2;
xmin = 0.0;
xmax = 3.0;
ymin = -2.0;
ymax= 2.0;
xmax2 = 0.5;
ymin2 = -0.5;
ymax2 = 0.5;

// Define the points for the double rectangle.
cp1 = newp;
Point(cp1) = {xmin,ymin,z,lcar_air};
cp2 = newp;
Point(cp2) = {xmax,ymin,z,lcar_air};
cp3 = newp;
Point(cp3) = {xmax,ymax,z,lcar_air};
cp4 = newp;
Point(cp4) = {xmin,ymax,z,lcar_air};
cp5 = newp;
Point(cp5) = {xmin,ymin2,z,lcar};
cp6 = newp;
Point(cp6) = {xmax2,ymin2,z,lcar};
cp7 = newp;
Point(cp7) = {xmax2,ymax2,z,lcar};
cp8 = newp;
Point(cp8) = {xmin,ymax2,z,lcar};

// Lines.
cl1 = newc;
Line(cl1) = {cp1,cp2};
cl2 = newc;
Line(cl2) = {cp2,cp3};
cl3 = newc;
Line(cl3) = {cp3,cp4};
cl4 = newc;
Line(cl4) = {cp4,cp8};
cl5 = newc;
Line(cl5) = {cp8,cp5};
cl6 = newc;
Line(cl6) = {cp5,cp1};
cl7 = newc;
Line(cl7) = {cp5,cp6};

cl8 = newc;
//// Straight line:
//Line(cl8) = {cp6,cp7};
// Circle:
cc = newp;
Point(cc) = {2.0, 0.5*(ymin2+ymax2), z, lcar};
Circle(cl8) = {cp6,cc,cp7};

tc1 = newc;
Circle(tc1) = {tp2,tp1,tp3};

cl9 = newc;
Line(cl9) = {cp7,cp8};

// Workpiece.
loop = newreg;
Line Loop(loop) = {cl7,cl8,cl9,cl5};
wp = news;
Plane Surface(wp) = {loop};
Physical Surface(Sprintf("workpiece")) = wp;

// Workpiece boundary refinement.
attractor_id = newf;
Field[attractor_id] = Attractor;
Field[attractor_id].NNodesByEdge = 1000;
Field[attractor_id].EdgesList = {cl7,cl8,cl9};

workpiece_refinement_id = newf;
Field[workpiece_refinement_id] = Threshold;
Field[workpiece_refinement_id].IField = attractor_id;
Field[workpiece_refinement_id].LcMin = lcar_b;
Field[workpiece_refinement_id].LcMax = lcar;
Field[workpiece_refinement_id].DistMin = w_b0;
Field[workpiece_refinement_id].DistMax = w_b1;

// Hold all domain.
t = 0;
theloops[t] = newreg;
Line Loop(theloops[t]) = {cl1,cl2,cl3,cl4,-cl9,-cl8,-cl7,cl6};
air = news;
Plane Surface(air) = {theloops[]};
Physical Surface(Sprintf("air")) = air;

// Make elements match at interface boundaries.
// Without 'Coherence', the elements may still match, but the
// edges may be registered twice. This causes trouble with
// Extrude.
Coherence;

// Finally, let's use the minimum of all the fields as the background
// mesh field.
background_field_id = newf;
Field[background_field_id] = Min;
Field[background_field_id].FieldsList = {thefields[],
                                         workpiece_refinement_id};
Background Field = background_field_id;

Mesh.CharacteristicLengthExtendFromBoundary= 0;
