//Function Rectangle
//  // Points.
//  cp1 = newp;
//  Point(cp1) = {xmin,ymin,z,lcar};
//  cp2 = newp;
//  Point(cp2) = {xmax,ymin,z,lcar};
//  cp3 = newp;
//  Point(cp3) = {xmax,ymax,z,lcar};
//  cp4 = newp;
//  Point(cp4) = {xmin,ymax,z,lcar};
//
//  // Lines.
//  cl1 = newc;
//  Line(cl1) = {cp1,cp2};
//  cl2 = newc;
//  Line(cl2) = {cp2,cp3};
//  cl3 = newc;
//  Line(cl3) = {cp3,cp4};
//  cl4 = newc;
//  Line(cl4) = {cp4,cp1};
//
//  theloops[t] = newreg;
//  Line Loop(theloops[t]) = {cl1,cl2,cl3,cl4};
//  surf = news;
//  Plane Surface(surf) = {theloops[t]};
//
//  Physical Surface(Sprintf("coil %g", t)) = surf;
//
//Return

Function NestedRectangle
  // Outer points.
  cp1 = newp;
  Point(cp1) = {xmin,ymin,z,lcar};
  cp2 = newp;
  Point(cp2) = {xmax,ymin,z,lcar};
  cp3 = newp;
  Point(cp3) = {xmax,ymax,z,lcar};
  cp4 = newp;
  Point(cp4) = {xmin,ymax,z,lcar};
  // Outer lines.
  cl1 = newc;
  Line(cl1) = {cp1,cp2};
  cl2 = newc;
  Line(cl2) = {cp2,cp3};
  cl3 = newc;
  Line(cl3) = {cp3,cp4};
  cl4 = newc;
  Line(cl4) = {cp4,cp1};

  s = 0;
  myloops[s] = newreg;
  Line Loop(myloops[s]) = {cl1,cl2,cl3,cl4};

  // Save for cutting out.
  theloops[t] = myloops[s];

  // Inner points.
  cp5 = newp;
  Point(cp5) = {xmin+d,ymin+d,z,lcar};
  cp6 = newp;
  Point(cp6) = {xmax-d,ymin+d,z,lcar};
  cp7 = newp;
  Point(cp7) = {xmax-d,ymax-d,z,lcar};
  cp8 = newp;
  Point(cp8) = {xmin+d,ymax-d,z,lcar};
  // Inner lines.
  cl5 = newc;
  Line(cl5) = {cp5,cp6};
  cl6 = newc;
  Line(cl6) = {cp6,cp7};
  cl7 = newc;
  Line(cl7) = {cp7,cp8};
  cl8 = newc;
  Line(cl8) = {cp8,cp5};

  s = 1;
  myloops[s] = newreg;
  Line Loop(myloops[s]) = {cl5,cl6,cl7,cl8};

  // Make the interior a surface.
  air = news;
  Plane Surface(air) = {myloops[1]};
  Physical Surface(Sprintf("air %g", t)) = air;

  // Coil.
  surf = news;
  Plane Surface(surf) = {myloops[]};
  Physical Surface(Sprintf("coil %g", t)) = surf;

  // Mark the boundary area for refinement.
  attractor_id = newf;
  Field[attractor_id] = Attractor;
  Field[attractor_id].NNodesByEdge = 1000;
  Field[attractor_id].EdgesList = {cl1,cl2,cl3,cl4,cl5,cl6,cl7,cl8};

  refinement_id = newf;
  Field[refinement_id] = Threshold;
  Field[refinement_id].IField = attractor_id;
  Field[refinement_id].LcMin = lcar_b;
  Field[refinement_id].LcMax = lcar;
  Field[refinement_id].DistMin = w_b0;
  Field[refinement_id].DistMax = w_b1;

  thefields[t] = refinement_id;

Return

// All length quantities in file are in meters.
z = 0;
lcar = 5e-3;
lcar_air = 10e-3;
lcar_b = 0.1e-3;

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
eps = 1e-2;
lcar_b = -eps * lcar * Log(lcar);
// The width of the boundary layer has to have a minimum value; otherwise
// Gmsh will come up with a weird mesh around the boundaries.
//w_b = -eps * Log(lcar);
w_b0 = 1e-3;
w_b1 = 15 * w_b0;


// Rectangular coils.
xmin = 0.025;
xmax = 0.03;

t = 1;
ymax = 0.150;
ymin = 0.140;
d = 0.001;
Call NestedRectangle;

t = 2;
ymax = 0.137;
ymin = 0.127;
Call NestedRectangle;

t = 3;
ymax = 0.1215;
ymin = 0.1115;
Call NestedRectangle;

t = 4;
ymax = 0.1035;
ymin = 0.0935;
Call NestedRectangle;

t = 5;
ymax = 0.088;
ymin = 0.078;
Call NestedRectangle;

t = 6;
ymax = 0.075;
ymin = 0.065;
Call NestedRectangle;


// Build workpiece.
// The most convenient thing would be to define the workpiece as a rectangle
// and then treat it as a whole in the domain just like the circles above.
// However, one edge of the workpiece coincides with an edge of the hold all
// domain. Gmsh's meshing algorithm cannot deal with this situation yet.
// Hence, define the lines and points such as to create workpiece and hold-all
// separately.
x0 = 0.0;
x1 = 0.010;
x2 = 0.015;
x3 = 0.2;
y0 = -0.05;
y1 = 0.0;
y2 = 0.085;
y3 = 0.115;
y4 = 0.135;
y5 = 0.20;

// Define the points for the double rectangle.
cp1 = newp;
Point(cp1) = {x0,y1,z,lcar};
cp2 = newp;
Point(cp2) = {x2,y1,z,lcar};
cp3 = newp;
Point(cp3) = {x2,y2,z,lcar};
cp4 = newp;
Point(cp4) = {x1,y3,z,lcar};
cp5 = newp;
Point(cp5) = {x1,y4,z,lcar};
cp6 = newp;
Point(cp6) = {x0,y4,z,lcar};
cp7 = newp;
Point(cp7) = {x0,y0,z,lcar_air};
cp8 = newp;
Point(cp8) = {x3,y0,z,lcar_air};
cp9 = newp;
Point(cp9) = {x3,y5,z,lcar_air};
cp10 = newp;
Point(cp10) = {x0,y5,z,lcar_air};

// Lines.
cl1 = newc;
Line(cl1) = {cp1,cp2};
cl2 = newc;
Line(cl2) = {cp2,cp3};
cl3 = newc;
Line(cl3) = {cp3,cp4};
cl4 = newc;
Line(cl4) = {cp4,cp5};
cl5 = newc;
Line(cl5) = {cp5,cp6};
cl6 = newc;
Line(cl6) = {cp6,cp1};
cl7 = newc;
Line(cl7) = {cp1,cp7};
cl8 = newc;
Line(cl8) = {cp7,cp8};
cl9 = newc;
Line(cl9) = {cp8,cp9};
cl10 = newc;
Line(cl10) = {cp9,cp10};
cl11 = newc;
Line(cl11) = {cp10,cp6};

// Mark the boundary area for refinement.
attractor_id = newf;
Field[attractor_id] = Attractor;
Field[attractor_id].NNodesByEdge = 1000;
Field[attractor_id].EdgesList = {cl1,cl2,cl3,cl4,cl5};

workpiece_refinement_id = newf;
Field[workpiece_refinement_id] = Threshold;
Field[workpiece_refinement_id].IField = attractor_id;
Field[workpiece_refinement_id].LcMin = lcar_b;
Field[workpiece_refinement_id].LcMax = lcar;
Field[workpiece_refinement_id].DistMin = w_b0;
Field[workpiece_refinement_id].DistMax = w_b1;

// Workpiece surface.
loop = newreg;
Line Loop(loop) = {cl1,cl2,cl3,cl4,cl5,cl6};
wp = news;
Plane Surface(wp) = {loop};
Physical Surface(Sprintf("workpiece")) = wp;

// Hold all domain.
t = 0;
theloops[t] = newreg;
Line Loop(theloops[t]) = {cl1,cl2,cl3,cl4,cl5,-cl11,-cl10,-cl9,-cl8,-cl7};
air = news;
Plane Surface(air) = {theloops[]};
Physical Surface(Sprintf("air")) = air;

// Make elements match at interface boundaries.
// Without 'Coherence', the elements may still match, but the edges may be
// registered twice. This causes trouble with Extrude.
Coherence;

// Finally, let's use the minimum of all the fields as the background
// mesh field.
background_field_id = newf;
Field[background_field_id] = Min;
Field[background_field_id].FieldsList = {thefields[],
                                         workpiece_refinement_id};
Background Field = background_field_id;

// Setting CharacteristicLengthExtendFromBoundary to 0 makes sure that the mesh
// isn't unneccisarily refined at the interior of certain structures.
Mesh.CharacteristicLengthExtendFromBoundary = 0;
