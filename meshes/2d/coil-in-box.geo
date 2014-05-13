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
  //surf1 = news;
  //Plane Surface(surf1) = {theloops[t]};

  //// Name the circles.
  //Physical Surface(Sprintf("coil %g", t)) = surf1;
Return

z = 0;
lcar = 2.0e-2;

// Circles (coils).
t = 1;
rad = 0.05;
xshift = 0.2;
yshift = 0.2;
Call MyCircle;

// The most convenient thing would be to define the workpiece as a rectangle
// and then treat it as a whole in the domain just like the circles above.
// However, one edge of the workpiece coincides with an edge of the hold all
// domain. Gmsh's meshing algorithm cannot deal with this situation yet.
// Hence, define the lines and points such as to create workpiece and hold-all
// separately.
t = 2;
xmin = 0.0;
xmax = 2.5;
ymin = 0.0;
ymax= 0.4;

// Define the points for the double rectangle.
cp1 = newp;
Point(cp1) = {xmin,ymin,z,lcar};
cp2 = newp;
Point(cp2) = {xmax,ymin,z,lcar};
cp3 = newp;
Point(cp3) = {xmax,ymax,z,lcar};
cp4 = newp;
Point(cp4) = {xmin,ymax,z,lcar};

// Lines.
cl1 = newc;
Line(cl1) = {cp1,cp2};
cl2 = newc;
Line(cl2) = {cp2,cp3};
cl3 = newc;
Line(cl3) = {cp3,cp4};
cl4 = newc;
Line(cl4) = {cp4,cp1};

// Hold all domain.
t = 0;
theloops[t] = newreg;
Line Loop(theloops[t]) = {cl1,cl2,cl3,cl4};
liquid = news;
Plane Surface(liquid) = {theloops[]};
Physical Surface(Sprintf("liquid")) = liquid;
