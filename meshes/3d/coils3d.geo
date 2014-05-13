// Gmsh geometry for a rectangular coil surrounded by by two tori.

Function Brick
  // Creates a rectangular brick.

  // Define points.
  cp1 = newp;
  Point(cp1) = {xmax,ymax,zmax,lcar};
  cp2 = newp;
  Point(cp2) = {xmax,ymax,zmin,lcar};
  cp3 = newp;
  Point(cp3) = {xmax,ymin,zmax,lcar};
  cp4 = newp;
  Point(cp4) = {xmax,ymin,zmin,lcar};
  cp5 = newp;
  Point(cp5) = {xmin,ymax,zmax,lcar};
  cp6 = newp;
  Point(cp6) = {xmin,ymax,zmin,lcar};
  cp7 = newp;
  Point(cp7) = {xmin,ymin,zmax,lcar};
  cp8 = newp;
  Point(cp8) = {xmin,ymin,zmin,lcar};

  // Lines.
  cl1 = newreg;
  Line(cl1) = {cp1,cp2};
  cl2 = newreg;
  Line(cl2) = {cp1,cp3};
  cl3 = newreg;
  Line(cl3) = {cp1,cp5};
  cl4 = newreg;
  Line(cl4) = {cp2,cp4};
  cl5 = newreg;
  Line(cl5) = {cp2,cp6};
  cl6 = newreg;
  Line(cl6) = {cp3,cp4};
  cl7 = newreg;
  Line(cl7) = {cp3,cp7};
  cl8 = newreg;
  Line(cl8) = {cp4,cp8};
  cl9 = newreg;
  Line(cl9) = {cp5,cp6};
  cl10 = newreg;
  Line(cl10) = {cp5,cp7};
  cl11 = newreg;
  Line(cl11) = {cp6,cp8};
  cl12 = newreg;
  Line(cl12) = {cp7,cp8};

  // Surfaces.
  cll1 = newreg; Line Loop(cll1) = {cl1,cl4,-cl6,-cl2};    Plane Surface(newreg) = {cll1};
  cll2 = newreg; Line Loop(cll2) = {cl1,cl5,-cl9,-cl3};    Plane Surface(newreg) = {cll2};
  cll3 = newreg; Line Loop(cll3) = {cl2,cl7,-cl10,-cl3};   Plane Surface(newreg) = {cll3};
  cll4 = newreg; Line Loop(cll4) = {cl4,cl8,-cl11,-cl5};   Plane Surface(newreg) = {cll4};
  cll5 = newreg; Line Loop(cll5) = {cl6,cl8,-cl12,-cl7};   Plane Surface(newreg) = {cll5};
  cll6 = newreg; Line Loop(cll6) = {cl9,cl11,-cl12,-cl10}; Plane Surface(newreg) = {cll6};

  // We then store the surface loops identification numbers in a list
  // for later reference (we will need these to define the final
  // volume).
  theloops[t] = newreg;
  // Define outer surface.
  Surface Loop(theloops[t]) = {cll1+1,cll2+1,cll3+1,cll4+1,cll5+1,cll6+1};

Return

Function Torus
  // Given a zshift and two radii irad and orad, and a zshift, this
  // creates a torus parallel to the x-y-plane.
  // The points:
  tp1 = newp;
  Point(tp1) = {0,orad,zshift,lcar};
  tp2 = newp;
  Point(tp2) = {0,irad+orad,zshift,lcar};
  tp3 = newp;
  Point(tp3) = {0,orad,zshift+irad,lcar};
  tp4 = newp;
  Point(tp4) = {0,orad,zshift-irad,lcar};
  tp5 = newp;
  Point(tp5) = {0,-irad+orad,zshift,lcar};
  // One circle:
  tc1 = newreg;
  Circle(tc1) = {tp2,tp1,tp3};
  tc2 = newreg;
  Circle(tc2) = {tp3,tp1,tp5};
  tc3 = newreg;
  Circle(tc3) = {tp5,tp1,tp4};
  tc4 = newreg;
  Circle(tc4) = {tp4,tp1,tp2};

  //// The extrusion to the torus:
  //tll1 = newreg;
  //Line Loop(tll1) = {tc1,tc2,tc3,tc4};

  ts1[] = Extrude {{0,0,1}, {0,0,0}, 2*Pi/3}{Line{tc1};};
  ts2[] = Extrude {{0,0,1}, {0,0,0}, 2*Pi/3}{Line{tc2};};
  ts3[] = Extrude {{0,0,1}, {0,0,0}, 2*Pi/3}{Line{tc3};};
  ts4[] = Extrude {{0,0,1}, {0,0,0}, 2*Pi/3}{Line{tc4};};

  // Extrude those once more.
  ts5[] = Extrude {{0,0,1}, {0,0,0}, 2*Pi/3}{Line{ts1[0]};};
  ts6[] = Extrude {{0,0,1}, {0,0,0}, 2*Pi/3}{Line{ts2[0]};};
  ts7[] = Extrude {{0,0,1}, {0,0,0}, 2*Pi/3}{Line{ts3[0]};};
  ts8[] = Extrude {{0,0,1}, {0,0,0}, 2*Pi/3}{Line{ts4[0]};};

  // And one last time.
  ts9[] = Extrude {{0,0,1}, {0,0,0}, 2*Pi/3}{Line{ts5[0]};};
  ts10[] = Extrude {{0,0,1}, {0,0,0}, 2*Pi/3}{Line{ts6[0]};};
  ts11[] = Extrude {{0,0,1}, {0,0,0}, 2*Pi/3}{Line{ts7[0]};};
  ts12[] = Extrude {{0,0,1}, {0,0,0}, 2*Pi/3}{Line{ts8[0]};};

  // Now build the volume out of all those surfaces.
  // We then store the surface loops identification numbers in a list
  // for later reference (we will need these to define the final
  // volume).
  theloops[t] = newreg;
  Surface Loop(theloops[t]) = {ts1[1], ts2[1], ts3[1], ts4[1],
                               ts5[1], ts6[1], ts7[1], ts8[1],
                               ts9[1], ts10[1], ts11[1], ts12[1]};
  thetorus = newreg;
  Volume(thetorus) = {theloops[t]};

  Physical Volume(Sprintf("coil %g", t)) = thetorus;
Return


// Actual creation starts here.

// Tori.
t = 1;
lcar = 0.005;
orad = 0.1;
irad = 0.015;
zshift = 0.04;
Call Torus;
t = 2;
zshift = -0.04;
Call Torus;

// Brick.
t = 3;
lcar = 0.005;
xmin = -0.05;
xmax = 0.05;
ymin = -0.05;
ymax = 0.05;
zmin = -0.15;
zmax = 0.15;
Call Brick;
thebrick = newreg;
Volume(thebrick) = theloops[t];
//Physical Volume (t) = thebrick;
Physical Volume("brick") = thebrick;


// Outer cube.
t = 0;
lcar = 0.2;
xmin = -0.75;
xmax = 0.75;
ymin = -0.75;
ymax = 0.75;
zmin = -0.75;
zmax = 0.75;
Call Brick;
thebrick = newreg;
// The volume of the cube, without the cavities, is now defined by a number of
// surface loops: the first surface loop (t=0) defines the exterior surface;
// the surface loops other than the first one define holes.  (Again,
// to reference an array of variables, its identifier is followed by
// square brackets):
Volume(thebrick) = {theloops[]};
// We finally define a physical volume for the elements discretizing
// the cube, without the holes (whose elements were already tagged
// with numbers 1 to 5 in the `For' loop):
//Physical Volume (newreg) = thebrick;
Physical Volume ("air") = thebrick;
