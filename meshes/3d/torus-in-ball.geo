// Gmsh geometry for a torus embedded in a ball.
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
  // The extrusion to the torus:
  tll1 = newreg;
  Line Loop(tll1) = {tc1,tc2,tc3,tc4};

  ts1[] = Extrude {{0,0,1}, {0.0,0.0,0.0}, 2*Pi/3}{Line{tc1};};
  ts2[] = Extrude {{0,0,1}, {0.0,0.0,0.0}, 2*Pi/3}{Line{tc2};};
  ts3[] = Extrude {{0,0,1}, {0.0,0.0,0.0}, 2*Pi/3}{Line{tc3};};
  ts4[] = Extrude {{0,0,1}, {0.0,0.0,0.0}, 2*Pi/3}{Line{tc4};};

  // Extrude those once more.
  ts5[] = Extrude {{0,0,1}, {0.0,0.0,0.0}, 2*Pi/3}{Line{ts1[0]};};
  ts6[] = Extrude {{0,0,1}, {0.0,0.0,0.0}, 2*Pi/3}{Line{ts2[0]};};
  ts7[] = Extrude {{0,0,1}, {0.0,0.0,0.0}, 2*Pi/3}{Line{ts3[0]};};
  ts8[] = Extrude {{0,0,1}, {0.0,0.0,0.0}, 2*Pi/3}{Line{ts4[0]};};

  // And one last time.
  ts9[] = Extrude {{0,0,1}, {0.0,0.0,0.0}, 2*Pi/3}{Line{ts5[0]};};
  ts10[] = Extrude {{0,0,1}, {0.0,0.0,0.0}, 2*Pi/3}{Line{ts6[0]};};
  ts11[] = Extrude {{0,0,1}, {0.0,0.0,0.0}, 2*Pi/3}{Line{ts7[0]};};
  ts12[] = Extrude {{0,0,1}, {0.0,0.0,0.0}, 2*Pi/3}{Line{ts8[0]};};

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

Function Ball

  // two points for an arc
  tp1 = newp;
  Point(tp1) = {0.0,0.0,0.0-rad,lcar};
  tp2 = newp;
  Point(tp2) = {0.0+rad,0.0,0.0,lcar};
  tp3 = newp;
  Point(tp3) = {0.0,0.0,0.0+rad,lcar};
  // center
  tpc = newp;
  Point(tpc) = {0.0,0.0,0.0,lcar};

  // semi-circles
  c1 = newreg;
  Circle(c1) = {tp1,tpc,tp2};
  c2 = newreg;
  Circle(c2) = {tp2,tpc,tp3};

  // Extrude in three steps.
  ts1[] = Extrude {{0,0,1}, {0.0,0.0,0.0}, 2*Pi/3}{Line{c1};};
  ts2[] = Extrude {{0,0,1}, {0.0,0.0,0.0}, 2*Pi/3}{Line{c2};};

  ts3[] = Extrude {{0,0,1}, {0.0,0.0,0.0}, 2*Pi/3}{Line{ts1[0]};};
  ts4[] = Extrude {{0,0,1}, {0.0,0.0,0.0}, 2*Pi/3}{Line{ts2[0]};};

  ts5[] = Extrude {{0,0,1}, {0.0,0.0,0.0}, 2*Pi/3}{Line{ts3[0]};};
  ts6[] = Extrude {{0,0,1}, {0.0,0.0,0.0}, 2*Pi/3}{Line{ts4[0]};};

  // create a volume
  theloops[t] = newreg;
  Surface Loop(theloops[t]) = {ts1[1], ts2[1], ts3[1], ts4[1],
                               ts5[1], ts6[1]};
Return

lcar = 0.1;

// Create torus.
t = 1;
orad = 0.5;
irad = 0.1;
zshift = 0.0;
Call Torus;

// Enclosing ball.
t = 0;
rad = 5.0;
lcar *= 10;
Call Ball;
theball = newreg;
Volume(theball) = {theloops[]};
Physical Volume("air") = theball;
