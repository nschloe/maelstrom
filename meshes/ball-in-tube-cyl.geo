//
// Half-circle in a box.
// If rotated around x=0, this geometry corresponds to a ball in a cylindrical
// tube.
//
z = 0;
lcar = 4.0e-2;

xmin = 0.0;
xmax = 1.0;
ymin = 0.0;
ymax = 5.0;
ball_radius = 0.5;
ball_y = 1.5;

// Define the points for the double rectangle.
cp1 = newp;
Point(cp1) = {xmin,ymin,z,lcar};
cp2 = newp;
Point(cp2) = {xmax,ymin,z,lcar};
cp3 = newp;
Point(cp3) = {xmax,ymax,z,lcar};
cp4 = newp;
Point(cp4) = {xmin,ymax,z,lcar};

cp5 = newp;
Point(cp5) = {xmin,ball_y + ball_radius,z,lcar};
cp6 = newp;
Point(cp6) = {xmin,ball_y,z,lcar};
cp7 = newp;
Point(cp7) = {xmin+ball_radius,ball_y,z,lcar};
cp8 = newp;
Point(cp8) = {xmin,ball_y - ball_radius,z,lcar};

// Lines.
cl1 = newc;
Line(cl1) = {cp1,cp2};
cl2 = newc;
Line(cl2) = {cp2,cp3};
cl3 = newc;
Line(cl3) = {cp3,cp4};
cl4 = newc;
Line(cl4) = {cp4,cp5};
// Half-circle.
tc1 = newc;
Circle(tc1) = {cp5,cp6,cp7};
tc2 = newc;
Circle(tc2) = {cp7,cp6,cp8};
// Close curve.
cl5 = newc;
Line(cl5) = {cp8,cp1};

// Make it all a surface.
t = 0;
theloops[t] = newreg;
Line Loop(theloops[t]) = {cl1,cl2,cl3,cl4,tc1,tc2,cl5};
liquid = news;
Plane Surface(liquid) = {theloops[]};
Physical Surface(Sprintf("liquid")) = liquid;
