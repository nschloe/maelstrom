// Geometry as specified in
//
//     Correct voltage distribution for axisymmetric sinusoidal modeling of
//     induction heating with prescribed current, voltage, or power;
//     O. Klein, P. Philip;
//     IEEE Transaction of Magnetic, vol 38, no. 3, 2002.
//
// ----------------------------------------------------------------------------
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
  ccl1 = newc;
  Line(ccl1) = {cp1,cp2};
  ccl2 = newc;
  Line(ccl2) = {cp2,cp3};
  ccl3 = newc;
  Line(ccl3) = {cp3,cp4};
  ccl4 = newc;
  Line(ccl4) = {cp4,cp1};

  myloops[0] = newreg;
  Line Loop(myloops[0]) = {ccl1,ccl2,ccl3,ccl4};

  // Save for cutting out.
  theloops[t] = myloops[0];

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
  ccl5 = newc;
  Line(ccl5) = {cp5,cp6};
  ccl6 = newc;
  Line(ccl6) = {cp6,cp7};
  ccl7 = newc;
  Line(ccl7) = {cp7,cp8};
  ccl8 = newc;
  Line(ccl8) = {cp8,cp5};

  myloops[1] = newreg;
  Line Loop(myloops[1]) = {ccl5,ccl6,ccl7,ccl8};

  // Make the interior a surface.
  air = news;
  Plane Surface(air) = {myloops[1]};
  Physical Surface(Sprintf("air %g", t)) = air;

  // Coil.
  surf = news;
  Plane Surface(surf) = {myloops[]};
  Physical Surface(Sprintf("coil %g", t)) = surf;

  // Refinement around the boundaries.
  b_id = newf;
  Field[b_id] = BoundaryLayer;
  Field[b_id].EdgesList = {ccl1,ccl2,ccl3,ccl4,ccl5,ccl6,ccl7,ccl8};
  Field[b_id].hfar = lcar_far;
  Field[b_id].hwall_n = lcar_b;
  Field[b_id].hwall_t = lcar_b;
  Field[b_id].ratio = 1.1;
  Field[b_id].thickness = w_b0;
  Field[b_id].AnisoMax = 100.0;

  thefields[s] = b_id;
Return
// ----------------------------------------------------------------------------
z = 0.0;
lcar = 1.0e-3;
lcar_far = 1.0e-1;

// Indices for fields, line loops.
s = 0;
t = 0;
// ----------------------------------------------------------------------------
// Coils.
// Material parameters for copper.
omega = 2*Pi*10.0e3;
T = 293.0;
sigma = 1.0 / (-3.033e-9 + 68.85e-12*T - 6.72e-15*T^2 + 8.56e-18*T^3);
mu0 = Pi * 4.0e-7;
mu = 0.999994*mu0;
// Layer width according to <https://en.wikipedia.org/wiki/Skin_effect>:
w_b0 = Sqrt(2.0/(mu*sigma*omega));
Printf("boundary layer width: %e", w_b0);
// Allow for about 2*k elements normal to the boundary layer.
k = 5;
lcar_b = w_b0 / k;

// Place the coils.
xmin = 0.126;
xmax = 0.136;
ymin = -0.04;
ymax = -0.02;
d = 0.001;
For k In {1:9}
    s += 1;
    t += 1;
    Call NestedRectangle;
    ymin += 0.03;
    ymax += 0.03;
EndFor
// ----------------------------------------------------------------------------
// The apparatus.
x0 = 0.0;
y0 = 0.028;
x1 = 0.052;
y1 = 0.21;
x2 = 0.018;
y2 = 0.17;
cp1 = newp;
Point(cp1) = {x0,y0,z,lcar};
cp2 = newp;
Point(cp2) = {x1,y0,z,lcar};
cp3 = newp;
Point(cp3) = {x1,y1,z,lcar};
//cp4 = newp;
//Point(cp4) = {x0,y1,z,lcar};
cp5 = newp;
Point(cp5) = {x2,y1,z,lcar};
cp5a = newp;
Point(cp5a) = {x2,y2+0.017,z,lcar};
cp5b = newp;
Point(cp5b) = {x2-0.01,y2+0.017,z,lcar};
cp5c = newp;
Point(cp5c) = {x2-0.01,y2,z,lcar};
//cp6 = newp;
//Point(cp6) = {x2,y2,z,lcar};
cp7 = newp;
Point(cp7) = {x0,y2,z,lcar};
// Cavity.
cp8 = newp;
Point(cp8) = {x0,y2-0.015,z,lcar};
//cp9 = newp;
//Point(cp9) = {x0+0.044,y2-0.015,z,lcar};
cp9a = newp;
Point(cp9a) = {x0+0.040,y2-0.015,z,lcar};
cp9b = newp;
Point(cp9b) = {x0+0.046,y2-0.009,z,lcar};
cp9c = newp;
Point(cp9c) = {x0+0.046,y2-0.030,z,lcar};
cp9d = newp;
Point(cp9d) = {x0+0.044,y2-0.030,z,lcar};
cp10 = newp;
Point(cp10) = {x0+0.044,y2-0.048,z,lcar};
cp11 = newp;
Point(cp11) = {x0,y2-0.048,z,lcar};

// Apparatus lines.
cl1 = newc;
Line(cl1) = {cp1,cp2};
cl2 = newc;
Line(cl2) = {cp2,cp3};
cl3 = newc;
Line(cl3) = {cp3,cp5};
cl3a = newc;
Line(cl3a) = {cp5,cp5a};
cl3b = newc;
Line(cl3b) = {cp5a,cp5b};
cl3c = newc;
Line(cl3c) = {cp5b,cp5c};
//cl4 = newc;
//Line(cl4) = {cp5c,cp6};
cl5 = newc;
Line(cl5) = {cp5c,cp7};
cl6 = newc;
Line(cl6) = {cp7,cp8};
cl7 = newc;
Line(cl7) = {cp8,cp9a};
cl7a = newc;
Line(cl7a) = {cp9a,cp9b};
cl7b = newc;
Line(cl7b) = {cp9b,cp9c};
cl7c = newc;
Line(cl7c) = {cp9c,cp9d};
cl8 = newc;
Line(cl8) = {cp9d,cp10};
cl9 = newc;
Line(cl9) = {cp10,cp11};
cl10 = newc;
Line(cl10) = {cp11,cp1};

app = newreg;
Line Loop(app) = {cl1,cl2,cl3,cl3a,cl3b,cl3c,cl5,cl6,cl7,cl7a,cl7b,cl7c,cl8,cl9,cl10};
cs = news;
Plane Surface(cs) = {app};
Physical Surface("crucible") = cs;
// ----------------------------------------------------------------------------
// Melt.
cm1 = newc;
Line(cm1) = {cp8,cp11};
llm = newreg;
Line Loop(llm) = {cl7,cl7,cl7a,cl7b,cl7c,cl8,cl9,-cm1};
sm = news;
Plane Surface(sm) = {llm};
Physical Surface("melt") = sm;
// ----------------------------------------------------------------------------
// Hold-all domain.
y00 = -0.8;
y11 = 1.0;
x11 = 1.2;
o1 = newp;
Point(o1) = {x0,y00,z,lcar_far};
o2 = newp;
Point(o2) = {x11,y00,z,lcar_far};
o3 = newp;
Point(o3) = {x11,y11,z,lcar_far};
o4 = newp;
Point(o4) = {x0,y11,z,lcar_far};

// Lines.
ol1 = newc;
Line(ol1) = {cp1,o1};
ol2 = newc;
Line(ol2) = {o1,o2};
ol3 = newc;
Line(ol3) = {o2,o3};
ol4 = newc;
Line(ol4) = {o3,o4};
ol5 = newc;
Line(ol5) = {o4,cp7};

// Surface.
t = 0;
theloops[t] = newreg;
Line Loop(theloops[t]) = {cl1,cl2,cl3,cl3a,cl3b,cl3c,cl5,-ol5,-ol4,-ol3,-ol2,-ol1};
air = news;
Plane Surface(air) = {theloops[]};
Physical Surface("air") = air;
// ----------------------------------------------------------------------------
//background_field_id = newf;
//Field[background_field_id] = Min;
//Field[background_field_id].FieldsList = {thefields[]};
//Background Field = background_field_id;
//
//Mesh.CharacteristicLengthExtendFromBoundary = 0;
//
//// Decrease the precision of 1D integration. If set to default, mesh generation
//// may take very long.
//Mesh.LcIntegrationPrecision = 1.0e-3;
// ----------------------------------------------------------------------------
