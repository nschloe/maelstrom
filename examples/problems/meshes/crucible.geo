// Domain as presented in
//
//     Mathematical modeling of Czochralski-type growth processes for
//     semicondictor bulk single crystals;
//     W. Dreyer, P.-E. Druet, O. Klein, J. Sprekels.
//


z = 0.0;

lcar = 6.0e-4;

crucible_width = 0.001;

// Assume that the outer outline of the crucible is defined by the ellipse
//
//     ((x-x0)/(x1-x0))^2 + ((y-y0)/(y1-y0))^2 = 1.
//
// with
x0 = 0.0;
x1 = 0.076;
y0 = 0.388;
y1 = 0.365;
y2 = 0.418;
y3 = 0.411;

center = newp;
Point(center) = {x0, y0, z};

// outer ellipse
tp1 = newp;
Point(tp1) = {x0, y1, z, lcar};
tp2 = newp;
Point(tp2) = {x1, y0, z, lcar};
tc1 = newc;
Ellipse(tc1) = {tp1, center, tp2, tp2};

// inner ellipse
tp3 = newp;
Point(tp3) = {x0, y1+crucible_width, z, lcar};
tp4 = newp;
Point(tp4) = {x1-crucible_width, y0, z, lcar};
tc2 = newc;
Ellipse(tc2) = {tp3, center, tp4, tp4};

// Extend and close.
tp5 = newp;
Point(tp5) = {x1, y2, z, lcar};
tp6 = newp;
Point(tp6) = {x1 - crucible_width, y2, z, lcar};
tp7 = newp;
Point(tp7) = {x1 - crucible_width, y3, z, lcar};
cl1 = newc;
Line(cl1) = {tp2,tp5};
cl2 = newc;
Line(cl2) = {tp5,tp6};
cl3 = newc;
Line(cl3) = {tp6,tp7};
cl4 = newc;
Line(cl4) = {tp7,tp4};
cl5 = newc;
Line(cl5) = {tp1,tp3};

// Define crucible surface.
ll1 = newreg;
Line Loop(ll1) = {tc1, cl1, cl2, cl3, cl4, -tc2, -cl5};
crucible = news;
Plane Surface(crucible) = {ll1};
Physical Surface(Sprintf("crucible")) = crucible;

// The boron oxide.
lcar2 = 4 * lcar;
x2 = 0.04;
tp8 = newp;
Point(tp8) = {x2, y3, z, lcar2};
tp9 = newp;
Point(tp9) = {x2, y2, z, lcar2};
cl6 = newc;
Line(cl6) = {tp6,tp9};
cl7 = newc;
Line(cl7) = {tp9,tp8};
cl8 = newc;
Line(cl8) = {tp8,tp7};
ll2 = newreg;
Line Loop(ll2) = {cl6, cl7, cl8, -cl3};
boron = news;
Plane Surface(boron) = {ll2};
Physical Surface(Sprintf("boron")) = boron;

// The crystal.
tp10 = newp;
Point(tp10) = {x0, y3, z, lcar2};
tp11 = newp;
Point(tp11) = {x0, y2, z, lcar2};
cl9 = newc;
Line(cl9) = {tp9,tp11};
cl10 = newc;
Line(cl10) = {tp11,tp10};
cl11 = newc;
Line(cl11) = {tp10,tp8};
ll3 = newreg;
Line Loop(ll3) = {cl9, cl10, cl11, -cl7};
crystal = news;
Plane Surface(crystal) = {ll3};
Physical Surface(Sprintf("crystal")) = crystal;

// The melt.
cl12 = newc;
Line(cl12) = {tp10,tp3};
ll4 = newreg;
Line Loop(ll4) = {cl12, tc2, -cl4, -cl8, -cl11};
melt = news;
Plane Surface(melt) = {ll4};
Physical Surface(Sprintf("melt")) = melt;
