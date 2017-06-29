# -*- coding: utf-8 -*-
#
'''
Stabilization techniques for PDEs with dominating convection.
For an overview of methods, see :cite:`sold` and :cite:`bgs2004`.
'''
from dolfin import (
    Expression
    )


#
# SUPG stabilization as in the classical paper
#
#     Streamline upwind/Petrov-Galerkin formulations for
#     convection dominated flows with particular emphasis on the
#     incompressible Navier-Stokes equations;
#     A.N. Brooks, T.J.R. Hughes;
#     1982;
#     <http://ccpo.odu.edu/~tejada/USACM07/BrooksHughes.pdf>.
#
def supg(convection, diffusion, element_degree):
    cppcode = '''#include <dolfin/mesh/Vertex.h>

class SupgStab : public Expression {
public:
double sigma;
int p;
boost::shared_ptr<GenericFunction> convection;
boost::shared_ptr<Mesh> mesh;

SupgStab(): Expression(2)
{}

void eval(Array<double>& b_tau,
          const Array<double>& x,
          const ufc::cell& c
          ) const
{
  assert(b_tau.size() == x.size());
  convection->eval(b_tau, x, c);
  double conv_norm = 0.0;
  for (uint i = 0; i < b_tau.size(); ++i)
    conv_norm += b_tau[i]*b_tau[i];
  conv_norm = sqrt(conv_norm);

  if (conv_norm > DOLFIN_EPS)
  {
    Cell cell(*mesh, c.index);

    // Compute the directed diameter of the cell, cf.
    //
    //    On spurious oscillations at layers diminishing (SOLD) methods
    //    for convection-diffusion equations: Part II - Analysis for P1 and Q1
    //    finite elements;
    //    Volker John, Petr Knobloch;
    //    Comput. Methods Appl. Mech. Engrg. 197 (2008) 1997-2014.
    //
    //    diam(cell, s) = 2*||s|| / sum_{nodes n_i} |s.\\grad\\psi|
    //
    // where \\psi is the P_1 basis function of n_i.
    //
    const double area = cell.volume();
    const unsigned int* vertices = cell.entities(0);
    assert(vertices);

    double sum = 0.0;
    for (int i=0; i<3; i++)
    {
      for (int j=i+1; j<3; j++)
      {
        // Get edge coords.
        const dolfin::Vertex v0(*mesh, vertices[i]);
        const dolfin::Vertex v1(*mesh, vertices[j]);
        const Point p0 = v0.point();
        const Point p1 = v1.point();
        const double e0 = p0[0] - p1[0];
        const double e1 = p0[1] - p1[1];

        // Note that
        //
        //     \\grad\\psi = ortho_edge / edgelength / height
        //               = ortho_edge / (2*area)
        //
        // so
        //
        //   (v.\\grad\\psi) = (v.ortho_edge) / (2*area).
        //
        // Move the constant factors out of the summation.
        //
        // It would be really nice if we could just do
        //    edge.dot((-v[1], v[0]))
        // but unfortunately, edges just dot with other edges.
        sum += fabs(e1*b_tau[0] - e0*b_tau[1]);
      }
    }
    const double h = 4 * conv_norm * area / sum;

    //// The alternative for the lazy:
    //const double h = cell.diameter();

    // Just a little sanity check here.
    assert(h <= cell.diameter());

    const double Pe = 0.5*conv_norm * h/(p*sigma);
    assert(Pe > 0.0);
    double xi = 0.0;
    //if (Pe > DOLFIN_EPS)
    //  xi = 1.0/tanh(Pe) - 1.0/Pe;
    // lim_{a->0} (1.0/tanh(a) - 1.0/a)/a = 1/3.
    //xi = min(1.0, Pe/3.0);
    if (Pe > 1.0)
      xi = 1.0 - 1.0/Pe;

    const double tau = 0.5*h*xi / (p*conv_norm);

    if (tau > 1.0e3)
    {
      std::cout << "tau   = " << tau << std::endl;
      std::cout << "||b|| = " << conv_norm << std::endl;
      std::cout << "Pe    = " << Pe << std::endl;
      std::cout << "h     = " << h << std::endl;
      std::cout << "xi    = " << xi << std::endl;
      throw 1;
    }

    // tau*b:
    for (uint i = 0; i < b_tau.size(); ++i)
      b_tau[i] *= 0.5*h*xi / (p*conv_norm);
  }
  else
  {
    for (uint i = 0; i < b_tau.size(); ++i)
      b_tau[i] = 0.0;
  }
  // We don't need no else-clause here since the (negibly small) convection
  // is already stored in v.

  return;
}
};
'''
    # TODO set degree
    b_tau = Expression(cppcode, degree=5)
    b_tau.convection = convection
    b_tau.mesh = convection.function_space().mesh()
    b_tau.sigma = diffusion
    b_tau.p = element_degree
    return b_tau


#
# SUPG stabilization as in the classical paper
#
#     Streamline upwind/Petrov-Galerkin formulations for
#     convection dominated flows with particular emphasis on the
#     incompressible Navier-Stokes equations;
#     A.N. Brooks, T.J.R. Hughes;
#     1982;
#     <http://ccpo.odu.edu/~tejada/USACM07/BrooksHughes.pdf>.
#
def supg2(mesh, convection, diffusion_coefficient, element_degree):
    cppcode = '''#include <dolfin/mesh/Vertex.h>

class SupgStab : public Expression {
public:
double sigma;
int p;
std::shared_ptr<GenericFunction> convection;
std::shared_ptr<Mesh> mesh;

SupgStab(): Expression()
{}

void eval(Array<double>& tau,
          const Array<double>& x,
          const ufc::cell& c
          ) const
{
  Array<double> v(x.size());
  convection->eval(v, x, c);
  double conv_norm = 0.0;
  for (uint i = 0; i < v.size(); ++i)
    conv_norm += v[i]*v[i];
  conv_norm = sqrt(conv_norm);

  if (conv_norm < DOLFIN_EPS) {
    tau[0] = 0.0;
  }

  Cell cell(*mesh, c.index);

  // Compute the directed diameter of the cell, cf.
  //
  // On spurious oscillations at layers diminishing (SOLD) methods for
  // convection-diffusion equations: Part II - Analysis for P1 and Q1 finite
  // elements
  // Volker John, Petr Knobloch;
  // Comput. Methods Appl. Mech. Engrg. 197 (2008) 1997-2014.
  //
  // The diameter in a direction s is defined as
  //
  //    diam(cell, s) = 2*||s|| / sum_{nodes n_i} |s.\\grad\\psi|
  //
  // where \\psi is the P_1 basis function of n_i.
  //
  const double area = cell.volume();
  const unsigned int* vertices = cell.entities(0);
  assert(vertices);

  double sum = 0.0;
  for (int i=0; i<3; i++) {
    for (int j=i+1; j<3; j++) {
      // Get edge coords.
      const dolfin::Vertex v0(*mesh, vertices[i]);
      const dolfin::Vertex v1(*mesh, vertices[j]);
      const Point p0 = v0.point();
      const Point p1 = v1.point();
      const double e0 = p0[0] - p1[0];
      const double e1 = p0[1] - p1[1];

      // Note that
      //
      //   \\grad\\psi = ortho_edge / edgelength / height
      //               = ortho_edge / (2*area)
      //
      // so
      //
      //   (v.\\grad\\psi) = (v.ortho_edge) / (2*area).
      //
      // Move the constant factors out of the summation.
      //
      // It would be really nice if we could just do
      //    edge.dot((-v[1], v[0]))
      // but unfortunately, edges just dot with other edges.
      sum += fabs(e1*v[0] - e0*v[1]);
    }
  }
  const double h = 4 * conv_norm * area / sum;

  // // The alternative for the lazy:
  // const double h = cell.diameter();

  // // Just a little sanity check here.
  // const double eps = 1.0e-12;
  // if (h > cell.diameter() + eps) {
  //     std::cout << "The directed diameter h (" << h << ") "
  //               << "should not be larger than the actual cell diameter "
  //               << "(" << cell.diameter() << ")."
  //               << std::endl;
  // }
  // assert(h < cell.diameter() + eps);

  const double Pe = 0.5*conv_norm * h/(p*sigma);
  assert(Pe > 0.0);
  // We'd like to compute `xi = 1.0/tanh(Pe) - 1.0/Pe`. This expression can
  // hardly be evaluated for small Pe, see
  // <https://stackoverflow.com/a/43279491/353337>. Hence, use its Taylor
  // expansion around 0.
  const double xi = Pe > 1.0e-5 ?
      1.0/tanh(Pe) - 1.0/Pe :
      Pe/3.0 - Pe*Pe*Pe / 45.0 + 2.0/945.0 * Pe*Pe*Pe*Pe*Pe;

  tau[0] = 0.5*h*xi / (p*conv_norm);

  //if (tau[0] > 1.0e3)
  //{
  //  std::cout << "tau   = " << tau[0] << std::endl;
  //  std::cout << "||b|| = " << conv_norm << std::endl;
  //  std::cout << "Pe    = " << Pe << std::endl;
  //  std::cout << "h     = " << h << std::endl;
  //  std::cout << "xi    = " << xi << std::endl;
  //  //throw 1;
  //}

  return;
}
};
'''
    tau = Expression(cppcode, degree=1)
    tau.convection = convection
    tau.mesh = mesh  # convection.function_space().mesh()
    tau.sigma = diffusion_coefficient
    tau.p = element_degree

    return tau
