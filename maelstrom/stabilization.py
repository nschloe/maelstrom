# -*- coding: utf-8 -*-
#
"""
Stabilization techniques for PDEs with dominating convection.
The classical article about SUPG is :cite:`brooks`; for an overview
of methods, see :cite:`sold1`, :cite:`sold2`, and :cite:`bgs2004`.
"""
from dolfin import CompiledExpression, compile_cpp_code


def supg(mesh, convection, diffusion, element_degree):
    """For each cell, this function return the expression

    ..math::

        \\begin{align*}
        \\tau &= \\frac{h}{2\\|b\\|}
        \\left(\\frac{1}{\\tanh Pe} - \\frac{1}{Pe}\\right)\\\\
        & = \\frac{h^2}{4\\varepsilon} \\frac{1}{Pe}
        \\left(\\frac{1}{\\tanh Pe} - \\frac{1}{Pe}\\right)
        \\end{align*}

    with the element diameter in the direction of the convection vector
    :math:`b` and the Péclet number :math:`Pe = \\frac{\\|b\\|
    h}{2\\varepsilon}`; see (3) in :cite:`sold2`.
    Note that :math:`\\tau` does not have a singularity for :math:`\\|b\\|=0`
    since

    ..math::

        \\frac{1}{\\tanh Pe} - \\frac{1}{Pe} = \\frac{1}{3}Pe + O(Pe^3)

    for :math:`Pe\\approx 0`. This Taylor expansion (with a few more terms) is
    made use of in the code.
    """
    cppcode = """
#include <dolfin/common/Array.h>
#include <dolfin/function/Expression.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <iostream>


class SupgStab : public dolfin::Expression {
public:
SupgStab(): dolfin::Expression()
{}

void eval(
  dolfin::Array<double>& tau,
  const dolfin::Array<double>& x,
  const ufc::cell& c
  ) const
{
  dolfin::Array<double> v(x.size());
  convection->eval(v, x, c);
  double conv_norm = 0.0;
  for (uint i = 0; i < v.size(); ++i)
    conv_norm += v[i]*v[i];
  conv_norm = sqrt(conv_norm);

  dolfin::Cell cell(*mesh, c.index);

  //// The alternative for the lazy:
  //const double h = cell.diameter();

  // Compute the directed diameter of the cell, cf. :cite:`sold2`.
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
      const dolfin::Point p0 = v0.point();
      const dolfin::Point p1 = v1.point();
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
      sum += fabs(e1*v[0] - e0*v[1]);
    }
  }
  const double h = 4 * conv_norm * area / sum;

  // Just a little sanity check here.
  assert(h <= cell.h());

  const double Pe = 0.5*conv_norm * h/(p*epsilon);
  assert(Pe > 0.0);

  // We'd like to compute `xi = (1.0/tanh(Pe) - 1.0/Pe) / Pe`. This expression
  // can hardly be evaluated for small Pe, see
  // <https://stackoverflow.com/a/43279491/353337>. Hence, use its Taylor
  // expansion around 0.
  const double xi = Pe > 1.0e-5 ?
      (1.0/tanh(Pe) - 1.0/Pe) / Pe :
      1.0/3.0 - Pe*Pe / 45.0 + 2.0/945.0 * Pe*Pe*Pe*Pe;
  // const double xi =  (Pe > 1.0 ? 1.0 - 1.0/Pe : 0.0) / Pe;

  tau[0] = h*h / 4 / epsilon / p * xi;

  if (tau[0] > 1.0e3)
  {
    std::cout << "tau   = " << tau[0] << std::endl;
    std::cout << "||b|| = " << conv_norm << std::endl;
    std::cout << "Pe    = " << Pe << std::endl;
    std::cout << "h     = " << h << std::endl;
    std::cout << "xi    = " << xi << std::endl;
    throw 1;
  }
  return;
}

void set_convection(
    std::shared_ptr<GenericFunction> _convection
    )
{
  convection = _convection;
}

void set_mesh(
    std::shared_ptr<dolfin::Mesh> _mesh
    )
{
  mesh = _mesh;
}

void set_epsilon(double _epsilon)
{
  epsilon = _epsilon;
}

void set_p(int _p)
{
  p = _p;
}

private:
    std::shared_ptr<GenericFunction> convection;
    std::shared_ptr<dolfin::Mesh> mesh;
    double epsilon;
    int p;
};

PYBIND11_MODULE(SIGNATURE, m)
{
    py::class_<SupgStab, std::shared_ptr<SupgStab>, dolfin::Expression>
    (m, "SupgStab")
    .def(py::init<>())
    .def("set_convection", &SupgStab::set_convection)
    .def("set_mesh", &SupgStab::set_mesh)
    .def("set_epsilon", &SupgStab::set_epsilon)
    .def("set_p", &SupgStab::set_p);
}
"""
    # TODO set degree
    compiled_module = compile_cpp_code(cppcode).SupgStab()
    # compiled_module.set(convection, mesh, diffusion, element_degree)
    compiled_module.set_p(element_degree)
    compiled_module.set_epsilon(diffusion)
    compiled_module.set_mesh(mesh)
    compiled_module.set_convection(convection.cpp_object())
    tau = CompiledExpression(compiled_module, degree=5)
    return tau
