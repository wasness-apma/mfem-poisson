/// Poisson equation in MFEM
/// with pure Neumann boundary conditions

#include "mfem.hpp"
#include "src/helper.hpp"
#include "src/mass_zero.hpp"

using namespace mfem;
using namespace std;

int main(int argc, char *argv[])
{
   int order = 1;
   int ref_levels = 1;
   bool vis = false;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Polynomial order for the finite element space.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of uniform refinements.");
   args.AddOption(&vis, "-v", "--visualize", "--no-vis", "--no-visualization",
                  "-v 1 to visualize the solution.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   MFEM_ASSERT(order >= 1, "Order must be at least 1.");

   // Initialize MFEM
   Mesh mesh = mfem::Mesh::MakeCartesian2D(8, 8, Element::Type::TRIANGLE, true);

   // Define finite element space
   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);

   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 0;

   for (int lv = 0; lv < ref_levels; lv++)
   {
      mesh.UniformRefinement();
      fes.Update();
      const int dof = fes.GetVSize();

      // Define Block offsets.
      // The first block is the Poisson equation, and the second block is average zero condition
      Array<int> offsets({0, dof, 1});
      offsets.PartialSum();

      GridFunction u(&fes);
      u = 0.0;

      // Define the linear form
      LinearForm load(&fes);
      FunctionCoefficient load_cf([](const Vector &x) { return 2*M_PI*M_PI*cos(M_PI*x[0])*cos(M_PI*x[1]); });
      load.AddDomainIntegrator(new DomainLFIntegrator(load_cf));
      load.Assemble();

      // Define the bilinear forms
      BilinearForm diffusion(&fes);
      diffusion.AddDomainIntegrator(new DiffusionIntegrator());
      diffusion.Assemble();
      diffusion.Finalize();

      LinearForm avg_zero(&fes);
      ConstantCoefficient one_cf(1.0);
      avg_zero.AddDomainIntegrator(new DomainLFIntegrator(one_cf));
      avg_zero.Assemble();

      MassZeroOperator mass_zero_op(diffusion, avg_zero);
      mass_zero_op.CorrectVolume(load);

      CGSolver solver;
      solver.SetOperator(mass_zero_op);
      solver.SetRelTol(1e-10);
      solver.SetAbsTol(1e-10);
      solver.SetMaxIter(1e06);
      solver.SetPrintLevel(0);
      solver.Mult(load, u);

      FunctionCoefficient exact_solution([](const Vector &x)
      {
         return cos(M_PI*x[0])*cos(M_PI*x[1]);
      });

      real_t err = u.ComputeL2Error(exact_solution);
      avg_zero.SetSize(dof);
      avg_zero.Assemble();
      real_t mass_err = avg_zero(u);

      // Print the solution
      std::cout << "Error : " << err << std::endl;
      std::cout << "Avg   : " << mass_err << std::endl;
      GLVis glvis("localhost", 19916, false);
      glvis.Append(u, "u");
      glvis.Update();
   }
   return 0;
}
