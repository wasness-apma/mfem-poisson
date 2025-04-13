/// Poisson equation in MFEM
/// with pure Neumann boundary conditions

#include "mfem.hpp"
#include "src/helper.hpp"
#include "src/mass_zero.hpp"
#include <cmath>

using namespace mfem;
using namespace std;

real_t kappa = 4.0 * M_PI; // should be an integer multiple of pi so that p is zero sum.
real_t mu = 3.0;
void u_exact(const Vector &x, Vector & u);
real_t p_exact(const Vector &x);
void f_exact(const Vector &x, Vector &f);

void print_array(Array<int> arr);
// void multiplyOperator(Operator *op, int nrows, int ncols, BlockOperator*& blockOperator);

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
   int dim = mesh.Dimension();

   // Define finite element space. Need Dimension() + 1 to have a vector field u and pressure p
   // H1_FECollection fec(order, dim);  
   // FiniteElementSpace fes(&mesh, &fec);

   // FiniteElementCollection *hdiv_coll(new RT_FECollection(order, dim)); // used for u, and test vectors
   // FiniteElementCollection *h1_coll(new H1_FECollection(order, dim)); // used for p, and for test functions

   // RT_FECollection rt_col(order, dim);  
   H1_FECollection h1_coll(order, dim);

   // FiniteElementSpace *R_space = new FiniteElementSpace(mesh, hdiv_coll);
   // FiniteElementSpace R_space(&mesh, &h1_coll, dim);
   FiniteElementSpace R_space(&mesh, &h1_coll, dim=dim);
   FiniteElementSpace H_space(&mesh, &h1_coll);

   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 0;

   // Apply boundary conditions on all external boundaries:
   Array<int> ess_tdof_list;
   mesh.MarkExternalBoundaries(ess_bdr);
   R_space.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   for (int lv = 0; lv < ref_levels; lv++)
   {
      mesh.UniformRefinement();

      R_space.Update();
      const int dof_RT = R_space.GetVSize();

      H_space.Update();
      const int dof_H = H_space.GetVSize();

      // Define Block offsets.
      // The first block is the momentum equation, the second block is the continuity equation, and the third block is average zero condition
      Array<int> block_offsets(4);
      block_offsets[0] = 0;
      block_offsets[1] = dof_RT;
      block_offsets[2] = dof_H;
      block_offsets[3] = 1;
      block_offsets.PartialSum();
      BlockVector x(block_offsets), rhs(block_offsets);

      std::cout << "***********************************************************\n";
      std::cout << "RT Space dofs = " << dof_RT << "\n";
      std::cout << "H1 Space dofs = " << dof_H << "\n";
      std::cout << "dim(Diffusion) = " << block_offsets[1] - block_offsets[0] << "\n";
      std::cout << "dim(Div) = " << block_offsets[2] - block_offsets[1] << "\n";
      std::cout << "dim(Zero) = " << block_offsets[3] - block_offsets[2] << "\n";
      std::cout << "***********************************************************\n\n";

      GridFunction u, p;
      u.MakeRef(&R_space, x.GetBlock(0), 0);
      p.MakeRef(&H_space, x.GetBlock(1), 0);

      u = 0.0;
      p = 0.0;

      // Define the linear form
      // LinearForm load(&fes);
      // VectorFunctionCoefficient load_cf(dim, f_exact);
      // load.AddDomainIntegrator(new VectorDomainLFIntegrator(load_cf));
      // load.Assemble();

      LinearForm load(&R_space);
      VectorFunctionCoefficient load_cf(dim, f_exact);
      load.AddDomainIntegrator(new VectorDomainLFIntegrator(load_cf));
      load.Assemble();

      LinearForm zeroForm(&H_space);
      zeroForm.Assemble();

      BilinearForm diffusionOperator(&R_space);
      diffusionOperator.AddDomainIntegrator(new VectorDiffusionIntegrator(dim));
      diffusionOperator.Assemble();

      MixedBilinearForm divergenceOperator(&R_space, &H_space);
      divergenceOperator.AddDomainIntegrator(new VectorDivergenceIntegrator);
      divergenceOperator.Assemble();

      TransposeOperator transposeDivergenceOperator(&divergenceOperator);

      BlockOperator stokesOperator(block_offsets);
      stokesOperator.SetBlock(0,0, &diffusionOperator, mu);
      stokesOperator.SetBlock(0,1, &transposeDivergenceOperator, -1.0);
      stokesOperator.SetBlock(1,0, &divergenceOperator, 1.0);

      // ************************
      // Define the bilinear forms
      // BilinearForm diffusion(&fes);
      // diffusion.AddDomainIntegrator(new VectorDiffusionIntegrator(dim));
      // diffusion.Assemble();
      // diffusion.Finalize();

      // Array<int> stackedDiffusionIntegrationOffsets(dim + 1);
      // stackedDiffusionIntegrationOffsets[0] = 0;
      // for (int d = 0; d < dim; d++) {
      //    stackedDiffusionIntegrationOffsets[1 + d] = dof;
      // }
      // stackedDiffusionIntegrationOffsets.PartialSum();
      
      // BlockOperator stackedDiffusionIntegrator(stackedDiffusionIntegrationOffsets);
      // for (int d1 = 0; d1 < dim; d1++) {
      //    for (int d2 = 0; d2 < dim; d2++){
      //       stackedDiffusionIntegrator.SetBlock(d1, d2, &diffusion);
      //    }
      // }
      // BlockOperator* stackedDiffusionIntegrator;
      // multiplyOperator(&diffusion, dim, dim, stackedDiffusionIntegrator);
      // std::cout << "Created Diffusion Integrator.\n";

      // ************************

      /*
      // ******************* 
      // Divergence Integrator
      BilinearForm div(&fes);
      div.AddDomainIntegrator(new VectorFEDivergenceIntegrator());
      div.Assemble();
      div.Finalize();

      // Array<int> stackedDivColumnIntegrationOffsets(dim + 1);
      // stackedDivColumnIntegrationOffsets[0] = 0;
      // for (int d = 0; d < dim; d++) {
      //    stackedDivColumnIntegrationOffsets[1 + d] = dof;
      // }
      // stackedDivColumnIntegrationOffsets.PartialSum();
      

      // Array<int> stackedDivRowIntegrationOffsets(2);
      // stackedDivRowIntegrationOffsets[0] = 0;
      // stackedDivRowIntegrationOffsets[1] = dof;
      // stackedDivRowIntegrationOffsets.PartialSum();

      // BlockOperator stackedDivIntegrator(stackedDivRowIntegrationOffsets, stackedDivColumnIntegrationOffsets);
      // for (int d = 0; d < dim; d++) {
      //    stackedDivIntegrator.SetBlock(0, d, &div);
      // }
      BlockOperator* stackedDivIntegrator;
      multiplyOperator(&div, 1, 2, stackedDivIntegrator);
      std::cout << "Created Div Integrator.\n";

      TransposeOperator transposeDivIntegrator = new TransposeOperator(div);
      BlockOperator* stackedTransposeDivIntegrator;
      multiplyOperator(&transposeDivIntegrator, 2, 1, stackedTransposeDivIntegrator);
      std::cout << "Created Transpose Div Integrator.\n";
      // ***************

      BlockOperator stokesOp(block_offsets);

      std::cout << "***********************************************************\n";
      std::cout << "Diffusion Row Offsets: ";
      print_array(stackedDiffusionIntegrator -> RowOffsets());
      std::cout << "\n";

      std::cout << "Diffusion Column Offsets: ";
      print_array(stackedDiffusionIntegrator -> ColOffsets());
      std::cout << "\n\n";

      std::cout << "Div Row Offsets: ";
      print_array(stackedDivIntegrator -> RowOffsets());
      std::cout << "\n\n";

      std::cout << "Div Column Offsets: ";
      print_array(stackedDivIntegrator -> ColOffsets());
      std::cout << "\n\n";

      std::cout << "dim(Load Vector) = " << load.Size() << "\n";
      std::cout << "***********************************************************\n";

      std::cout << "Setting blocks.\n";
      stokesOp.SetBlock(0, 0, stackedDiffusionIntegrator, mu);
      std::cout << "Set block (0, 0).\n";
      stokesOp.SetBlock(1, 0, stackedDivIntegrator, -1.0);
      std::cout << "Set block (1, 0).\n";
      stokesOp.SetBlock(0, 1, stackedTransposeDivIntegrator, -1.0);
      std::cout << "Set block (0, 1).\n";
      */


      LinearForm avg_zero(&H_space);
      ConstantCoefficient one_cf(1.0);
      avg_zero.AddDomainIntegrator(new DomainLFIntegrator(one_cf));
      avg_zero.Assemble();

      MassZeroOperator mass_zero_op(stokesOperator, avg_zero);
      mass_zero_op.CorrectVolume(load);

      CGSolver solver;
      solver.SetOperator(mass_zero_op);
      solver.SetRelTol(1e-10);
      solver.SetAbsTol(1e-10);
      solver.SetMaxIter(1e06);
      solver.SetPrintLevel(0);
      solver.Mult(load, u);

      VectorFunctionCoefficient ucoeff(dim, u_exact);
      FunctionCoefficient pcoeff(p_exact);

      real_t err = u.ComputeL2Error(ucoeff);
      avg_zero.SetSize(dof_H);
      avg_zero.Assemble();
      real_t mass_err = avg_zero(u);

      // Print the solution
      std::cout << "Error : " << err << std::endl;
      std::cout << "Avg   : " << mass_err << std::endl;
      GLVis glvis("localhost", 19916, false);
      glvis.Append(u, "u");
      glvis.Append(p, "p");
      glvis.Update();
   }
   return 0;
}

void print_array(Array<int> arr) {
   int len = sizeof(arr) / sizeof(*arr);
   for (int i = 0; i < len; i++) {
      if (i < len - 1)
         std::cout << arr[i] << " ";
      else
         std::cout << arr[i];
   }
}

// void multiplyOperator(Operator *op, int nrows, int ncols, BlockOperator *&blockOperator) {
//       int oprows = op -> NumRows();
//       int opcols = op -> NumCols(); 

//       Array<int> stackeColOffsets(ncols + 1);
//       stackeColOffsets[0] = 0;
//       for (int d = 0; d < ncols; d++) {
//          stackeColOffsets[1 + d] = opcols;
//       }
//       stackeColOffsets.PartialSum();

//       Array<int> stackedRowOffsets(nrows + 1);
//       stackedRowOffsets[0] = 0;
//       for (int d = 0; d < nrows; d++) {
//          stackedRowOffsets[1 + d] = oprows;
//       }
//       stackedRowOffsets.PartialSum();

//       blockOperator = new BlockOperator(stackedRowOffsets, stackeColOffsets);
//       for (int dr = 0; dr < nrows; dr++) {
//          for (int dc = 0; dc < ncols; dc++) {
//             blockOperator -> SetBlock(dr, dc, op);
//          }
//       }
// }

void u_exact(const Vector &x, Vector &u)
{
   // note divergence free
   u(0) = sin(kappa * x(1));
   u(1) = sin(kappa * x(0));
}

real_t p_exact(const Vector &x)
{
   return cos(kappa * x(0))*cos(kappa * x(1));
}

void f_exact(const Vector &x, Vector &f)
{
   f(0) = -mu * (-kappa * kappa * sin(kappa * x(1))) - kappa * sin(kappa * x(0)) * cos(kappa * x(1));
   f(1) = -mu * (-kappa * kappa * sin(kappa * x(0))) - kappa * cos(kappa * x(0)) * sin(kappa * x(1));
}


