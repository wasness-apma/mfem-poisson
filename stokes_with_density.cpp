/// Poisson equation in MFEM
/// with pure Neumann boundary conditions

#include "mfem.hpp"
#include "src/helper.hpp"
#include "src/mass_zero.hpp"
#include <cmath>

using namespace mfem;
using namespace std;

const real_t pi = M_PI;

const real_t alpha_under = 1e-6;
const real_t alpha_over = 1e6;
const real_t q = 0.1;

real_t kappa = 3.0 * M_PI; // should be an integer multiple of pi so that p is zero sum.
real_t mu = 1.0;
void u_exact(const Vector &x, Vector & u);
// real_t p_exact(const Vector &x);
void f_exact(const Vector &x, Vector &f);

real_t density(const Vector &x);
real_t alpha(const real_t rho);

void print_array(Array<int> arr);
void print_array(Array<float> arr);
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

   L2_FECollection l2_coll(0, dim); 
   H1_FECollection h1_coll(order, dim);
   H1_FECollection h1_coll_op1(order + 1, dim); 

   // FiniteElementSpace *velocity_space = new FiniteElementSpace(mesh, hdiv_coll);
   // FiniteElementSpace velocity_space(&mesh, &h1_coll, dim);
   FiniteElementSpace velocity_space(&mesh, &h1_coll_op1, dim=dim); // velocity space
   FiniteElementSpace velocity_component_space(&mesh, &h1_coll_op1); // velocity space. Appears as derivative so need more regularity.
   FiniteElementSpace pressure_space(&mesh, &h1_coll); // pressure space
   FiniteElementSpace density_space(&mesh, &l2_coll);

   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;

   for (int lv = 0; lv < ref_levels; lv++)
   {
      mesh.UniformRefinement();

      velocity_space.Update();
      velocity_component_space.Update();
      pressure_space.Update();
      density_space.Update();

      const int dof_velocity = velocity_space.GetVSize();
      const int dof_pressure = pressure_space.GetVSize();
      const int dof_density = density_space.GetVSize();

      // Apply boundary conditions on all external boundaries:
      Array<int> ess_tdof_list;
      velocity_space.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);


      // Define Block offsets.
      // The first block is the momentum equation, the second block is the continuity equation, and the third block is average zero condition
      Array<int> block_offsets(4);
      block_offsets[0] = 0;
      block_offsets[1] = dof_velocity;
      block_offsets[2] = dof_pressure;
      block_offsets[3] = 1;
      block_offsets.PartialSum();

      // true solution vector. The mt part is a copy past.
      // MemoryType mt = device.GetMemoryType();
      BlockVector x(block_offsets), rhs(block_offsets);
      x = 0.0;
      rhs = 0.0;

      // Vector density_vector(dof_density);
      // density_vector = 0.0;

      std::cout << "***********************************************************\n";
      std::cout << "RT Space dofs = " << dof_velocity << "\n";
      std::cout << "H1 Space dofs = " << dof_pressure << "\n";
      std::cout << "Density Space dofs = " << dof_density << "\n";
      std::cout << "dim(Diffusion) = " << block_offsets[1] - block_offsets[0] << "\n";
      std::cout << "dim(Div) = " << block_offsets[2] - block_offsets[1] << "\n";
      std::cout << "dim(Zero) = " << block_offsets[3] - block_offsets[2] << "\n";
      std::cout << "***********************************************************\n\n";

      // GridFunction u, p;
      // u.MakeRef(&velocity_space, x.GetBlock(0), 0);
      // p.MakeRef(&pressure_space, x.GetBlock(1), 0);

      // u = 0.0;
      // p = 0.0;

      // separate this for the sake of later
      GridFunction rho(&density_space);
      rho = 0.0;
      FunctionCoefficient densityCoefficient(&density);
      rho.ProjectCoefficient(densityCoefficient); 


      GridFunction u(&velocity_space, x.GetBlock(0));
      VectorFunctionCoefficient bdrConditions(dim=dim, &u_exact);
      // Vector one_vec({1.0, 1.0});
      // VectorConstantCoefficient bdrConditions(one_vec);
      u.ProjectBdrCoefficient(bdrConditions, ess_bdr);

      GridFunction ux(&velocity_component_space, x.GetBlock(0));
      GridFunction uy(&velocity_component_space, x.GetBlock(0), velocity_component_space.GetVSize());
      

      GridFunction p(&pressure_space, x.GetBlock(1));

      // define the linear form, witout boundary conditions

      LinearForm load(&velocity_space, rhs.GetBlock(0).GetData());
      //fform->Update(R_space, rhs.GetBlock(0), 0);
      VectorFunctionCoefficient load_cf(dim, f_exact);
      load.AddDomainIntegrator(new VectorDomainLFIntegrator(load_cf));
      load.Assemble();

      // Define the bilinear form for mass integration and diffusion integration
      BilinearForm massDiffusionOperator(&velocity_space);

      ConstantCoefficient mu_coeff(mu);
      massDiffusionOperator.AddDomainIntegrator(new VectorDiffusionIntegrator(mu_coeff));

      GridFunctionCoefficient rho_coeff(&rho);
      TransformedCoefficient alpha_coeff(&rho_coeff, alpha);
      massDiffusionOperator.AddDomainIntegrator(new VectorMassIntegrator(alpha_coeff));

      massDiffusionOperator.Assemble();
      massDiffusionOperator.EliminateEssentialBC(ess_bdr, u, rhs.GetBlock(0));
      massDiffusionOperator.Finalize();


      // SparseMatrix diffusion_assembled;
      
      // diffusionOperator.FormSystemMatrix(ess_tdof_list, diffusion_assembled);

      MixedBilinearForm divergenceOperator(&velocity_space, &pressure_space);
      divergenceOperator.AddDomainIntegrator(new VectorDivergenceIntegrator);
      divergenceOperator.Assemble();
      divergenceOperator.EliminateTrialEssentialBC(ess_bdr, u, rhs.GetBlock(1));
      divergenceOperator.Finalize();

      SparseMatrix * transposeDivergenceOperator = Transpose(divergenceOperator.SpMat());

      LinearForm avg_zero(&pressure_space);
      ConstantCoefficient one_cf(1.0);
      avg_zero.AddDomainIntegrator(new DomainLFIntegrator(one_cf));
      avg_zero.Assemble();

      SparseMatrix linearFormZero = ToRowMatrix(avg_zero);
      // std::unique_ptr<SparseMatrix> linearFormZeroTranspose(Transpose(linearFormZero)); // unique_ptr basically manages ownership of pointer for me
      SparseMatrix* linearFormZeroTranspose; 
      linearFormZeroTranspose = Transpose(linearFormZero);

      // BlockOperator stokesOperator(block_offsets);
      // stokesOperator.SetBlock(0,0, &diffusionOperator.SpMat(), mu);
      // stokesOperator.SetBlock(0,1, transposeDivergenceOperator, 1.0);
      // stokesOperator.SetBlock(1,0, &divergenceOperator.SpMat(), 1.0);
      // stokesOperator.SetBlock(1,2, linearFormZeroTranspose, 1.0);
      // stokesOperator.SetBlock(2,1, &linearFormZero, 1.0);

      BlockMatrix stokesOperator(block_offsets);
      stokesOperator.SetBlock(0,0, &massDiffusionOperator.SpMat());
      stokesOperator.SetBlock(0,1, transposeDivergenceOperator);
      stokesOperator.SetBlock(1,0, &divergenceOperator.SpMat());
      stokesOperator.SetBlock(1,2, linearFormZeroTranspose);
      stokesOperator.SetBlock(2,1, &linearFormZero);

      SparseMatrix * stokesMatrix = stokesOperator.CreateMonolithic();

      // std::cout << "no segfault yet\n";

      // instead of mass zero operator, convert average zero into sparse matrix
      // MassZeroOperator mass_zero_op(stokesOperator, avg_zero);
      // mass_zero_op.CorrectVolume(load);

      // GMRESSolver solver;
      // solver.SetOperator(stokesOperator);
      // solver.SetRelTol(1e-10);
      // solver.SetAbsTol(1e-10);
      // solver.SetMaxIter(1e06);
      // solver.SetKDim(1000);
      // solver.SetPrintLevel(1);
      // solver.Mult(rhs, x);

      UMFPackSolver umf_solver;
      umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      umf_solver.SetOperator(*stokesMatrix);
      umf_solver.Mult(rhs, x);

         // 12. Create the grid functions u and p. Compute the L2 error norms.
      // GridFunction u, p;
      // u.MakeRef(&velocity_space, x.GetBlock(0), 0);
      // p.MakeRef(&pressure_space, x.GetBlock(1), 0);

      VectorFunctionCoefficient ucoeff(dim, u_exact);
      // FunctionCoefficient pcoeff(p_exact);

      real_t err = u.ComputeL2Error(ucoeff);
      // u.ProjectCoefficient(ucoeff);
      avg_zero.SetSize(dof_pressure);
      avg_zero.Assemble();
      real_t mass_err = avg_zero(p);

      // Print the solution
      std::cout << "Error : " << err << std::endl;
      std::cout << "Avg   : " << mass_err << std::endl;
      GLVis glvis("localhost", 19916, false);
      glvis.Append(u, "u");
      glvis.Append(ux, "ux");
      glvis.Append(uy, "uy");
      glvis.Append(p, "p");
      glvis.Append(rho, "rho");
      glvis.Update();

      delete stokesMatrix;
      delete linearFormZeroTranspose;
      delete transposeDivergenceOperator;
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

void print_array(Array<float> arr) {
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


const real_t minbound = 0.3;
const real_t maxbound = 0.7;
const real_t normalizer = pow(2 / (maxbound - minbound), 2);

real_t ux(const real_t y) {
   return std::max(0.0, -normalizer * (y - minbound) * (y - maxbound));
}
real_t ux_laplacian(const real_t y) {
   if (minbound < y && y < maxbound) {
      return -2 * normalizer;
   } else {
      return 0.0;
   }
}

// horizontal flow, constant
void u_exact(const Vector &x, Vector &u)
{
   // constant => divergence free
   u(0) = ux(x(1));
   u(1) = 0;
}

// // up to a constant
// real_t p_exact(const Vector &x)
// {
//    return -x(0) *(alpha(density(x)) * ux(x(1)) - mu * ux_laplacian(x(1)));
//    // return cos(kappa * x(0))*cos(kappa * x(1));
//    // return 0;
// }

void f_exact(const Vector &x, Vector &f)
{
   // f(0) = -mu * (-kappa * kappa * sin(kappa * x(1))) - kappa * sin(kappa * x(0)) * cos(kappa * x(1));
   // f(1) = -mu * (-kappa * kappa * sin(kappa * x(0))) - kappa * cos(kappa * x(0)) * sin(kappa * x(1));
   f(0) = 0;
   f(1) = 0;
   // const real_t x = xvec(0);
   // const real_t y = xvec(1);

   // f(0) = 2*pi*sin(2*pi*y)*(2*pow(pi,2)*pow(x,4) - 4*pow(pi,
   //                                2)*pow(x,3) + 2*pow(pi,2)*pow(x,2) - 6*pow(x,2) + 6*x - 1)
   //              - cos(x)*cos(y);
   // f(1) = sin(x)*sin(y) - 2*(2*x - 1)*(2*pow(pi,2)*cos(2*y*pi)*x - 2*pow(pi,
   //                                           2)*cos(2*y*pi)*pow(x,2) + 3*cos(2*y*pi) - 3);

}

real_t density(const Vector &x)
{
   if (minbound < x(1) && x(1) < maxbound) {
      return 0;
   } else {
      return 1;
   }
}

real_t alpha(const real_t rho) {
   if (rho < 0) {
      std::string errstring = "Rho value should not be negative: ";
      errstring += std::to_string(rho);
      throw std::runtime_error(errstring);
   }
   return alpha_under + (alpha_over - alpha_under) * rho * (1 + q) / (rho + q);
}


