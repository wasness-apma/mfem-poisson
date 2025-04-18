//                                MFEM Stokes TO [THIS IS A COPY-PASTE JOB]
//
// Compile with: make ex37
//
// Sample runs:
//     ex37 -alpha 10
//     ex37 -alpha 10 -pv
//     ex37 -lambda 0.1 -mu 0.1
//     ex37 -o 2 -alpha 5.0 -mi 50 -vf 0.4 -ntol 1e-5
//     ex37 -r 6 -o 1 -alpha 25.0 -epsilon 0.02 -mi 50 -ntol 1e-5
//
// Description: This example code demonstrates the use of MFEM to solve a
//              density-filtered [3] topology optimization problem. The
//              objective is to minimize the compliance
//
//                  minimize ∫_Ω f⋅u dx over u ∈ [H¹(Ω)]² and ρ ∈ L¹(Ω)
//
//                  subject to
//
//                    -Div(r(ρ̃)Cε(u)) = f       in Ω + BCs
//                    -ϵ²Δρ̃ + ρ̃ = ρ             in Ω + Neumann BCs
//                    0 ≤ ρ ≤ 1                 in Ω
//                    ∫_Ω ρ dx = θ vol(Ω)
//
//              Here, r(ρ̃) = ρ₀ + ρ̃³ (1-ρ₀) is the solid isotropic material
//              penalization (SIMP) law, C is the elasticity tensor for an
//              isotropic linearly elastic material, ϵ > 0 is the design
//              length scale, and 0 < θ < 1 is the volume fraction.
//
//              The problem is discretized and gradients are computing using
//              finite elements [1]. The design is optimized using an entropic
//              mirror descent algorithm introduced by Keith and Surowiec [2]
//              that is tailored to the bound constraint 0 ≤ ρ ≤ 1.
//
//              This example highlights the ability of MFEM to deliver high-
//              order solutions to inverse design problems and showcases how
//              to set up and solve PDE-constrained optimization problems
//              using the so-called reduced space approach.
//
// [1] Andreassen, E., Clausen, A., Schevenels, M., Lazarov, B. S., & Sigmund, O.
//     (2011). Efficient topology optimization in MATLAB using 88 lines of
//     code. Structural and Multidisciplinary Optimization, 43(1), 1-16.
// [2] Keith, B. and Surowiec, T. (2023) Proximal Galerkin: A structure-
//     preserving finite element method for pointwise bound constraints.
//     arXiv:2307.12444 [math.NA]
// [3] Lazarov, B. S., & Sigmund, O. (2011). Filters in topology optimization
//     based on Helmholtz‐type differential equations. International Journal
//     for Numerical Methods in Engineering, 86(6), 765-781.

#include "mfem.hpp"
#include "stokes_to.hpp"
#include "src/helper.hpp"
#include "src/mass_zero.hpp"
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;

/**
 * @brief Bregman projection of ρ = sigmoid(ψ) onto the subspace
 *        ∫_Ω ρ dx = θ vol(Ω) as follows:
 *
 *        1. Compute the root of the R → R function
 *            f(c) = ∫_Ω sigmoid(ψ + c) dx - θ vol(Ω)
 *        2. Set ψ ← ψ + c.
 *
 * @param psi a GridFunction to be updated
 * @param target_volume θ vol(Ω)
 * @param tol Newton iteration tolerance
 * @param max_its Newton maximum iteration number
 * @return real_t Final volume, ∫_Ω sigmoid(ψ)
 */
real_t proj(GridFunction &psi, real_t target_volume, real_t domain_volume, real_t tol=1e-12,
            int max_its=10)
{
    GridFunctionCoefficient psi_coeff(&psi);

    const real_t  psimax = psi.Normlinf();
    const real_t volume_proportion = target_volume / domain_volume;

    real_t a = inv_sigmoid(volume_proportion) - psimax; // lower bound of 0
    real_t b = inv_sigmoid(volume_proportion) + psimax; // upper bound of 0

    ConstantCoefficient aCoefficient(a);
    ConstantCoefficient bCoefficient(b);

    SumCoefficient psiA(psi_coeff, aCoefficient);
    SumCoefficient psiB(psi_coeff, bCoefficient);

    TransformedCoefficient sigmoid_psi_a(&psiA, sigmoid);
    TransformedCoefficient sigmoid_psi_b(&psiB, sigmoid);

    LinearForm int_sigmoid_psi_a(psi.FESpace());
    int_sigmoid_psi_a.AddDomainIntegrator(new DomainLFIntegrator(sigmoid_psi_a));
    int_sigmoid_psi_a.Assemble();
    real_t a_vol_minus = int_sigmoid_psi_a.Sum() - target_volume;

    LinearForm int_sigmoid_psi_b(psi.FESpace());
    int_sigmoid_psi_b.AddDomainIntegrator(new DomainLFIntegrator(sigmoid_psi_b));
    int_sigmoid_psi_b.Assemble();
    real_t b_vol_minus = int_sigmoid_psi_b.Sum() - target_volume;

   bool done = false;
   real_t x;
   real_t x_vol;

   for (int k=0; k<max_its; k++) // Newton iteration
   {
        // std::cout << "\n iter: " << k << "\n";
        x = b - b_vol_minus * (b - a) / (b_vol_minus - a_vol_minus);

        LinearForm int_sigmoid_psi_x(psi.FESpace());
        ConstantCoefficient xCoefficient(x);
        SumCoefficient psiX(psi_coeff, xCoefficient);
        TransformedCoefficient sigmoid_psi_x(&psiX, sigmoid);
        int_sigmoid_psi_x.AddDomainIntegrator(new DomainLFIntegrator(sigmoid_psi_x));
        int_sigmoid_psi_x.Assemble();
        x_vol = int_sigmoid_psi_x.Sum();

        real_t x_vol_minus = x_vol - target_volume;

        if (b_vol_minus * x_vol_minus < 0) {
            a = b;
            a_vol_minus = b_vol_minus;   
        } else {
            a_vol_minus = a_vol_minus / 2;
        }
        b = x;
        b_vol_minus = x_vol_minus;

        if (abs(x_vol_minus) < tol) {
            // std::cout << "done, breaking, x_vol_minus, tol: " << x_vol_minus << " _ " << tol;
            done = true;
            break;
        }
   }

    psi += x;
    if (!done)
    {
        mfem_warning("Projection reached maximum iteration without converging. "
                    "Result may not be accurate.");
    }
//    int_sigmoid_psi.Assemble();
   return x_vol;
}

/*
 * ---------------------------------------------------------------
 *                      ALGORITHM PREAMBLE - THIS IS INACCURATE FOR NOW, WE USE DIFFERENT PHYSICS
 * ---------------------------------------------------------------
 *
 *  The Lagrangian for this problem is
 *
 *          L(u,ρ,ρ̃,w,w̃) = (f,u) - (r(ρ̃) C ε(u),ε(w)) + (f,w)
 *                       - (ϵ² ∇ρ̃,∇w̃) - (ρ̃,w̃) + (ρ,w̃)
 *
 *  where
 *
 *    r(ρ̃) = ρ₀ + ρ̃³ (1 - ρ₀)       (SIMP rule)
 *
 *    ε(u) = (∇u + ∇uᵀ)/2           (symmetric gradient)
 *
 *    C e = λtr(e)I + 2μe           (isotropic material)
 *
 *  NOTE: The Lame parameters can be computed from Young's modulus E
 *        and Poisson's ratio ν as follows:
 *
 *             λ = E ν/((1+ν)(1-2ν)),      μ = E/(2(1+ν))
 *
 * ---------------------------------------------------------------
 *
 *  Discretization choices:
 *
 *     u ∈ V ⊂ (H¹)ᵈ (order p)
 *     ψ ∈ L² (order p - 1), ρ = sigmoid(ψ)
 *     ρ̃ ∈ H¹ (order p)
 *     w ∈ V  (order p)
 *     w̃ ∈ H¹ (order p)
 *
 * ---------------------------------------------------------------
 *                          ALGORITHM
 * ---------------------------------------------------------------
 *
 *  Update ρ with projected mirror descent via the following algorithm.
 *
 *  1. Initialize ψ = inv_sigmoid(vol_fraction) so that ∫ sigmoid(ψ) = θ vol(Ω)
 *
 *  While not converged:
 *
 *     2. Solve filter equation ∂_w̃ L = 0; i.e.,
 *
 *           (ϵ² ∇ ρ̃, ∇ v ) + (ρ̃,v) = (ρ,v)   ∀ v ∈ H¹.
 *
 *     3. Solve primal problem ∂_w L = 0; i.e.,
 *
 *      (λ r(ρ̃) ∇⋅u, ∇⋅v) + (2 μ r(ρ̃) ε(u), ε(v)) = (f,v)   ∀ v ∈ V.
 *
 *     NB. The dual problem ∂_u L = 0 is the negative of the primal problem due to symmetry.
 *
 *     4. Solve for filtered gradient ∂_ρ̃ L = 0; i.e.,
 *
 *      (ϵ² ∇ w̃ , ∇ v ) + (w̃ ,v) = (-r'(ρ̃) ( λ |∇⋅u|² + 2 μ |ε(u)|²),v)   ∀ v ∈ H¹.
 *
 *     5. Project the gradient onto the discrete latent space; i.e., solve
 *
 *                         (G,v) = (w̃,v)   ∀ v ∈ L².
 *
 *     6. Bregman proximal gradient update; i.e.,
 *
 *                            ψ ← ψ - αG + c,
 *
 *     where α > 0 is a step size parameter and c ∈ R is a constant ensuring
 *
 *                     ∫_Ω sigmoid(ψ - αG + c) dx = θ vol(Ω).
 *
 *  end
 */

void u_bdry(const Vector &x, Vector & u);
void f(const Vector &x, Vector &f);
real_t alpha_cutoff_func(const real_t rho);
real_t alpha_cutoff_func_derivative(const real_t rho);
real_t clamp(const real_t psi);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
    int ref_levels = 5;
    int order = 1;
    real_t alpha = 5.0;
    real_t vol_fraction = 1./3.;
    int max_it = 1e3;
    real_t itol = 1e-1;
    real_t ntol = 1e-4;
    real_t rho_min = 1e-6;
    real_t mu = 1.0;

    real_t delta = 1.5;

    bool glvis_visualization = true;
    bool paraview_output = true;

    OptionsParser args(argc, argv);
    args.AddOption(&ref_levels, "-r", "--refine",
                    "Number of times to refine the mesh uniformly.");
    args.AddOption(&order, "-o", "--order",
                    "Order (degree) of the finite elements.");
    args.AddOption(&alpha, "-alpha", "--alpha-step-length",
                    "Step length for gradient descent.");
    args.AddOption(&max_it, "-mi", "--max-it",
                    "Maximum number of gradient descent iterations.");
    args.AddOption(&ntol, "-ntol", "--rel-tol",
                    "Normalized exit tolerance.");
    args.AddOption(&itol, "-itol", "--abs-tol",
                    "Increment exit tolerance.");
    args.AddOption(&vol_fraction, "-vf", "--volume-fraction",
                    "Volume fraction for the material density.");
    args.AddOption(&mu, "-mu", "--mu",
                    "Lamé constant μ.");
    args.AddOption(&rho_min, "-rmin", "--psi-min",
                    "Minimum of density coefficient.");
    args.AddOption(&glvis_visualization, "-vis", "--visualization", "-no-vis",
                    "--no-visualization",
                    "Enable or disable GLVis visualization.");
    args.AddOption(&paraview_output, "-pv", "--paraview", "-no-pv",
                    "--no-paraview",
                    "Enable or disable ParaView output.");
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(mfem::out);
        return 1;
    }
    args.PrintOptions(mfem::out);

    Mesh mesh = Mesh::MakeCartesian2D(1.0, 1.0, mfem::Element::Type::QUADRILATERAL,
                                        true, 1.0 * delta, 1.0);

    int dim = mesh.Dimension();

    // 2. Set BCs.
    Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 1;

    // 3. Refine the mesh.
    for (int lev = 0; lev < ref_levels; lev++)
    {
        mesh.UniformRefinement();
    }

    // 4. Define the necessary finite element spaces on the mesh.
    H1_FECollection velocity_fec(order + 1, dim); // space for u
    H1_FECollection pressure_fec(order, dim); // space for p
    L2_FECollection control_fec(0, dim,
                                BasisType::GaussLobatto); // space for ψ

    FiniteElementSpace velocity_fes(&mesh, &velocity_fec, dim=dim);
    FiniteElementSpace pressure_fes(&mesh, &pressure_fec);
    FiniteElementSpace control_fes(&mesh, &control_fec);
    FiniteElementSpace velocity_component_space(&mesh, &velocity_fec); // component projection space

    velocity_fes.Update();
    pressure_fes.Update();
    control_fes.Update();
    velocity_component_space.Update();


    int velocity_size = velocity_fes.GetTrueVSize();
    int pressure_size = pressure_fes.GetTrueVSize();
    int control_size = control_fes.GetTrueVSize();
    mfem::out << "Number of velocity unknowns: " << velocity_size << std::endl;
    mfem::out << "Number of pressure unknowns: " << pressure_size << std::endl;
    mfem::out << "Number of control unknowns: " << control_size << std::endl;

    // Apply boundary conditions on all external boundaries:
    Array<int> ess_tdof_list;
    velocity_fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    Array<int> block_offsets(4);
    block_offsets[0] = 0;
    block_offsets[1] = velocity_size;
    block_offsets[2] = pressure_size;
    block_offsets[3] = 1;
    block_offsets.PartialSum();
    BlockVector x(block_offsets), rhs(block_offsets);
    x = 0.0;
    rhs = 0.0;

    // 5. Set the initial guess for ρ.
    GridFunction u(&velocity_fes, x.GetBlock(0));
    // GridFunction dudrho(&velocity_fes);
    GridFunction p(&pressure_fes, x.GetBlock(1));
    GridFunction psi(&control_fes);
    GridFunction psi_old(&control_fes);
    u = 0.0;
    // dudrho = 0.0;
    p = 0.0;
    psi = inv_sigmoid(vol_fraction);

    psi_old = inv_sigmoid(vol_fraction);

    // ρ = sigmoid(ψ)
    MappedGridFunctionCoefficient rho(&psi, sigmoid);
    // Interpolation of ρ = sigmoid(ψ) in control fes (for ParaView output)
    GridFunction rho_gf(&control_fes);
    // ρ - ρ_old = sigmoid(ψ) - sigmoid(ψ_old)
    VectorGridFunctionCoefficient u_coeff(&u);
    DiffMappedGridFunctionCoefficient succ_diff_rho(&psi, &psi_old, sigmoid);

    // add boundary conditions for u
    VectorFunctionCoefficient bdrConditions(dim=dim, &u_bdry);
    u.ProjectBdrCoefficient(bdrConditions, ess_bdr);


    // 6. Set-up the physics solver.

    // and projections for u
    GridFunction ux(&velocity_component_space, x.GetBlock(0));
    GridFunction uy(&velocity_component_space, x.GetBlock(0), velocity_component_space.GetVSize());

    LinearForm load(&velocity_fes, rhs.GetBlock(0).GetData());
    VectorFunctionCoefficient load_cf(dim, f);
    load.AddDomainIntegrator(new VectorDomainLFIntegrator(load_cf));
    load.Assemble();

    // used to update density
    Vector previousRHSFromMassDiffusion(velocity_size), currentRHSFromMassDiffusion(velocity_size);
    previousRHSFromMassDiffusion = 0.0;
    currentRHSFromMassDiffusion = 0.0;

    MixedBilinearForm divergenceOperator(&velocity_fes, &pressure_fes);
    divergenceOperator.AddDomainIntegrator(new VectorDivergenceIntegrator);
    divergenceOperator.Assemble();
    divergenceOperator.EliminateTrialEssentialBC(ess_bdr, u, rhs.GetBlock(1));
    divergenceOperator.Finalize();

    SparseMatrix * transposeDivergenceOperator = Transpose(divergenceOperator.SpMat());

    LinearForm avg_zero(&pressure_fes);
    ConstantCoefficient one_cf(1.0);
    avg_zero.AddDomainIntegrator(new DomainLFIntegrator(one_cf));
    avg_zero.Assemble();

    SparseMatrix linearFormZero = ToRowMatrix(avg_zero);
    SparseMatrix* linearFormZeroTranspose; 
    linearFormZeroTranspose = Transpose(linearFormZero);

    BlockMatrix stokesOperator(block_offsets);
    stokesOperator.SetBlock(0,1, transposeDivergenceOperator);
    stokesOperator.SetBlock(1,0, &divergenceOperator.SpMat());
    stokesOperator.SetBlock(1,2, linearFormZeroTranspose);
    stokesOperator.SetBlock(2,1, &linearFormZero);

    // 8. Define the gradient function.
    GridFunction grad(&control_fes);

    // 9. Define some tools for later.
    ConstantCoefficient one(1.0);
    ConstantCoefficient zero(0.0);
    GridFunction onegf(&control_fes);
    onegf = 1.0;
    GridFunction zerogf(&control_fes);
    zerogf = 0.0;
    LinearForm vol_form(&control_fes);
    vol_form.AddDomainIntegrator(new DomainLFIntegrator(one));
    vol_form.Assemble();
    real_t domain_volume = vol_form(onegf);
    const real_t target_volume = domain_volume * vol_fraction;

    BilinearForm mass(&control_fes);
    mass.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator(one)));
    mass.Assemble();
    SparseMatrix M;
    Array<int> empty;
    mass.FormSystemMatrix(empty,M);

    // 10. Connect to GLVis. Prepare for VisIt output.
    char vishost[] = "localhost";
    int  visport   = 19916;
    socketstream sout_r;
    socketstream sout_u;
    if (glvis_visualization)
    {
        sout_r.open(vishost, visport);
        sout_r.precision(8);

        sout_u.open(vishost, visport);
        sout_u.precision(8);
    }

    mfem::ParaViewDataCollection paraview_dc("ex37", &mesh);
    if (paraview_output)
    {
        rho_gf.ProjectCoefficient(rho);
        paraview_dc.SetPrefixPath("ParaView");
        paraview_dc.SetLevelsOfDetail(order);
        paraview_dc.SetDataFormat(VTKFormat::BINARY);
        paraview_dc.SetHighOrderOutput(true);
        paraview_dc.SetCycle(0);
        paraview_dc.SetTime(0.0);
        paraview_dc.RegisterField("velocities",&u);
        paraview_dc.RegisterField("density",&rho_gf);
        paraview_dc.Save();
    }

    //  GLVis glvis("localhost", 19916, false);

    // glvis.Append(grad, "grad");
    // glvis.Append(psi, "psi");
    // glvis.Append(rho_gf, "rho");
    // glvis.Append(u, "u");

    // 11. Iterate:
    for (int k = 1; k <= max_it; k++)
    {
        if (k > 1) { alpha *= ((real_t) k) / ((real_t) k-1); } // divergent sequence

        mfem::out << "\nStep = " << k << std::endl;

        // Step 1 - State solve
        mfem::out << "Forming mass-diffusion matrix." << std::endl;

        // BilinearForm *massDiffusionOperator;
        BilinearForm massDiffusionOperator(&velocity_fes);
        ConstantCoefficient mu_coeff(mu);
        massDiffusionOperator.AddDomainIntegrator(new VectorDiffusionIntegrator(mu_coeff));
        TransformedCoefficient alpha_coeff(&rho, alpha_cutoff_func);
        massDiffusionOperator.AddDomainIntegrator(new VectorMassIntegrator(alpha_coeff));
        massDiffusionOperator.Assemble();

        massDiffusionOperator.EliminateEssentialBC(ess_bdr, u, currentRHSFromMassDiffusion);
        rhs.GetBlock(0).Add(-1.0, previousRHSFromMassDiffusion);
        rhs.GetBlock(0).Add(1.0, currentRHSFromMassDiffusion);
        previousRHSFromMassDiffusion = currentRHSFromMassDiffusion;
        currentRHSFromMassDiffusion = 0.0;

        massDiffusionOperator.Finalize();
        stokesOperator.SetBlock(0,0, &massDiffusionOperator.SpMat());
        SparseMatrix * stokesMatrix = stokesOperator.CreateMonolithic();    

        mfem::out << "Sovling physics." << std::endl;
        UMFPackSolver umf_solver;
        umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
        umf_solver.SetOperator(*stokesMatrix);
        umf_solver.Mult(rhs, x);

        delete stokesMatrix;

        // Step 4 - Compute gradient
        // Solve G = M⁻¹w̃
        mfem::out << "Computing gradient." << std::endl;
        LinearForm w_rhs(&control_fes);

        TransformedCoefficient alpha_derivative_coeff(&rho, alpha_cutoff_func_derivative);
        InnerProductCoefficient usquared_coeff(u_coeff, u_coeff);
        ConstantCoefficient half(0.5);
        ProductCoefficient alpha_times_usquared_coeff(alpha_derivative_coeff, usquared_coeff);
        ProductCoefficient half_times_alpha_times_usquared_coeff(half, alpha_times_usquared_coeff);
        w_rhs.AddDomainIntegrator(new DomainLFIntegrator(half_times_alpha_times_usquared_coeff));
        w_rhs.Assemble();
        M.Mult(w_rhs,grad);

        // mfem::out << "Projecting and updating glvis." << std::endl;
        // rho_gf.ProjectCoefficient(rho);
        // glvis.Update();

        // Step 5 - Update design variable ψ ← proj(ψ - αG)
        mfem::out << "Updating mirror and projecting." << std::endl;
        psi.Add(-alpha, grad);
        const real_t material_volume = proj(psi, target_volume, domain_volume, ntol=ntol, max_it=max_it);

        // Compute ||ρ - ρ_old|| in control fes.
        real_t norm_increment = zerogf.ComputeL1Error(succ_diff_rho);
        real_t norm_reduced_gradient = norm_increment/alpha;
        psi_old = psi;

        if (norm_reduced_gradient != norm_reduced_gradient) {
            throw std::abort;
        }

        // real_t compliance = (*(ElasticitySolver->GetLinearForm()))(u);
        mfem::out << "norm of the reduced gradient = " << norm_reduced_gradient <<
                std::endl;
        mfem::out << "norm of the increment = " << norm_increment << endl;
        // mfem::out << "compliance = " << compliance << std::endl;
        mfem::out << "volume fraction = " << material_volume / domain_volume <<
                std::endl;


        // throw std::abort;

        if (glvis_visualization)
        {
            GridFunction r_gf(&control_fes);
            r_gf.ProjectCoefficient(rho);
            sout_r << "solution\n" << mesh << r_gf
                << "window_title 'Design density r(ρ)'" << flush;

            sout_u << "solution\n" << mesh << u
                << "window_title 'Design physics u'" << flush;
        }

        if (paraview_output)
        {
            rho_gf.ProjectCoefficient(rho);
            paraview_dc.SetCycle(k);
            paraview_dc.SetTime((real_t)k);
            paraview_dc.Save();
        }

        if (norm_reduced_gradient < ntol && norm_increment < itol)
        {
            break;
        }
    }

    return 0;
}


// constants from the dolphin_adjoint paper
const real_t alpha_under = 2.5 / (pow(100.0, 2));
const real_t alpha_over = 2.5 / (pow(0.01, 2));
const real_t q = 0.1;

// for numerical stability, keep psi constrained
real_t psi_maxmin = 5.0;

void u_bdry(const Vector &x, Vector & u) {
    float y = x(1);
    // u(0) = std::max(0.0, -normalizer * (y - minbound) * (y - maxbound));
    u(0) = (
        std::max(0.0,  1 - 144. * pow(y - 3./4., 2)) + 
        std::max(0.0,  1 - 144. * pow(y - 1./4., 2))
    );
    u(1) = 0.0;
}

void f(const Vector &x, Vector &f) {
    f(0) = 0.0;
    f(1) = 0.0;
}

real_t alpha_cutoff_func(const real_t rho) {
    return alpha_over + (alpha_under - alpha_over) * rho * (1 + q) / (rho + q);
}
real_t alpha_cutoff_func_derivative(const real_t rho) {
    return (alpha_under - alpha_over) * q * (1 + q) / pow(rho + q, 2);
}

real_t clamp(const real_t psi) {
    return std::min(std::max(psi, -psi_maxmin), psi_maxmin);
}