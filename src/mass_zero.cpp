#include "mass_zero.hpp"

namespace mfem
{

SparseMatrix ToRowMatrix(LinearForm &lf)
{
   const int size = lf.FESpace()->GetTrueVSize();
   Array<int> row_ptr({0,size});
   Array<int> col_ind(size);
   std::iota(col_ind.begin(), col_ind.end(), int());

   lf.Assemble();
   double * data = lf.GetData();
   int *i, *j;
   row_ptr.StealData(&i);
   col_ind.StealData(&j);
   return SparseMatrix(i, j, data, size, 1, true, false, true);
}

HypreParMatrix *ToRowParMatrix(ParLinearForm &lf)
{
   ParFiniteElementSpace *pfes = lf.ParFESpace();
   // std::unique_ptr<int> i;
   // std::unique_ptr<HYPRE_BigInt> j;
   // std::unique_ptr<real_t> *d;
   Array<int> i; Array<HYPRE_BigInt> j; Vector d;
   Array<HYPRE_BigInt> cols;
   int local_siz = pfes->TrueVSize();
   HYPRE_BigInt global_siz = pfes->GlobalTrueVSize();
   int current;
   i.SetSize(local_siz+1); std::iota(i.begin(), i.end(), 0);
   j.SetSize(local_siz);
   if (HYPRE_AssumedPartitionCheck())
   {
      std::fill(j.begin(), j.end(), 0);
      cols.SetSize(2);
      cols[0] = 0;
      cols[1] = 1;
   }
   else
   {
      MFEM_ABORT("Not yet implemented");
   }
   d.SetSize(local_siz);

   lf.Assemble(); lf.ParallelAssemble(d);

   // return nullptr;
   return HypreParMatrix(pfes->GetComm(),
                         local_siz, global_siz, 1,
                         i.GetData(), j.GetData(), d.GetData(),
                         pfes->GetTrueDofOffsets(), cols.GetData()).Transpose();
}

MassZeroOperator::MassZeroOperator(Operator &op, LinearForm &mass_op,
                                   bool reassemble, int offset)
   : Operator(op.Height(), op.Width())
   , op(op), mass_op(mass_op)
   , reassemble(reassemble), offset(offset)
{
   if (!reassemble) { mass_op.Assemble(); domain_size = mass_op.Sum();}

#ifdef MFEM_USE_MPI
   ParLinearForm * p_mass_op = dynamic_cast<ParLinearForm*>(&mass_op);
   if (p_mass_op)
   {
      isParallel = true;
      comm = p_mass_op->ParFESpace()->GetComm();
      pmass_vec.SetSize(p_mass_op->ParFESpace()->GetTrueVSize());
      if (!reassemble)
      {
         p_mass_op->ParallelAssemble(pmass_vec);
         MPI_Allreduce(MPI_IN_PLACE, &domain_size, 1, MFEM_MPI_REAL_T, MPI_SUM, comm);
      }
   }
#endif
}

void MassZeroOperator::CorrectVolume(Vector &x) const
{
   if (reassemble) // reassemble if needed
   {
      mass_op.Assemble();
      domain_size = mass_op.Sum();
      if (isParallel)
      {
#ifdef MFEM_USE_MPI
         pmass_vec.SetSize(static_cast<ParLinearForm*>
                           (&mass_op)->ParFESpace()->GetTrueVSize());
         static_cast<ParLinearForm*>(&mass_op)->ParallelAssemble(pmass_vec);
         MPI_Allreduce(MPI_IN_PLACE, &domain_size, 1, MFEM_MPI_REAL_T, MPI_SUM, comm);
#endif
      }
   }

   Vector x_view(x, offset, mass_op.FESpace()->GetTrueVSize());
   if (isParallel)
   {
#ifdef MFEM_USE_MPI
      const real_t curr_volume = InnerProduct(comm, pmass_vec, x_view);
      x_view -= curr_volume / domain_size;
#endif
   }
   else
   {
      const real_t curr_volume = mass_op*x_view;
      x_view -= curr_volume / domain_size;
   }
}


}
