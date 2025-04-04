#include "linearop.hpp"

namespace mfem
{
SparseMatrix * ToRowMatrix(LinearForm &lf)
{
   int width = lf.Size();
   lf.Assemble();
   SparseMatrix *mat = new SparseMatrix(1, width, width);
   Array<int> cols(width);
   std::iota(cols.begin(), cols.end(), 0);
   mat->SetRow(0, cols, lf);

   return mat;
}

#ifdef MFEM_USE_MPI
HypreParMatrix * ToParRowMatrix(ParLinearForm &lf)
{
   MFEM_ABORT("Not yet implemented")
   return nullptr;
}
#endif
} // namespace mfem
