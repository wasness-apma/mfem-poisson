#include "mfem.hpp"

namespace mfem
{
SparseMatrix * ToRowMatrix(LinearForm &lf);

#ifdef MFEM_USE_MPI
HypreParMatrix * ToParRowMatrix(ParLinearForm &lf);
#endif
} // namespace mfem
