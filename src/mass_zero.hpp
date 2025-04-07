#include "mfem.hpp"

namespace mfem
{

SparseMatrix ToRowMatrix(LinearForm &lf);
HypreParMatrix ToColParMatrix(ParLinearForm &lf);

class MassZeroOperator : public Operator
{
   // member variables
private:
   Operator &op;
   LinearForm &mass_op;
   bool reassemble = false;
   const int offset;
   bool isParallel = false;
#ifdef MFEM_USE_MPI
   mutable Vector pmass_vec;
   MPI_Comm comm;
#endif
protected:
public:
   // member functions
private:
protected:
public:
   MassZeroOperator(Operator &op, LinearForm &mass_op, bool reassemble=false,
                    int offset=0);

   void CorrectVolume(Vector &x) const;

   void Mult(const Vector &x, Vector &y) const override
   {
      op.Mult(x, y);
      CorrectVolume(y);
   }

   void MultTranspose(const Vector &x, Vector &y) const override
   {
      op.MultTranspose(x, y);
      CorrectVolume(y);
   }
};
} // namespace mfem
