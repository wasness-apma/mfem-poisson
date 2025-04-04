#include <iomanip>
#include <iostream>
#include <vector>
#include "mfem.hpp"

namespace mfem
{

class GLVis
{
   Array<mfem::socketstream *> sockets;
   Array<mfem::GridFunction *> gfs;
   Array<Mesh *> meshes;
   bool parallel;
   const char *hostname;
   const int port;
   bool secure;

public:
#ifdef MFEM_USE_GNUTLS
   static const bool secure_default = true;
#else
   static const bool secure_default = false;
#endif
   GLVis(const char hostname[], int port, bool parallel,
         bool secure = secure_default)
      : sockets(0), gfs(0), meshes(0), parallel(parallel), hostname(hostname),
        port(port), secure(secure_default) {}

   ~GLVis()
   {
      for (socketstream *socket : sockets)
      {
         if (socket)
         {
            delete socket;
         }
      }
   }

   void Append(GridFunction &gf, const char window_title[] = nullptr,
               const char keys[] = nullptr);
   void Update();
   socketstream &GetSocket(int i)
   {
      return *sockets[i];
   }
};

} // namespace mfem
