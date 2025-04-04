#include "helper.hpp"

namespace mfem
{
void GLVis::Append(GridFunction &gf, const char window_title[],
                   const char keys[])
{
   sockets.Append(new socketstream(hostname, port, secure));
   socketstream *socket = sockets.Last();
   if (!socket->is_open())
   {
      return;
   }
   Mesh *mesh = gf.FESpace()->GetMesh();
   gfs.Append(&gf);
   meshes.Append(mesh);
   socket->precision(8);
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      *socket << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n";
   }
#endif
   *socket << "solution\n" << *mesh << gf;
   if (keys)
   {
      *socket << "keys " << keys << "\n";
   }
   if (window_title)
   {
      *socket << "window_title '" << window_title <<"'\n";
   }
   *socket << std::flush;
}

void GLVis::Update()
{
   for (int i=0; i<sockets.Size(); i++)
   {
      if (!sockets[i]->is_open() && !sockets[i]->good())
      {
         continue;
      }
#ifdef MFEM_USE_MPI
      if (parallel)
      {
         *sockets[i] << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() <<
                     "\n";
      }
#endif
      *sockets[i] << "solution\n" << *meshes[i] << *gfs[i];
      *sockets[i] << std::flush;
   }
}
}
