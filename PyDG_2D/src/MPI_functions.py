import numpy as np
import mpi4py as MPI

def sendEdges(main,var):
  if (main.num_processes == 1):
    var.uU_edge[:] = var.uD[:,:,:,0 ]
    var.uD_edge[:] = var.uU[:,:,:,-1]
  else:
    main.comm.Send(var.uD[:,:,:,0].flatten(),dest=main.rank_connect[0],tag=main.mpi_rank)
    main.comm.Recv(var.edge_tmp[:],source=main.rank_connect[1],tag=main.rank_connect[1])
    var.uU_edge[:] = np.reshape(var.edge_tmp,(var.nvars,var.quadpoints,main.Npx))
    var.edge_tmp[:] = 0. 
    main.comm.Send(var.uU[:,:,:,-1].flatten(),dest=main.rank_connect[1],tag=main.mpi_rank*10)
    main.comm.Recv(var.edge_tmp,source=main.rank_connect[0],tag=main.rank_connect[0]*10)
    var.uD_edge[:] = np.reshape(var.edge_tmp,(var.nvars,var.quadpoints,main.Npx))

def gatherSol(main,eqns,var):
  if (main.mpi_rank == 0):
    uG = np.zeros((var.nvars,var.quadpoints,var.quadpoints,main.Nel[0],main.Nel[1]))
    uG[:,:,:,:,0:(main.mpi_rank+1)*main.Npy] = var.u[:]
    for i in range(1,main.num_processes):
      loc_rank = i
      data = np.zeros(np.shape(var.u)).flatten()
      main.comm.Recv(data,source=loc_rank,tag = loc_rank)
      uG[:,:,:,:,loc_rank*main.Npy:(loc_rank+1)*main.Npy] = np.reshape(data,(var.nvars,var.quadpoints,var.quadpoints,main.Npx,main.Npy))
    return uG
  else:
    main.comm.Send(var.u.flatten(),dest=0,tag=main.mpi_rank)

def getRankConnections(mpi_rank,num_processes):
  rank_connect = np.zeros((2))
  rank_connect[0] = mpi_rank - 1
  rank_connect[1] = mpi_rank+1
  if (mpi_rank == 0):
    rank_connect[0] = num_processes - 1
  
  if (mpi_rank == num_processes - 1):
    rank_connect[1] = 0
  return rank_connect
