import numpy as np
import mpi4py as MPI


def sendEdgesGeneral(fL,fR,main):
  if (main.num_processes == 1):
    uR = fL[:,:,:,0 ]
    uL = fR[:,:,:,-1]
  else:
    tmp = np.zeros((main.nvars,main.quadpoints,main.Npx)).flatten()
    main.comm.Send(fL[:,:,:,0].flatten(),dest=main.rank_connect[0],tag=main.mpi_rank)
    main.comm.Recv(tmp,source=main.rank_connect[1],tag=main.rank_connect[1])
    uR = np.reshape(tmp,(main.nvars,main.quadpoints,main.Npx))
    tmp = np.zeros((main.nvars,main.quadpoints,main.Npx)).flatten()
    main.comm.Send(fR[:,:,:,-1].flatten(),dest=main.rank_connect[1],tag=main.mpi_rank*10)
    main.comm.Recv(tmp,source=main.rank_connect[0],tag=main.rank_connect[0]*10)
    uL = np.reshape(tmp,(main.nvars,main.quadpoints,main.Npx))
  return uR,uL


def sendEdgesGeneralSlab(fL,fR,fD,fU,main):
  if (main.rank_connect[0] == main.mpi_rank and main.rank_connect[1] == main.mpi_rank):
    uR = fL[:,:,0, :]
    uL = fR[:,:,-1,:]
  else:
    ## Send right and left fluxes
    tmp = np.zeros((main.nvars,main.quadpoints,main.Npy)).flatten()
    main.comm.Send(fL[:,:,0,:].flatten(),dest=main.rank_connect[0],tag=main.mpi_rank)
    main.comm.Recv(tmp,source=main.rank_connect[1],tag=main.rank_connect[1])
    uR = np.reshape(tmp,(main.nvars,main.quadpoints,main.Npy))
    tmp = np.zeros((main.nvars,main.quadpoints,main.Npy)).flatten()
    main.comm.Send(fR[:,:,-1,:].flatten(),dest=main.rank_connect[1],tag=main.mpi_rank*10)
    main.comm.Recv(tmp,source=main.rank_connect[0],tag=main.rank_connect[0]*10)
    uL = np.reshape(tmp,(main.nvars,main.quadpoints,main.Npy))

  if (main.rank_connect[2] == main.mpi_rank and main.rank_connect[3] == main.mpi_rank):
    uU = fD[:,:,:,0 ]
    uD = fU[:,:,:,-1]
  else:
    ## Send up and down fluxes
    tmp = np.zeros((main.nvars,main.quadpoints,main.Npx)).flatten()
    main.comm.Send(fD[:,:,:,0].flatten(),dest=main.rank_connect[2],tag=main.mpi_rank)
    main.comm.Recv(tmp,source=main.rank_connect[3],tag=main.rank_connect[3])
    uU = np.reshape(tmp,(main.nvars,main.quadpoints,main.Npx))
    tmp = np.zeros((main.nvars,main.quadpoints,main.Npx)).flatten()
    main.comm.Send(fU[:,:,:,-1].flatten(),dest=main.rank_connect[3],tag=main.mpi_rank*100)
    main.comm.Recv(tmp,source=main.rank_connect[2],tag=main.rank_connect[2]*100)
    uD = np.reshape(tmp,(main.nvars,main.quadpoints,main.Npx))

  return uR,uL,uU,uD


def sendEdgesSlab(main,var):
  if (main.rank_connect[0] == main.mpi_rank and main.rank_connect[1] == main.mpi_rank):
    var.uR_edge[:] = var.uL[:,:,0, :]
    var.uL_edge[:] = var.uR[:,:,-1,:]
  else:
    main.comm.Send(var.uL[:,:,0,:].flatten(),dest=main.rank_connect[0],tag=main.mpi_rank)
    main.comm.Recv(var.edge_tmpx,source=main.rank_connect[1],tag=main.rank_connect[1])
    var.uR_edge[:] = np.reshape(var.edge_tmpx,(main.nvars,main.quadpoints,main.Npy))
    main.comm.Send(var.uR[:,:,-1,:].flatten(),dest=main.rank_connect[1],tag=main.mpi_rank*10)
    main.comm.Recv(var.edge_tmpx,source=main.rank_connect[0],tag=main.rank_connect[0]*10)
    var.uL_edge[:] = np.reshape(var.edge_tmpx,(main.nvars,main.quadpoints,main.Npy))

    ## Send up and down fluxes
  if (main.rank_connect[2] == main.mpi_rank and main.rank_connect[3] == main.mpi_rank):
    var.uU_edge[:] = var.uD[:,:,:,0 ]
    var.uD_edge[:] = var.uU[:,:,:,-1]
  else:
    main.comm.Send(var.uD[:,:,:,0].flatten(),dest=main.rank_connect[2],tag=main.mpi_rank)
    main.comm.Recv(var.edge_tmpy,source=main.rank_connect[3],tag=main.rank_connect[3])
    var.uU_edge[:] = np.reshape(var.edge_tmpy,(main.nvars,main.quadpoints,main.Npx))
    main.comm.Send(var.uU[:,:,:,-1].flatten(),dest=main.rank_connect[3],tag=main.mpi_rank*100)
    main.comm.Recv(var.edge_tmpy,source=main.rank_connect[2],tag=main.rank_connect[2]*100)
    var.uD_edge[:] = np.reshape(var.edge_tmpy,(main.nvars,main.quadpoints,main.Npx))


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

def gatherSolSlab(main,eqns,var):
  if (main.mpi_rank == 0):
    uG = np.zeros((var.nvars,var.quadpoints,var.quadpoints,main.Nel[0],main.Nel[1]))
    uG[:,:,:,0:main.Npx,0:main.Npy] = var.u[:]
    for i in range(1,main.num_processes):
      loc_rank = i
      data = np.zeros(np.shape(var.u)).flatten()
      main.comm.Recv(data,source=loc_rank,tag = loc_rank)
      xL = int( (loc_rank%main.procx)*main.Npx)
      xR = int(((loc_rank%main.procx) +1)*main.Npx)
      yD = int(loc_rank)/int(main.procx)*main.Npy
      yU = (int(loc_rank)/int(main.procx) + 1)*main.Npy
      uG[:,:,:,xL:xR,yD:yU] = np.reshape(data,(var.nvars,var.quadpoints,var.quadpoints,main.Npx,main.Npy))
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


def getRankConnectionsSlab(mpi_rank,num_processes,procx,procy):
  ##============== MPI INFORMATION ===================
  if (procx*procy != num_processes):
    if (mpi_rank == 0):
      print('Error, correct x/y proc decomposition, now quitting!')
    sys.exit()
  rank_connect = np.zeros((4))
  rank_connect[0] = mpi_rank - 1
  rank_connect[1] = mpi_rank+1
  rank_connect[2] = mpi_rank-procx
  rank_connect[3] = mpi_rank+procx
  
  if (mpi_rank == 0): #bottom left corner
    rank_connect[0] = mpi_rank + procx - 1
    rank_connect[2] = num_processes - procx
  
  if ((mpi_rank +1) - procx == 0): #bottom right corner
    rank_connect[1] = 0
    rank_connect[2] = num_processes - 1
 
  if (mpi_rank == num_processes-1): #top right corner
    rank_connect[1] = num_processes - procx
    rank_connect[3] = procx - 1
 
  if (mpi_rank == num_processes - procx): #top left corner
    rank_connect[0] = num_processes - 1
    rank_connect[3] = 0
  #  
  if (mpi_rank > 0 and mpi_rank < procx - 1): #bottom row 
    rank_connect[2] = -procx + num_processes + mpi_rank
  #
  if (mpi_rank > num_processes - procx and mpi_rank < num_processes - 1): #top row 
    rank_connect[3] = mpi_rank - num_processes + procx
  #
  if ( (mpi_rank + 1)%procx == 0 and mpi_rank < num_processes - 1 and mpi_rank > procx): #right row
    rank_connect[1] = mpi_rank - procx + 1
  #
  if ( mpi_rank%procx == 0 and mpi_rank < num_processes - procx and mpi_rank > 0): #left row
    rank_connect[0] = mpi_rank +  procx - 1
  return rank_connect
