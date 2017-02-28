import numpy as np
import mpi4py as MPI



def sendEdgesGeneralSlab(fL,fR,fD,fU,fB,fF,main):
  if (main.rank_connect[0] == main.mpi_rank and main.rank_connect[1] == main.mpi_rank):
    uR = fL[:,:,:,0, :,:]
    uL = fR[:,:,:,-1,:,:]
  else:
    ## Send right and left fluxes
    tmp = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npy,main.Npz)).flatten()
    main.comm.Sendrecv(fL[:,:,:,0,:,:].flatten(),dest=main.rank_connect[0],sendtag=main.mpi_rank,\
                       recvbuf=tmp,source=main.rank_connect[1],recvtag=main.rank_connect[1])

    uR = np.reshape(tmp,(main.nvars,main.quadpoints,main.quadpoints,main.Npy,main.Npz))
    tmp[:] = 0. 
    main.comm.Sendrecv(fR[:,:,:,-1,:,:].flatten(),dest=main.rank_connect[1],sendtag=main.mpi_rank*10,\
                       recvbuf=tmp,source=main.rank_connect[0],recvtag=main.rank_connect[0]*10)
    uL = np.reshape(tmp,(main.nvars,main.quadpoints,main.quadpoints,main.Npy,main.Npz))

  if (main.rank_connect[2] == main.mpi_rank and main.rank_connect[3] == main.mpi_rank):
    uU = fD[:,:,:,:,0 ,:]
    uD = fU[:,:,:,:,-1,:]
  else:
    ## Send up and down fluxes
    tmp = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npz)).flatten()
    main.comm.Sendrecv(fD[:,:,:,:,0,:].flatten(),dest=main.rank_connect[2],sendtag=main.mpi_rank,\
                       recvbuf=tmp,source=main.rank_connect[3],recvtag=main.rank_connect[3])
    uU = np.reshape(tmp,(main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npz))
    tmp[:] = 0. 
    main.comm.Sendrecv(fU[:,:,:,:,-1,:].flatten(),dest=main.rank_connect[3],sendtag=main.mpi_rank*100,\
                       recvbuf=tmp,source=main.rank_connect[2],recvtag=main.rank_connect[2]*100)
    uD = np.reshape(tmp,(main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npz))

    
  uF = fB[:,:,:,:,:,0]
  uB = fF[:,:,:,:,:,-1]

  return uR,uL,uU,uD,uF,uB


def gatherSolSlab(main,eqns,var):
  if (main.mpi_rank == 0):
    uG = np.zeros((var.nvars,var.quadpoints,var.quadpoints,var.quadpoints,main.Nel[0],main.Nel[1],main.Nel[2]))
    uG[:,:,:,:,0:main.Npx,0:main.Npy,:] = var.u[:]
    for i in range(1,main.num_processes):
      loc_rank = i
      data = np.zeros(np.shape(var.u)).flatten()
      main.comm.Recv(data,source=loc_rank,tag = loc_rank)
      xL = int( (loc_rank%main.procx)*main.Npx)
      xR = int(((loc_rank%main.procx) +1)*main.Npx)
      yD = int(loc_rank)/int(main.procx)*main.Npy
      yU = (int(loc_rank)/int(main.procx) + 1)*main.Npy
      uG[:,:,:,:,xL:xR,yD:yU,:] = np.reshape(data,(var.nvars,var.quadpoints,var.quadpoints,var.quadpoints,main.Npx,main.Npy,main.Npz))
    return uG
  else:
    main.comm.Send(var.u.flatten(),dest=0,tag=main.mpi_rank)

def gatherSolSpectral(a,main):
  if (main.mpi_rank == 0):
    aG = np.zeros((main.nvars,main.order,main.order,main.order,main.Nel[0],main.Nel[1],main.Nel[2]))
    aG[:,:,:,0:main.Npx,0:main.Npy,:] = a[:]
    for i in range(1,main.num_processes):
      loc_rank = i
      data = np.zeros(np.shape(a)).flatten()
      main.comm.Recv(data,source=loc_rank,tag = loc_rank)
      xL = int( (loc_rank%main.procx)*main.Npx)
      xR = int(((loc_rank%main.procx) +1)*main.Npx)
      yD = int(loc_rank)/int(main.procx)*main.Npy
      yU = (int(loc_rank)/int(main.procx) + 1)*main.Npy
      aG[:,:,:,xL:xR,yD:yU,:] = np.reshape(data,(main.nvars,main.order,main.order,main.order,main.Npx,main.Npy,main.Npz))
    return aG
  else:
    main.comm.Send(a.flatten(),dest=0,tag=main.mpi_rank)

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
