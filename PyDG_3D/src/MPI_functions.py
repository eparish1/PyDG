import numpy as np
import mpi4py as MPI


def sendaEdgesGeneralSlab(a,main):
  if (main.rank_connect[0] == main.mpi_rank and main.rank_connect[1] == main.mpi_rank):
    aR_edge = a[:,:,:,:,0, :,:]
    aL_edge = a[:,:,:,:,-1,:,:]
  else:
    ## Send right and left fluxes
    tmp = np.zeros(np.size(a[:,:,:,:,0,:,:]))
    #tmp = np.zeros((main.nvars,main.order[0],main.order[1],main.order[2],main.Npy,main.Npz)).flatten()
    main.comm.Sendrecv(a[:,:,:,:,0,:,:].flatten(),dest=main.rank_connect[0],sendtag=main.mpi_rank,\
                       recvbuf=tmp,source=main.rank_connect[1],recvtag=main.rank_connect[1])
    #aR_edge = np.reshape(tmp,(main.nvars,main.order,main.order,main.order,main.Npy,main.Npz))
    aR_edge = np.reshape(tmp,np.shape(a[:,:,:,:,0,:,:]))
    #tmp = np.zeros((main.nvars,main.order,main.order,main.order,main.Npy,main.Npz)).flatten()
    tmp = np.zeros(np.size(a[:,:,:,:,-1,:,:]))
    main.comm.Sendrecv(a[:,:,:,:,-1,:,:].flatten(),dest=main.rank_connect[1],sendtag=main.mpi_rank*10,\
                       recvbuf=tmp,source=main.rank_connect[0],recvtag=main.rank_connect[0]*10)
    #aL_edge = np.reshape(tmp,(main.nvars,main.order,main.order,main.order,main.Npy,main.Npz))
    aL_edge = np.reshape(tmp,np.shape(a[:,:,:,:,-1,:,:]))

  if (main.rank_connect[2] == main.mpi_rank and main.rank_connect[3] == main.mpi_rank):
    aU_edge = a[:,:,:,:,:,0 ,:]
    aD_edge = a[:,:,:,:,:,-1,:]
  else:
    ## Send up and down fluxes
    #tmp = np.zeros((main.nvars,main.order,main.order,main.order,main.Npx,main.Npz)).flatten()
    tmp = np.zeros(np.size(a[:,:,:,:,:,0,:]))
    main.comm.Sendrecv(a[:,:,:,:,:,0,:].flatten(),dest=main.rank_connect[2],sendtag=main.mpi_rank,\
                       recvbuf=tmp,source=main.rank_connect[3],recvtag=main.rank_connect[3])
    #aU_edge = np.reshape(tmp,(main.nvars,main.order,main.order,main.order,main.Npx,main.Npz))
    aU_edge = np.reshape(tmp,np.shape(a[:,:,:,:,:,0,:]))
    #tmp = np.zeros((main.nvars,main.order,main.order,main.order,main.Npx,main.Npz)).flatten()
    tmp = np.zeros(np.size(a[:,:,:,:,:,-1,:]))
    main.comm.Sendrecv(a[:,:,:,:,:,-1,:].flatten(),dest=main.rank_connect[3],sendtag=main.mpi_rank*100,\
                       recvbuf=tmp,source=main.rank_connect[2],recvtag=main.rank_connect[2]*100)
    #aD_edge = np.reshape(tmp,(main.nvars,main.order,main.order,main.order,main.Npx,main.Npz))
    aD_edge = np.reshape(tmp,np.shape(a[:,:,:,:,:,-1,:]))
   
  aF_edge = a[:,:,:,:,:,:,0]
  aB_edge = a[:,:,:,:,:,:,-1]
  return aR_edge,aL_edge,aU_edge,aD_edge,aF_edge,aB_edge


def sendEdgesGeneralSlab(fL,fR,fD,fU,fB,fF,main):
  uR = np.zeros(np.shape(fL[:,:,:,0,:,:]))
  uL = np.zeros(np.shape(fR[:,:,:,0,:,:]))
  uU = np.zeros(np.shape(fD[:,:,:,:,0,:]))
  uD = np.zeros(np.shape(fU[:,:,:,:,0,:]))
  uF = np.zeros(np.shape(fB[:,:,:,:,:,0]))
  uB = np.zeros(np.shape(fF[:,:,:,:,:,0]))

  if (main.rank_connect[0] == main.mpi_rank and main.rank_connect[1] == main.mpi_rank):
    uR[:] = fL[:,:,:,0, :,:]
    uL[:] = fR[:,:,:,-1,:,:]
    uR[:] = main.rightBC.applyBC(fR[:,:,:,-1,:,:],uR,main.rightBC.args,main.gas)
    uL[:] = main.leftBC.applyBC(fL[:,:,:,0 ,:,:],uL,main.leftBC.args,main.gas)
  else:
    ## Send right and left fluxes

    tmp = np.zeros(np.size(fL[:,:,:,0,:,:]))
    main.comm.Sendrecv(fL[:,:,:,0,:,:].flatten(),dest=main.rank_connect[0],sendtag=main.mpi_rank,\
                       recvbuf=tmp,source=main.rank_connect[1],recvtag=main.rank_connect[1])
    uR[:] = np.reshape(tmp,np.shape(fL[:,:,:,0,:,:]))
    tmp = np.zeros(np.size(fL[:,:,:,-1,:,:]))
    main.comm.Sendrecv(fR[:,:,:,-1,:,:].flatten(),dest=main.rank_connect[1],sendtag=main.mpi_rank*10,\
                       recvbuf=tmp,source=main.rank_connect[0],recvtag=main.rank_connect[0]*10)
    #uL = np.reshape(tmp,(main.nvars,main.quadpoints,main.quadpoints,main.Npy,main.Npz))
    uL[:] = np.reshape(tmp,np.shape(fL[:,:,:,-1,:,:]))

    ### Boundary conditions. Overright uL and uR if we are on a boundary. 
    if (main.BC_rank[0]):
      uR[:] = main.rightBC.applyBC(fR[:,:,:,-1,:,:],uR,main.rightBC.args,main.gas)
    if (main.BC_rank[2]):
      uL[:] = main.leftBC.applyBC(fL[:,:,:,0 ,:,:],uL,main.leftBC.args,main.gas)


  if (main.rank_connect[2] == main.mpi_rank and main.rank_connect[3] == main.mpi_rank):
    uU[:] = fD[:,:,:,:,0 ,:]
    uD[:] = fU[:,:,:,:,-1,:]
    uU[:] = main.topBC.applyBC(fU[:,:,:,:,-1,:],uU,main.topBC.args,main.gas)
    uD[:] = main.bottomBC.applyBC(fD[:,:,:,:,0,:],uD,main.bottomBC.args,main.gas)

  else:
    ## Send up and down fluxes
    tmp = np.zeros(np.size(fD[:,:,:,:,0,:]))
    main.comm.Sendrecv(fD[:,:,:,:,0,:].flatten(),dest=main.rank_connect[2],sendtag=main.mpi_rank,\
                       recvbuf=tmp,source=main.rank_connect[3],recvtag=main.rank_connect[3])
    uU[:] = np.reshape(tmp,np.shape(fD[:,:,:,:,0,:]))
    tmp = np.zeros(np.size(fU[:,:,:,:,-1,:]))
    main.comm.Sendrecv(fU[:,:,:,:,-1,:].flatten(),dest=main.rank_connect[3],sendtag=main.mpi_rank*100,\
                       recvbuf=tmp,source=main.rank_connect[2],recvtag=main.rank_connect[2]*100)
    uD[:] = np.reshape(tmp,np.shape(fU[:,:,:,:,-1,:]))

    ### Boundary conditions. Overright uU and uD if we are on a boundary. 
    if (main.BC_rank[1]):
      uU[:] = main.topBC.applyBC(fU[:,:,:,:,-1,:],uU,main.topBC.args,main.gas)
    if (main.BC_rank[3]):
      uD[:] = main.bottomBC.applyBC(fD[:,:,:,:,0,:],uD,main.bottomBC.args,main.gas)

    
  uF[:] = fB[:,:,:,:,:,0]
  uB[:] = fF[:,:,:,:,:,-1]
  return uR,uL,uU,uD,uF,uB


def sendEdgesGeneralSlab_Derivs(fL,fR,fD,fU,fB,fF,main):
  uR = np.zeros(np.shape(fL[:,:,:,0,:,:]))
  uL = np.zeros(np.shape(fR[:,:,:,0,:,:]))
  uU = np.zeros(np.shape(fD[:,:,:,:,0,:]))
  uD = np.zeros(np.shape(fU[:,:,:,:,0,:]))
  uF = np.zeros(np.shape(fB[:,:,:,:,:,0]))
  uB = np.zeros(np.shape(fF[:,:,:,:,:,0]))

  if (main.rank_connect[0] == main.mpi_rank and main.rank_connect[1] == main.mpi_rank):
    uR[:] = fL[:,:,:,0, :,:]
    uL[:] = fR[:,:,:,-1,:,:]
    if (main.rightBC.BC_type != 'periodic'):
      uR[:] = fR[:,:,:,-1,:,:]
    if (main.leftBC.BC_type != 'periodic'):
      uL[:] = fL[:,:,:,0 ,:,:]

  else:
    ## Send right and left fluxes
    tmp = np.zeros(np.size(fL[:,:,:,0,:,:]))
    main.comm.Sendrecv(fL[:,:,:,0,:,:].flatten(),dest=main.rank_connect[0],sendtag=main.mpi_rank,\
                       recvbuf=tmp,source=main.rank_connect[1],recvtag=main.rank_connect[1])
    uR[:] = np.reshape(tmp,np.shape(fL[:,:,:,0,:,:]))
    tmp = np.zeros(np.size(fL[:,:,:,-1,:,:]))
    main.comm.Sendrecv(fR[:,:,:,-1,:,:].flatten(),dest=main.rank_connect[1],sendtag=main.mpi_rank*10,\
                       recvbuf=tmp,source=main.rank_connect[0],recvtag=main.rank_connect[0]*10)
    #uL = np.reshape(tmp,(main.nvars,main.quadpoints,main.quadpoints,main.Npy,main.Npz))
    uL[:] = np.reshape(tmp,np.shape(fL[:,:,:,-1,:,:]))

    ### Boundary conditions. Set uL and uR to interior values if on a boundary 
    if (main.BC_rank[0] and main.rightBC.BC_type != 'periodic'):
      uR[:] = fR[:,:,:,-1,:,:]
    if (main.BC_rank[2] and main.leftBC.BC_type != 'periodic'):
      uL[:] = fL[:,:,:,0 ,:,:]


  if (main.rank_connect[2] == main.mpi_rank and main.rank_connect[3] == main.mpi_rank):
    uU[:] = fD[:,:,:,:,0 ,:]
    uD[:] = fU[:,:,:,:,-1,:]
    if (main.topBC.BC_type != 'periodic'): 
      uU[:] = fU[:,:,:,:,-1,:]
    if (main.bottomBC.BC_type != 'periodic'):
      uD[:] = fD[:,:,:,:,0,:]
  else:
    ## Send up and down fluxes
    tmp = np.zeros(np.size(fD[:,:,:,:,0,:]))
    main.comm.Sendrecv(fD[:,:,:,:,0,:].flatten(),dest=main.rank_connect[2],sendtag=main.mpi_rank,\
                       recvbuf=tmp,source=main.rank_connect[3],recvtag=main.rank_connect[3])
    uU[:] = np.reshape(tmp,np.shape(fD[:,:,:,:,0,:]))
    tmp = np.zeros(np.size(fU[:,:,:,:,-1,:]))
    main.comm.Sendrecv(fU[:,:,:,:,-1,:].flatten(),dest=main.rank_connect[3],sendtag=main.mpi_rank*100,\
                       recvbuf=tmp,source=main.rank_connect[2],recvtag=main.rank_connect[2]*100)
    uD[:] = np.reshape(tmp,np.shape(fU[:,:,:,:,-1,:]))

    ### Boundary conditions. Overright uU and uD if we are on a boundary. 
    if (main.BC_rank[1] and main.topBC.BC_type != 'periodic'): 
      uU[:] = fU[:,:,:,:,-1,:]
    if (main.BC_rank[3] and main.bottomBC.BC_type != 'periodic'):
      uD[:] = fD[:,:,:,:,0,:]

    
  uF[:] = fB[:,:,:,:,:,0]
  uB[:] = fF[:,:,:,:,:,-1]
  return uR,uL,uU,uD,uF,uB



def gatherSolSlab(main,eqns,var):
  if (main.mpi_rank == 0):
    uG = np.zeros((var.nvars,var.quadpoints[0],var.quadpoints[1],var.quadpoints[2],main.Nel[0],main.Nel[1],main.Nel[2]))
    uG[:,:,:,:,0:main.Npx,0:main.Npy,:] = var.u[:]
    for i in range(1,main.num_processes):
      loc_rank = i
      data = np.zeros(np.shape(var.u)).flatten()
      main.comm.Recv(data,source=loc_rank,tag = loc_rank)
      xL = int( (loc_rank%main.procx)*main.Npx)
      xR = int(((loc_rank%main.procx) +1)*main.Npx)
      yD = int(loc_rank)/int(main.procx)*main.Npy
      yU = (int(loc_rank)/int(main.procx) + 1)*main.Npy
      #uG[:,:,:,:,xL:xR,yD:yU,:] = np.reshape(data,(var.nvars,var.quadpoints,var.quadpoints,var.quadpoints,main.Npx,main.Npy,main.Npz))
      uG[:,:,:,:,xL:xR,yD:yU,:] = np.reshape(data,np.shape(main.a.u))

    return uG
  else:
    main.comm.Send(var.u.flatten(),dest=0,tag=main.mpi_rank)

def gatherSolSpectral(a,main):
  if (main.mpi_rank == 0):
    aG = np.zeros((main.nvars,main.order[0],main.order[1],main.order[2],main.Nel[0],main.Nel[1],main.Nel[2]))
    aG[:,:,:,:,0:main.Npx,0:main.Npy,:] = a[:]
    for i in range(1,main.num_processes):
      loc_rank = i
      data = np.zeros(np.shape(a)).flatten()
      main.comm.Recv(data,source=loc_rank,tag = loc_rank)
      xL = int( (loc_rank%main.procx)*main.Npx)
      xR = int(((loc_rank%main.procx) +1)*main.Npx)
      yD = int(loc_rank)/int(main.procx)*main.Npy
      yU = (int(loc_rank)/int(main.procx) + 1)*main.Npy
      aG[:,:,:,:,xL:xR,yD:yU,:] = np.reshape(data,(main.nvars,main.order[0],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz))
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
  BC_rank = [False,False,False,False] #ordering is right, top, left, bottom  
  if (mpi_rank == 0): #bottom left corner
    rank_connect[0] = mpi_rank + procx - 1
    rank_connect[2] = num_processes - procx
    BC_rank[2],BC_rank[3] = True,True

  if ((mpi_rank +1) - procx == 0): #bottom right corner
    rank_connect[1] = 0
    rank_connect[2] = num_processes - 1
    BC_rank[0],BC_rank[3] = True,True

  if (mpi_rank == num_processes-1): #top right corner
    rank_connect[1] = num_processes - procx
    rank_connect[3] = procx - 1
    BC_rank[0],BC_rank[1] = True,True
 
  if (mpi_rank == num_processes - procx): #top left corner
    rank_connect[0] = num_processes - 1
    rank_connect[3] = 0
    BC_rank[1],BC_rank[2] = True,True
  #  
  if (mpi_rank > 0 and mpi_rank < procx - 1): #bottom row 
    rank_connect[2] = -procx + num_processes + mpi_rank
    BC_rank[3] = True,True
  #
  if (mpi_rank > num_processes - procx and mpi_rank < num_processes - 1): #top row 
    rank_connect[3] = mpi_rank - num_processes + procx
    BC_rank[1] = True,True
  #
  if ( (mpi_rank + 1)%procx == 0 and mpi_rank < num_processes - 1 and mpi_rank > procx): #right row
    rank_connect[1] = mpi_rank - procx + 1
    BC_rank[0] = True,True
  #
  if ( mpi_rank%procx == 0 and mpi_rank < num_processes - procx and mpi_rank > 0): #left row
    rank_connect[0] = mpi_rank +  procx - 1
    BC_rank[2] = True,True
  return rank_connect,BC_rank
