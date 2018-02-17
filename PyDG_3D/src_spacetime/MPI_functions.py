import numpy as np
import mpi4py as MPI
def globalNorm(r,main):
  ## Create Global residual
  data = main.comm.gather(np.linalg.norm(r)**2,root = 0)
  if (main.mpi_rank == 0):
    rn_glob = 0.
    for j in range(0,main.num_processes):
      rn_glob += data[j]
    rn_glob = np.sqrt(rn_glob)
    for j in range(1,main.num_processes):
      main.comm.send(rn_glob, dest=j)
  else:
    rn_glob = main.comm.recv(source=0)
  return rn_glob

def globalSum(r,main):
  ## Create Global residual
  data = main.comm.gather(np.sum(r),root = 0)
  if (main.mpi_rank == 0):
    rn_glob = 0.
    for j in range(0,main.num_processes_global):
      rn_glob += data[j]
    for j in range(1,main.num_processes_global):
      main.comm.send(rn_glob, dest=j)
  else:
    rn_glob = main.comm.recv(source=0)
  return rn_glob



def sendaEdgesGeneralSlab(a,main):
  if (main.rank_connect[0] == main.mpi_rank and main.rank_connect[1] == main.mpi_rank):
    aR_edge = a[:,:,:,:,:,0, :,:]
    aL_edge = a[:,:,:,:,:,-1,:,:]
  else:
    ## Send right and left fluxes
    tmp = np.zeros(np.size(a[:,:,:,:,:,0,:,:]))
    #tmp = np.zeros((main.nvars,main.order[0],main.order[1],main.order[2],main.Npy,main.Npz)).flatten()
    main.comm.Sendrecv(a[:,:,:,:,:,0,:,:].flatten(),dest=main.rank_connect[0],sendtag=main.mpi_rank,\
                       recvbuf=tmp,source=main.rank_connect[1],recvtag=main.rank_connect[1])
    #aR_edge = np.reshape(tmp,(main.nvars,main.order,main.order,main.order,main.Npy,main.Npz))
    aR_edge = np.reshape(tmp,np.shape(a[:,:,:,:,:,0,:,:]))
    #tmp = np.zeros((main.nvars,main.order,main.order,main.order,main.Npy,main.Npz)).flatten()
    tmp = np.zeros(np.size(a[:,:,:,:,:,-1,:,:]))
    main.comm.Sendrecv(a[:,:,:,:,:,-1,:,:].flatten(),dest=main.rank_connect[1],sendtag=main.mpi_rank*10,\
                       recvbuf=tmp,source=main.rank_connect[0],recvtag=main.rank_connect[0]*10)
    #aL_edge = np.reshape(tmp,(main.nvars,main.order,main.order,main.order,main.Npy,main.Npz))
    aL_edge = np.reshape(tmp,np.shape(a[:,:,:,:,:,-1,:,:]))

  if (main.rank_connect[2] == main.mpi_rank and main.rank_connect[3] == main.mpi_rank):
    aU_edge = a[:,:,:,:,:,:,0 ,:]
    aD_edge = a[:,:,:,:,:,:,-1,:]
  else:
    ## Send up and down fluxes
    #tmp = np.zeros((main.nvars,main.order,main.order,main.order,main.Npx,main.Npz)).flatten()
    tmp = np.zeros(np.size(a[:,:,:,:,:,:,0,:]))
    main.comm.Sendrecv(a[:,:,:,:,:,:,0,:].flatten(),dest=main.rank_connect[2],sendtag=main.mpi_rank,\
                       recvbuf=tmp,source=main.rank_connect[3],recvtag=main.rank_connect[3])
    #aU_edge = np.reshape(tmp,(main.nvars,main.order,main.order,main.order,main.Npx,main.Npz))
    aU_edge = np.reshape(tmp,np.shape(a[:,:,:,:,:,:,0,:]))
    #tmp = np.zeros((main.nvars,main.order,main.order,main.order,main.Npx,main.Npz)).flatten()
    tmp = np.zeros(np.size(a[:,:,:,:,:,:,-1,:]))
    main.comm.Sendrecv(a[:,:,:,:,:,:,-1,:].flatten(),dest=main.rank_connect[3],sendtag=main.mpi_rank*100,\
                       recvbuf=tmp,source=main.rank_connect[2],recvtag=main.rank_connect[2]*100)
    #aD_edge = np.reshape(tmp,(main.nvars,main.order,main.order,main.order,main.Npx,main.Npz))
    aD_edge = np.reshape(tmp,np.shape(a[:,:,:,:,:,-1,:]))
   
  aF_edge = a[:,:,:,:,:,:,:,0]
  aB_edge = a[:,:,:,:,:,:,:,-1]
  return aR_edge,aL_edge,aU_edge,aD_edge,aF_edge,aB_edge


def sendEdgesGeneralSlab_b(fL,fR,fD,fU,fB,fF,main,regionManager):
  uR = np.zeros(np.shape(fL[:,:,:,:,0,:,:]))
  uL = np.zeros(np.shape(fR[:,:,:,:,0,:,:]))
  uU = np.zeros(np.shape(fD[:,:,:,:,:,0,:]))
  uD = np.zeros(np.shape(fU[:,:,:,:,:,0,:]))
  uF = np.zeros(np.shape(fB[:,:,:,:,:,:,0]))
  uB = np.zeros(np.shape(fF[:,:,:,:,:,:,0]))

  ## If only using one processor ================
  if (main.rank_connect[0] == main.mpi_rank and main.rank_connect[1] == main.mpi_rank):
    uR[:] = fL[:,:,:,:,0, :,:]
    uL[:] = fR[:,:,:,:,-1,:,:]
    ### Boundary conditions. Overright uL and uR if we are on a boundary. 
    if (main.rightBC.BC_type == 'patch'):
      uR[:] = regionManager.region[main.rightBC.args[0]].b.uL[:,:,:,:,main.rightBC.args[1],:,:]
    else:
      uR[:] = main.rightBC.applyBC(fR[:,:,:,:,-1,:,:],uR,main.rightBC.args,main)
    if (main.leftBC.BC_type == 'patch'):
      uL[:] = regionManager.region[main.leftBC.args[0]].b.uR[:,:,:,:,main.leftBC.args[1],:,:]
    else:
      uL[:] = main.leftBC.applyBC(fL[:,:,:,:,0 ,:,:],uL,main.leftBC.args,main)

  #======================================================
  else:
    ## Send right and left fluxes
    tmp = np.zeros(np.size(fL[:,:,:,:,0,:,:]))
    main.comm.Sendrecv(fL[:,:,:,:,0,:,:].flatten(),dest=main.rank_connect[0],sendtag=main.mpi_rank,\
                       recvbuf=tmp,source=main.rank_connect[1],recvtag=main.rank_connect[1])
    uR[:] = np.reshape(tmp,np.shape(fL[:,:,:,:,0,:,:]))
    tmp = np.zeros(np.size(fL[:,:,:,:,-1,:,:]))
    main.comm.Sendrecv(fR[:,:,:,:,-1,:,:].flatten(),dest=main.rank_connect[1],sendtag=main.mpi_rank*10,\
                       recvbuf=tmp,source=main.rank_connect[0],recvtag=main.rank_connect[0]*10)
    #uL = np.reshape(tmp,(main.nvars,main.quadpoints,main.quadpoints,main.Npy,main.Npz))
    uL[:] = np.reshape(tmp,np.shape(fL[:,:,:,:,-1,:,:]))

    ### Boundary conditions. Overright uL and uR if we are on a boundary. 
    if (main.BC_rank[0]):
      uR[:] = main.rightBC.applyBC(fR[:,:,:,:,-1,:,:],uR,main.rightBC.args,main)
    if (main.BC_rank[2]):
      uL[:] = main.leftBC.applyBC(fL[:,:,:,:,0 ,:,:],uL,main.leftBC.args,main)


  if (main.rank_connect[2] == main.mpi_rank and main.rank_connect[3] == main.mpi_rank):
    uU[:] = fD[:,:,:,:,:,0 ,:]
    uD[:] = fU[:,:,:,:,:,-1,:]
    if (main.topBC.BC_type == 'patch'):
      uU[:] = regionManager.region[main.topBC.args[0]].b.uD[:,:,:,:,:,main.topBC.args[1],:]
    else:
      uU[:] = main.topBC.applyBC(fU[:,:,:,:,:,-1,:],uU,main.topBC.args,main)
    if (main.bottomBC.BC_type == 'patch'):
      uD[:] = regionManager.region[main.bottomBC.args[0]].b.uU[:,:,:,:,:,main.bottomBC.args[1],:]
    else:
      uD[:] = main.bottomBC.applyBC(fD[:,:,:,:,:,0,:],uD,main.bottomBC.args,main)

  else:
    ## Send up and down fluxes
    tmp = np.zeros(np.size(fD[:,:,:,:,:,0,:]))
    main.comm.Sendrecv(fD[:,:,:,:,:,0,:].flatten(),dest=main.rank_connect[2],sendtag=main.mpi_rank,\
                       recvbuf=tmp,source=main.rank_connect[3],recvtag=main.rank_connect[3])
    uU[:] = np.reshape(tmp,np.shape(fD[:,:,:,:,:,0,:]))
    tmp = np.zeros(np.size(fU[:,:,:,:,:,-1,:]))
    main.comm.Sendrecv(fU[:,:,:,:,:,-1,:].flatten(),dest=main.rank_connect[3],sendtag=main.mpi_rank*100,\
                       recvbuf=tmp,source=main.rank_connect[2],recvtag=main.rank_connect[2]*100)
    uD[:] = np.reshape(tmp,np.shape(fU[:,:,:,:,:,-1,:]))

    ### Boundary conditions. Overright uU and uD if we are on a boundary. 
    if (main.BC_rank[1]):
      uU[:] = main.topBC.applyBC(fU[:,:,:,:,:,-1,:],uU,main.topBC.args,main)
    if (main.BC_rank[3]):
      uD[:] = main.bottomBC.applyBC(fD[:,:,:,:,:,0,:],uD,main.bottomBC.args,main)

    
  uF[:] = fB[:,:,:,:,:,:,0]
  uB[:] = fF[:,:,:,:,:,:,-1]
  return uR,uL,uU,uD,uF,uB


def sendEdgesGeneralSlab(fL,fR,fD,fU,fB,fF,main,regionManager):
  uR = np.zeros(np.shape(fL[:,:,:,:,0,:,:]))
  uL = np.zeros(np.shape(fR[:,:,:,:,0,:,:]))
  uU = np.zeros(np.shape(fD[:,:,:,:,:,0,:]))
  uD = np.zeros(np.shape(fU[:,:,:,:,:,0,:]))
  uF = np.zeros(np.shape(fB[:,:,:,:,:,:,0]))
  uB = np.zeros(np.shape(fF[:,:,:,:,:,:,0]))
  ## If only using one processor ================
  print(main.mpi_rank,main.rank_connect[2])
  if (main.rank_connect[0] == main.mpi_rank and main.rank_connect[1] == main.mpi_rank):
    uR[:] = fL[:,:,:,:,0, :,:]
    uL[:] = fR[:,:,:,:,-1,:,:]
    ### Boundary conditions. Overright uL and uR if we are on a boundary. 
    if (main.rightBC.BC_type == 'patch'):
      uR[:] = regionManager.region[main.rightBC.args[0]].a.uL[:,:,:,:,main.rightBC.args[1],:,:]
    else:
      uR[:] = main.rightBC.applyBC(fR[:,:,:,:,-1,:,:],uR,main.rightBC.args,main)
    if (main.leftBC.BC_type == 'patch'):
      uL[:] = regionManager.region[main.leftBC.args[0]].a.uR[:,:,:,:,main.leftBC.args[1],:,:]
    else:
      uL[:] = main.leftBC.applyBC(fL[:,:,:,:,0 ,:,:],uL,main.leftBC.args,main)
 
  #======================================================
  else:
    ## Send right and left fluxes
    if (main.rank_connect[0] != main.mpi_rank):
      tmp = np.zeros(np.size(fL[:,:,:,:,0,:,:]))
      main.comm.Send(fL[:,:,:,:,0,:,:].flatten(),dest=main.rank_connect[0],tag=main.mpi_rank)
    if (main.rank_connect[1] != main.mpi_rank):
      tmp = np.zeros(np.size(fL[:,:,:,:,0,:,:]))
      main.comm.Recv(tmp,source=main.rank_connect[1],tag=main.rank_connect[1])
      uR[:] = np.reshape(tmp,np.shape(fL[:,:,:,:,0,:,:]))

    if (main.rank_connect[1] != main.mpi_rank):
      main.comm.Send(fR[:,:,:,:,-1,:,:].flatten(),dest=main.rank_connect[1],tag=main.mpi_rank*10)
    if (main.rank_connect[0] != main.mpi_rank):
      tmp = np.zeros(np.size(fL[:,:,:,:,0,:,:]))
      main.comm.Recv(tmp,source=main.rank_connect[0],tag=main.rank_connect[0]*10)
      uL[:] = np.reshape(tmp,np.shape(fL[:,:,:,:,-1,:,:]))

#    tmp = np.zeros(np.size(fL[:,:,:,:,0,:,:]))
#    main.comm.Sendrecv(fL[:,:,:,:,0,:,:].flatten(),dest=main.rank_connect[0],sendtag=main.mpi_rank,\
#                       recvbuf=tmp,source=main.rank_connect[1],recvtag=main.rank_connect[1])
#    uR[:] = np.reshape(tmp,np.shape(fL[:,:,:,:,0,:,:]))
#    tmp = np.zeros(np.size(fL[:,:,:,:,-1,:,:]))
#    main.comm.Sendrecv(fR[:,:,:,:,-1,:,:].flatten(),dest=main.rank_connect[1],sendtag=main.mpi_rank*10,\
#                       recvbuf=tmp,source=main.rank_connect[0],recvtag=main.rank_connect[0]*10)
#    #uL = np.reshape(tmp,(main.nvars,main.quadpoints,main.quadpoints,main.Npy,main.Npz))
#    uL[:] = np.reshape(tmp,np.shape(fL[:,:,:,:,-1,:,:]))
    ### Boundary conditions. Overright uL and uR if we are on a boundary. 
    if (main.BC_rank[0]):
      uR[:] = main.rightBC.applyBC(fR[:,:,:,:,-1,:,:],uR,main.rightBC.args,main)
    if (main.BC_rank[2]):
      uL[:] = main.leftBC.applyBC(fL[:,:,:,:,0 ,:,:],uL,main.leftBC.args,main)


  if (main.rank_connect[2] == main.mpi_rank and main.rank_connect[3] == main.mpi_rank):
    uU[:] = fD[:,:,:,:,:,0 ,:]
    uD[:] = fU[:,:,:,:,:,-1,:]
    if (main.topBC.BC_type == 'patch'):
      uU[:] = regionManager.region[main.topBC.args[0]].a.uD[:,:,:,:,:,main.topBC.args[1],:]
    else:
      uU[:] = main.topBC.applyBC(fU[:,:,:,:,:,-1,:],uU,main.topBC.args,main)
    if (main.bottomBC.BC_type == 'patch'):
      uD[:] = regionManager.region[main.bottomBC.args[0]].a.uU[:,:,:,:,:,main.bottomBC.args[1],:]
    else:
      uD[:] = main.bottomBC.applyBC(fD[:,:,:,:,:,0,:],uD,main.bottomBC.args,main)

  else:
    ## Send up and down fluxes
    if (main.rank_connect[2] != main.mpi_rank):
      main.comm.Send(fD[:,:,:,:,:,0,:].flatten(),dest=main.rank_connect[2],tag=main.mpi_rank)
    if (main.rank_connect[3] != main.mpi_rank): 
      tmp = np.zeros(np.size(fD[:,:,:,:,:,0,:]))
      main.comm.Recv(tmp,source=main.rank_connect[3],tag=main.rank_connect[3])
      uU[:] = np.reshape(tmp,np.shape(fD[:,:,:,:,:,0,:]))

    if (main.rank_connect[3] != main.mpi_rank):
      main.comm.Send(fU[:,:,:,:,:,-1,:].flatten(),dest=main.rank_connect[3],tag=main.mpi_rank*100)
    if (main.rank_connect[2] != main.mpi_rank):
      tmp = np.zeros(np.size(fD[:,:,:,:,:,0,:]))
      main.comm.Recv(tmp,source=main.rank_connect[2],tag=main.rank_connect[2]*100)
      uD[:] = np.reshape(tmp,np.shape(fU[:,:,:,:,:,-1,:]))

#    tmp = np.zeros(np.size(fD[:,:,:,:,:,0,:]))
#    main.comm.Sendrecv(fD[:,:,:,:,:,0,:].flatten(),dest=main.rank_connect[2],sendtag=main.mpi_rank,\
#                       recvbuf=tmp,source=main.rank_connect[3],recvtag=main.rank_connect[3])
#    uU[:] = np.reshape(tmp,np.shape(fD[:,:,:,:,:,0,:]))
#    tmp = np.zeros(np.size(fU[:,:,:,:,:,-1,:]))
#    main.comm.Sendrecv(fU[:,:,:,:,:,-1,:].flatten(),dest=main.rank_connect[3],sendtag=main.mpi_rank*100,\
#                       recvbuf=tmp,source=main.rank_connect[2],recvtag=main.rank_connect[2]*100)
#    uD[:] = np.reshape(tmp,np.shape(fU[:,:,:,:,:,-1,:]))

    ### Boundary conditions. Overright uU and uD if we are on a boundary. 
    if (main.BC_rank[1]):
      uU[:] = main.topBC.applyBC(fU[:,:,:,:,:,-1,:],uU,main.topBC.args,main)
    if (main.BC_rank[3]):
      uD[:] = main.bottomBC.applyBC(fD[:,:,:,:,:,0,:],uD,main.bottomBC.args,main)

  uF[:] = fB[:,:,:,:,:,:,0]
  uB[:] = fF[:,:,:,:,:,:,-1]

  #print('mpi_rank ' , main.mpi_rank )
  return uR,uL,uU,uD,uF,uB


def sendEdgesGeneralSlab_Derivs(fL,fR,fD,fU,fB,fF,main,regionManager):
  uR = np.zeros(np.shape(fL[:,:,:,:,0,:,:]))
  uL = np.zeros(np.shape(fR[:,:,:,:,0,:,:]))
  uU = np.zeros(np.shape(fD[:,:,:,:,:,0,:]))
  uD = np.zeros(np.shape(fU[:,:,:,:,:,0,:]))
  uF = np.zeros(np.shape(fB[:,:,:,:,:,:,0]))
  uB = np.zeros(np.shape(fF[:,:,:,:,:,:,0]))

  if (main.rank_connect[0] == main.mpi_rank and main.rank_connect[1] == main.mpi_rank):
    uR[:] = fL[:,:,:,:,0, :,:]
    uL[:] = fR[:,:,:,:,-1,:,:]
    if (main.rightBC.BC_type != 'periodic'):
      uR[:] = fR[:,:,:,:,-1,:,:]
    if (main.leftBC.BC_type != 'periodic'):
      uL[:] = fL[:,:,:,:,0 ,:,:]
    if (main.rightBC.BC_type == 'patch'):
      uR[:] = regionManager.region[main.rightBC.args[0]].b.uL[:,:,:,:,main.rightBC.args[1],:,:]
    if (main.leftBC.BC_type == 'patch'):
      uL[:] = regionManager.region[main.leftBC.args[0]].b.uR[:,:,:,:,main.leftBC.args[1],:,:]

  else:
    ## Send right and left fluxes
    if (main.rank_connect[0] != main.mpi_rank):
      tmp = np.zeros(np.size(fL[:,:,:,:,0,:,:]))
      main.comm.Send(fL[:,:,:,:,0,:,:].flatten(),dest=main.rank_connect[0],tag=main.mpi_rank)
    if (main.rank_connect[1] != main.mpi_rank):
      tmp = np.zeros(np.size(fL[:,:,:,:,0,:,:]))
      main.comm.Recv(tmp,source=main.rank_connect[1],tag=main.rank_connect[1])
      uR[:] = np.reshape(tmp,np.shape(fL[:,:,:,:,0,:,:]))

    if (main.rank_connect[1] != main.mpi_rank):
      main.comm.Send(fR[:,:,:,:,-1,:,:].flatten(),dest=main.rank_connect[1],tag=main.mpi_rank*10)
    if (main.rank_connect[0] != main.mpi_rank):
      tmp = np.zeros(np.size(fL[:,:,:,:,0,:,:]))
      main.comm.Recv(tmp,source=main.rank_connect[0],tag=main.rank_connect[0]*10)
      uL[:] = np.reshape(tmp,np.shape(fL[:,:,:,:,-1,:,:]))

    ## Send right and left fluxes
#    tmp = np.zeros(np.size(fL[:,:,:,:,0,:,:]))
#    main.comm.Sendrecv(fL[:,:,:,:,0,:,:].flatten(),dest=main.rank_connect[0],sendtag=main.mpi_rank,\
#                       recvbuf=tmp,source=main.rank_connect[1],recvtag=main.rank_connect[1])
#    uR[:] = np.reshape(tmp,np.shape(fL[:,:,:,:,0,:,:]))
#    tmp = np.zeros(np.size(fL[:,:,:,:,-1,:,:]))
#    main.comm.Sendrecv(fR[:,:,:,:,-1,:,:].flatten(),dest=main.rank_connect[1],sendtag=main.mpi_rank*10,\
#                       recvbuf=tmp,source=main.rank_connect[0],recvtag=main.rank_connect[0]*10)
#    #uL = np.reshape(tmp,(main.nvars,main.quadpoints,main.quadpoints,main.Npy,main.Npz))
#    uL[:] = np.reshape(tmp,np.shape(fL[:,:,:,:,-1,:,:]))

    ### Boundary conditions. Set uL and uR to interior values if on a boundary 
    if (main.BC_rank[0] and main.rightBC.BC_type != 'periodic'):
      uR[:] = fR[:,:,:,:,-1,:,:]
    if (main.BC_rank[2] and main.leftBC.BC_type != 'periodic'):
      uL[:] = fL[:,:,:,:,0 ,:,:]


  if (main.rank_connect[2] == main.mpi_rank and main.rank_connect[3] == main.mpi_rank):
    uU[:] = fD[:,:,:,:,:,0 ,:]
    uD[:] = fU[:,:,:,:,:,-1,:]
    if (main.topBC.BC_type != 'periodic'): 
      uU[:] = fU[:,:,:,:,:,-1,:]
    if (main.bottomBC.BC_type != 'periodic'):
      uD[:] = fD[:,:,:,:,:,0,:]

    if (main.topBC.BC_type == 'patch'):
      uU[:] = regionManager.region[main.topBC.args[0]].b.uD[:,:,:,:,:,main.topBC.args[1],:]
    if (main.bottomBC.BC_type == 'patch'):
      uD[:] = regionManager.region[main.bottomBC.args[0]].b.uU[:,:,:,:,:,main.bottomBC.args[1],:]


  else:
    ## Send up and down fluxes
    if (main.rank_connect[2] != main.mpi_rank):
      main.comm.Send(fD[:,:,:,:,:,0,:].flatten(),dest=main.rank_connect[2],tag=main.mpi_rank)
    if (main.rank_connect[3] != main.mpi_rank): 
      tmp = np.zeros(np.size(fD[:,:,:,:,:,0,:]))
      main.comm.Recv(tmp,source=main.rank_connect[3],tag=main.rank_connect[3])
      uU[:] = np.reshape(tmp,np.shape(fD[:,:,:,:,:,0,:]))

    if (main.rank_connect[3] != main.mpi_rank):
      main.comm.Send(fU[:,:,:,:,:,-1,:].flatten(),dest=main.rank_connect[3],tag=main.mpi_rank*100)
    if (main.rank_connect[2] != main.mpi_rank):
      tmp = np.zeros(np.size(fD[:,:,:,:,:,0,:]))
      main.comm.Recv(tmp,source=main.rank_connect[2],tag=main.rank_connect[2]*100)
      uD[:] = np.reshape(tmp,np.shape(fU[:,:,:,:,:,-1,:]))


#    tmp = np.zeros(np.size(fD[:,:,:,:,:,0,:]))
#    main.comm.Sendrecv(fD[:,:,:,:,:,0,:].flatten(),dest=main.rank_connect[2],sendtag=main.mpi_rank,\
#                       recvbuf=tmp,source=main.rank_connect[3],recvtag=main.rank_connect[3])
#    uU[:] = np.reshape(tmp,np.shape(fD[:,:,:,:,:,0,:]))
#    tmp = np.zeros(np.size(fU[:,:,:,:,:,-1,:]))
#    main.comm.Sendrecv(fU[:,:,:,:,:,-1,:].flatten(),dest=main.rank_connect[3],sendtag=main.mpi_rank*100,\
#                       recvbuf=tmp,source=main.rank_connect[2],recvtag=main.rank_connect[2]*100)
#    uD[:] = np.reshape(tmp,np.shape(fU[:,:,:,:,:,-1,:]))

    ### Boundary conditions. Overright uU and uD if we are on a boundary. 
    if (main.BC_rank[1] and main.topBC.BC_type != 'periodic'): 
      uU[:] = fU[:,:,:,:,:,-1,:]
    if (main.BC_rank[3] and main.bottomBC.BC_type != 'periodic'):
      uD[:] = fD[:,:,:,:,:,0,:]

    
  uF[:] = fB[:,:,:,:,:,:,0]
  uB[:] = fF[:,:,:,:,:,:,-1]
  return uR,uL,uU,uD,uF,uB

def gatherSolScalar(main,u):
  if (main.mpi_rank == main.starting_rank):
    uG = np.zeros((main.quadpoints[0],main.quadpoints[1],main.quadpoints[2],main.quadpoints[3],main.Nel[0],main.Nel[1],main.Nel[2],main.Nel[3]))
    uG[:,:,:,:,0:main.Npx,0:main.Npy,:] = u[:]
    #for i in range(1,main.num_processes):
    for i in main.all_mpi_ranks[1::]:
      loc_rank = i - main.starting_rank
      data = np.zeros(np.shape(u)).flatten()
      main.comm.Recv(data,source=loc_rank + main.starting_rank,tag = loc_rank + main.starting_rank)
      xL = int( (loc_rank%main.procx)*main.Npx)
      xR = int(((loc_rank%main.procx) +1)*main.Npx)
      yD = int(loc_rank)/int(main.procx)*main.Npy
      yU = (int(loc_rank)/int(main.procx) + 1)*main.Npy
      uG[:,:,:,:,xL:xR,yD:yU,:] = np.reshape(data,np.shape(u))
    return uG
  else:
    main.comm.Send(u.flatten(),dest=main.starting_rank,tag=main.mpi_rank)


def gatherSolSlab(main,eqns,var):
  if (main.mpi_rank == main.starting_rank):
    uG = np.zeros((var.nvars,var.quadpoints[0],var.quadpoints[1],var.quadpoints[2],var.quadpoints[3],main.Nel[0],main.Nel[1],main.Nel[2],main.Nel[3]))
    uG[:,:,:,:,:,0:main.Npx,0:main.Npy,:] = var.u[:]
    #for i in range(1,main.num_processes):
    for i in main.all_mpi_ranks[1::]:
      loc_rank = i - main.starting_rank
      data = np.zeros(np.shape(var.u)).flatten()
      main.comm.Recv(data,source=loc_rank + main.starting_rank,tag = loc_rank + main.starting_rank)
      xL = int( (loc_rank%main.procx)*main.Npx)
      xR = int(((loc_rank%main.procx) +1)*main.Npx)
      yD = int(loc_rank)/int(main.procx)*main.Npy
      yU = (int(loc_rank)/int(main.procx) + 1)*main.Npy
      #uG[:,:,:,:,xL:xR,yD:yU,:] = np.reshape(data,(var.nvars,var.quadpoints,var.quadpoints,var.quadpoints,main.Npx,main.Npy,main.Npz))
      uG[:,:,:,:,:,xL:xR,yD:yU,:] = np.reshape(data,np.shape(main.a.u))

    return uG
  else:
    main.comm.Send(var.u.flatten(),dest=main.starting_rank,tag=main.mpi_rank)

def gatherSolSpectral(a,main):
  if (main.mpi_rank == main.starting_rank):
    aG = np.zeros((main.nvars,main.order[0],main.order[1],main.order[2],main.order[3],main.Nel[0],main.Nel[1],main.Nel[2],main.Nel[3]))
    aG[:,:,:,:,:,0:main.Npx,0:main.Npy,:] = a[:]
    #for i in range(1,main.num_processes):
    for i in main.all_mpi_ranks[1::]:
      loc_rank = i - main.starting_rank
      data = np.zeros(np.shape(a)).flatten()
      main.comm.Recv(data,source=loc_rank + main.starting_rank,tag = loc_rank + main.starting_rank)
      xL = int( (loc_rank%main.procx)*main.Npx)
      xR = int(((loc_rank%main.procx) +1)*main.Npx)
      yD = int(loc_rank)/int(main.procx)*main.Npy
      yU = (int(loc_rank)/int(main.procx) + 1)*main.Npy
      aG[:,:,:,:,:,xL:xR,yD:yU,:] = np.reshape(data,(main.nvars,main.order[0],main.order[1],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    return aG
  else:
    main.comm.Send(a.flatten(),dest=main.starting_rank,tag=main.mpi_rank)


def regionConnector(regionManager):
#  for i in regionManager.mpi_regions_owned:
    region = regionManager.region[0]

    if (region.rightBC.BC_type == 'patch' and region.BC_rank[0]):
      row = (int(region.mpi_rank) - int(region.starting_rank))/int(region.procx)
      region_connect = region.rightBC.args[0]
      shift = regionManager.starting_rank[region_connect] - regionManager.starting_rank[region.region_number]
      region.rank_connect[1] = regionManager.starting_rank[region.region_number] + shift + regionManager.procx[region_connect]*row

    if (region.leftBC.BC_type == 'patch' and region.BC_rank[2]):
      row = (int(region.mpi_rank) - int(region.starting_rank))/int(region.procx)
      region_connect = region.leftBC.args[0]
      shift = regionManager.starting_rank[region_connect] - regionManager.starting_rank[region.region_number]
      region.rank_connect[0] = region.mpi_rank + shift + regionManager.procx[region_connect]*(row + 1) - 1
      #print('shift',region.mpi_rank,shift)

    if (region.topBC.BC_type == 'patch' and region.BC_rank[1]):
      column = (int(region.mpi_rank) - int(region.starting_rank))%int(region.procx)
      region_connect = region.topBC.args[0]
      shift = regionManager.starting_rank[region_connect] - regionManager.starting_rank[region.region_number]
      region.rank_connect[3] = region.mpi_rank + shift + column

    if (region.bottomBC.BC_type == 'patch' and region.BC_rank[3]):
      column = (int(region.mpi_rank) - int(region.starting_rank))%int(region.procx)
      region_connect = region.bottomBC.args[0]
      shift = regionManager.starting_rank[region_connect] - regionManager.starting_rank[region.region_number]
      region.rank_connect[2] =region.mpi_rank +  shift + regionManager.nprocs[region_connect] - region.procx + column
    #print(region.mpi_rank,region.rank_connect)

def getRankConnectionsSlab(mpi_rank,num_processes,procx,procy,starting_rank):
  ##============== MPI INFORMATION ===================
#  if (procx*procy != num_processes):
#    if (mpi_rank == 0):
#      print('Error, correct x/y proc decomposition, now quitting!')
#    sys.exit()
  rank_connect = np.zeros((4)) ##I'm an idiot and ordering for this is left right bottom top
  rank_connect[0] = mpi_rank - 1
  rank_connect[1] = mpi_rank+1
  rank_connect[2] = mpi_rank-procx
  rank_connect[3] = mpi_rank+procx
  BC_rank = [False,False,False,False] #ordering is right, top, left, bottom  
  if (mpi_rank == starting_rank): #bottom left corner
    rank_connect[0] = mpi_rank + procx - 1
    rank_connect[2] = starting_rank + num_processes - procx
    BC_rank[2],BC_rank[3] = True,True

  if ((mpi_rank +1) - procx == starting_rank): #bottom right corner
    rank_connect[1] = 0
    rank_connect[2] = starting_rank + num_processes - 1
    BC_rank[0],BC_rank[3] = True,True

  if (mpi_rank == starting_rank + num_processes-1): #top right corner
    rank_connect[1] = starting_rank + num_processes - procx
    rank_connect[3] = starting_rank + procx - 1
    BC_rank[0],BC_rank[1] = True,True
 
  if (mpi_rank == starting_rank +  num_processes - procx): #top left corner
    rank_connect[0] = starting_rank + num_processes - 1
    rank_connect[3] = starting_rank
    BC_rank[1],BC_rank[2] = True,True
  #  
  if (mpi_rank > starting_rank and mpi_rank < starting_rank + procx - 1): #bottom row 
    rank_connect[2] = -procx + num_processes + mpi_rank
    BC_rank[3] = True,True
  #
  if (mpi_rank > (starting_rank + num_processes - procx) and mpi_rank < (starting_rank + num_processes - 1)): #top row 
    rank_connect[3] = mpi_rank - num_processes + procx
    BC_rank[1] = True,True
  #
  if ( (mpi_rank + 1 - starting_rank)%procx == 0 and mpi_rank < (starting_rank + num_processes - 1) and mpi_rank > (starting_rank + procx)): #right row
    rank_connect[1] = mpi_rank - procx + 1
    BC_rank[0] = True,True
  #
  if ( (mpi_rank - starting_rank)%procx == 0 and mpi_rank < (starting_rank + num_processes - procx) and mpi_rank > starting_rank): #left row
    rank_connect[0] = mpi_rank +  procx - 1
    BC_rank[2] = True,True



  return rank_connect,BC_rank
