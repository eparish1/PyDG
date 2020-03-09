import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
num_processes = comm.Get_size()
#========================================================================================

def globalNorm(r,regionManager):
  ## Create Global residual
  data = comm.gather(np.linalg.norm(r)**2,root = 0)
  rn_glob = np.zeros(1)
  if (regionManager.mpi_rank == 0):
    for j in range(0,regionManager.num_processes):
      rn_glob[:] += data[j]
    rn_glob = np.sqrt(rn_glob)
    for j in range(1,regionManager.num_processes):
      comm.Send(rn_glob, dest=j)
  else:
    comm.Recv(rn_glob,source=0)
  return rn_glob[0]

#========================================================================================

def globalSum(r,regionManager):
  ## Create Global residual
  data = regionManager.comm.gather(np.sum(r),root = 0)
  rn_glob = np.zeros(1)
  if (regionManager.mpi_rank == 0):
    for j in range(0,regionManager.num_processes):
      rn_glob[:] += data[j]
    for j in range(1,regionManager.num_processes):
      comm.Send(rn_glob, dest=j)
  else:
    comm.Recv(rn_glob,source=0)
  return rn_glob[0]

def globalDot(V,r,regionManager):
  ## Create Global residual
  tmp = np.dot(V,r)
  data = regionManager.comm.gather(tmp,root = 0)
  #print(np.shape(data))
  rn_glob = np.zeros(np.size(tmp))
  if (regionManager.mpi_rank == 0):
    for j in range(0,regionManager.num_processes):
      rn_glob[:] += data[j]
    for j in range(1,regionManager.num_processes):
      regionManager.comm.Send(rn_glob, dest=j)
  else:
      regionManager.comm.Recv(rn_glob, source=0)
  return rn_glob
##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
##########################################################################################################################################################################################################

def sendEdgesGeneralSlab(fL,fR,fD,fU,fB,fF,main,regionManager):
  nRL = 0
  nUD = 0
  nFB = 0

  uR = np.zeros(np.shape(fL[:,:,:,:,0,:,:]),dtype=fR.dtype)
  uL = np.zeros(np.shape(fR[:,:,:,:,0,:,:]),dtype=fL.dtype)
  uU = np.zeros(np.shape(fD[:,:,:,:,:,0,:]),dtype=fU.dtype)
  uD = np.zeros(np.shape(fU[:,:,:,:,:,0,:]),dtype=fD.dtype)
  uF = np.zeros(np.shape(fB[:,:,:,:,:,:,0]),dtype=fF.dtype)
  uB = np.zeros(np.shape(fF[:,:,:,:,:,:,0]),dtype=fB.dtype)

  left_face   = ((main.mpi_rank-main.starting_rank)%main.procx==0)
  right_face  = ((main.mpi_rank-main.starting_rank)%main.procx==main.procx-1)
  
  bottom_face = (((main.mpi_rank-main.starting_rank)%(main.procx*main.procy))//main.procx==0)
  top_face    = (((main.mpi_rank-main.starting_rank)%(main.procx*main.procy))//main.procx==main.procy-1)

  back_face   = ((main.mpi_rank-main.starting_rank)//(main.procx*main.procy)==0)
  front_face  = ((main.mpi_rank-main.starting_rank)//(main.procx*main.procy)==main.procz-1)
  
  ######################################################################################################################################################



  ## If only using one processor ================
  
  if (main.rank_connect[0] == main.mpi_rank and main.rank_connect[1] == main.mpi_rank):
    uR[:] = fL[:,:,:,:, 0,:,:]
    uL[:] = fR[:,:,:,:,-1,:,:]
    if (main.rightBC.BC_type == 'patch'):
        if (main.rightBC.args[1] ==  0):
            uR[:] = regionManager.region[main.rightBC.args[0]].a.uL[:,:,:,:, 0,:,:]
            #uR[:,::main.rightBC.ydir,::main.rightBC.zdir,:,::main.rightBC.ydir,::main.rightBC.zdir] = regionManager.region[main.rightBC.args[0]].a.uL[:,:,:,:, 0,:,:]

        if (main.rightBC.args[1] == -1):
            uR[:] = regionManager.region[main.rightBC.args[0]].a.uR[:,:,:,:,-1,:,:]
            #uR[:,::main.rightBC.ydir,::main.rightBC.zdir,:,::main.rightBC.ydir,::main.rightBC.zdir] = regionManager.region[main.rightBC.args[0]].a.uR[:,:,:,:,-1,:,:]
   
    uR[:] = main.rightBC.applyBC(fR[:,:,:,:,-1,:,:],uR,main.rightBC.args,main,main.normals[0,:,-1,:,:])
    
    if (main.leftBC.BC_type == 'patch'):
        if (main.leftBC.args[1] == -1):
            uL[:] = regionManager.region[main.leftBC.args[0]].a.uR[:,:,:,:,-1,:,:]
            #uL[:,::main.leftBC.ydir,::main.leftBC.zdir,:,::main.leftBC.ydir,::main.leftBC.zdir] = regionManager.region[main.leftBC.args[0]].a.uR[:,:,:,:,-1,:,:]

        if (main.leftBC.args[1] ==  0):
            uL[:] = regionManager.region[main.leftBC.args[0]].a.uL[:,:,:,:, 0,:,:]
            #uL[:,::main.leftBC.ydir,::main.leftBC.zdir,:,::main.leftBC.ydir,::main.leftBC.zdir] = regionManager.region[main.leftBC.args[0]].a.uL[:,:,:,:, 0,:,:]
   
    uL[:] = main.leftBC.applyBC(fL[:,:,:,:, 0,:,:],uL,main.leftBC.args,main,main.normals[1,:,0,:,:]) # note that flipped directions gets taken care of here
# 
  #======================================================
  else:
    if (main.rank_connect[0] != main.mpi_rank and ((main.leftBC.args[1] == -1 and (main.leftBC.BC_type == 'periodic' or main.leftBC.BC_type == 'patch') and left_face==True) or left_face==False)):
      nRL+=1
      main.comm.Send(fL[:,:,:,:,0,:,:].flatten(),dest=main.rank_connect[0],tag=main.mpi_rank)
    #============================

    if (main.rank_connect[1] != main.mpi_rank and ((main.rightBC.args[1] == 0 and (main.rightBC.BC_type == 'periodic' or main.rightBC.BC_type == 'patch') and right_face==True) or right_face==False)):
      nRL-=1
      tmp = np.zeros(np.size(fL[:,:,:,:,0,:,:]),dtype=fL.dtype)
      main.comm.Recv(tmp,source=main.rank_connect[1],tag=main.rank_connect[1])
      uR[:] = np.reshape(tmp,np.shape(fL[:,:,:,:,0,:,:]))
      #uR[:,::main.rightBC.ydir,::main.rightBC.zdir,:,::main.rightBC.ydir,::main.rightBC.zdir] =  np.reshape(tmp,np.shape(fL[:,:,:,:,0,:,:]))
    #============================
    if (main.rank_connect[0] != main.mpi_rank and main.leftBC.args[1] == 0 and (main.leftBC.BC_type == 'periodic' or main.leftBC.BC_type == 'patch') and left_face==True):
      tmp = np.zeros(np.size(fL[:,:,:,:,0,:,:]),dtype=fL.dtype)
      main.comm.Sendrecv(fL[:,:,:,:,0,:,:].flatten(),dest=main.rank_connect[0],sendtag=main.mpi_rank,\
                         recvbuf=tmp,source=main.rank_connect[0],recvtag=main.rank_connect[0])
      uL[:] = np.reshape(tmp,np.shape(fL[:,:,:,:,0,:,:]))
      #uL[:,::main.leftBC.ydir,::main.leftBC.zdir,:,::main.leftBC.ydir,::main.leftBC.zdir] =  np.reshape(tmp,np.shape(fL[:,:,:,:,0,:,:]))
    #============================





    if (main.rank_connect[1] != main.mpi_rank and ((main.rightBC.args[1] == 0 and (main.rightBC.BC_type == 'periodic' or main.rightBC.BC_type == 'patch') and right_face==True) or right_face==False)):
      nRL+=1
      main.comm.Send(fR[:,:,:,:,-1,:,:].flatten(),dest=main.rank_connect[1],tag=main.mpi_rank+main.num_processes_global)

    #============================

    if (main.rank_connect[0] != main.mpi_rank and ((main.leftBC.args[1] == -1 and (main.leftBC.BC_type == 'periodic' or main.leftBC.BC_type == 'patch') and left_face==True) or left_face==False)):
      nRL-=1
      tmp = np.zeros(np.size(fL[:,:,:,:,0,:,:]),dtype=fL.dtype)
      main.comm.Recv(tmp,source=main.rank_connect[0],tag=main.rank_connect[0]+main.num_processes_global)
      uL[:] = np.reshape(tmp,np.shape(fL[:,:,:,:,-1,:,:]))
      #uL[:,::main.leftBC.ydir,::main.leftBC.zdir,:,::main.leftBC.ydir,::main.leftBC.zdir] = np.reshape(tmp,np.shape(fL[:,:,:,:,-1,:,:])) 
    #============================

    if (main.rank_connect[1] != main.mpi_rank and main.rightBC.args[1] == -1 and (main.rightBC.BC_type == 'periodic' or main.rightBC.BC_type == 'patch') and right_face==True):
      tmp = np.zeros(np.size(fL[:,:,:,:,0,:,:]),dtype=fL.dtype)
      main.comm.Sendrecv(fR[:,:,:,:,-1,:,:].flatten(),dest=main.rank_connect[1],sendtag=main.mpi_rank+main.num_processes_global,\
                         recvbuf=tmp,source=main.rank_connect[1],recvtag=main.rank_connect[1]+main.num_processes_global)
      uR[:] = np.reshape(tmp,np.shape(fR[:,:,:,:,-1,:,:]))
      #uR[:,::main.rightBC.ydir,::main.rightBC.zdir,:,::main.rightBC.ydir,::main.rightBC.zdir] = np.reshape(tmp,np.shape(fR[:,:,:,:,-1,:,:])) 
   
    #============================

    if (main.BC_rank[1]):
      uR[:] = main.rightBC.applyBC(fR[:,:,:,:,-1,:,:],uR,main.rightBC.args,main,main.normals[0,:,-1,:,:])

    if (main.BC_rank[0]):
      uL[:] = main.leftBC.applyBC(fL[:,:,:,:,0 ,:,:],uL,main.leftBC.args,main,main.normals[1,:,0,:,:])



  ######################################################################################################################################################



  ## If only using one processor ================
  if (main.rank_connect[2] == main.mpi_rank and main.rank_connect[3] == main.mpi_rank):
    uU[:] = fD[:,:,:,:,:,0 ,:]
    uD[:] = fU[:,:,:,:,:,-1,:]
    
    if (main.topBC.BC_type == 'patch'):
        if (main.topBC.args[1] ==  0):
            uU[:] = regionManager.region[main.topBC.args[0]].a.uD[:,:,:,:,:, 0]
            #uU[:,::main.topBC.xdir,::main.topBC.zdir,:,::main.topBC.xdir,::main.topBC.zdir,:] = regionManager.region[main.topBC.args[0]].a.uD[:,:,:,:,:, 0]
        if (main.topBC.args[1] == -1):
            uU[:] = regionManager.region[main.topBC.args[0]].a.uU[:,:,:,:,:,-1]
            #uU[:,::main.topBC.xdir,::main.topBC.zdir,:,::main.topBC.xdir,::main.topBC.zdir,:] = regionManager.region[main.topBC.args[0]].a.uU[:,:,:,:,:,-1]
    uU[:] = main.topBC.applyBC(fU[:,:,:,:,:,-1,:],uU,main.topBC.args,main,main.normals[2,:,:,-1,:])

    if (main.bottomBC.BC_type == 'patch'):
        if (main.bottomBC.args[1] == -1):
            uD[:]  = regionManager.region[main.bottomBC.args[0]].a.uU[:,:,:,:,:,-1]
            #uD[:,::main.bottomBC.xdir,::main.bottomBC.zdir,:,::main.bottomBC.xdir,::main.bottomBC.zdir,:]  = regionManager.region[main.bottomBC.args[0]].a.uU[:,:,:,:,:,-1]
        if (main.bottomBC.args[1] ==  0):
            uD[:]  = regionManager.region[main.bottomBC.args[0]].a.uD[:,:,:,:,:, 0]
            #uD[:,::main.bottomBC.xdir,::main.bottomBC.zdir,:,::main.bottomBC.xdir,::main.bottomBC.zdir,:]  = regionManager.region[main.bottomBC.args[0]].a.uD[:,:,:,:,:, 0]


    uD[:] = main.bottomBC.applyBC(fD[:,:,:,:,:,0,:],uD,main.bottomBC.args,main,main.normals[3,:,:,0,:])

  #======================================================
  else:
    if (main.rank_connect[2] != main.mpi_rank and ((main.bottomBC.args[1] == -1 and (main.bottomBC.BC_type == 'periodic' or main.bottomBC.BC_type == 'patch') and bottom_face==True) or bottom_face==False)):
      nUD+=1
      main.comm.Send(fD[:,:,:,:,:,0,:].flatten(),dest=main.rank_connect[2],tag=main.mpi_rank+2*main.num_processes_global)

    #============================

    if (main.rank_connect[3] != main.mpi_rank and ((main.topBC.args[1] == 0 and (main.topBC.BC_type == 'periodic' or main.topBC.BC_type == 'patch') and top_face==True) or top_face==False)):
      nUD-=1
      tmp = np.zeros(np.size(fD[:,:,:,:,:,0,:]),dtype=fD.dtype)
      main.comm.Recv(tmp,source=main.rank_connect[3],tag=main.rank_connect[3]+2*main.num_processes_global)
      uU[:] = np.reshape(tmp,np.shape(fD[:,:,:,:,:,0,:]))

    #============================

    if (main.rank_connect[2] != main.mpi_rank and main.bottomBC.args[1] == 0 and (main.bottomBC.BC_type == 'periodic' or main.bottomBC.BC_type == 'patch') and bottom_face==True):
      tmp = np.zeros(np.size(fD[:,:,:,:,:,0,:]),dtype=fD.dtype)
      main.comm.Sendrecv(fD[:,:,:,:,:,0,:].flatten(),dest=main.rank_connect[2],sendtag=main.mpi_rank+2*main.num_processes_global,\
                         recvbuf=tmp,source=main.rank_connect[2],recvtag=main.rank_connect[2]+2*main.num_processes_global)
      uD[:] = np.reshape(tmp,np.shape(fD[:,:,:,:,:,0,:]))

    #============================





    if (main.rank_connect[3] != main.mpi_rank and ((main.topBC.args[1] == 0 and (main.topBC.BC_type == 'periodic' or main.topBC.BC_type == 'patch') and top_face==True) or top_face==False)):
      nUD+=1
      main.comm.Send(fU[:,:,:,:,:,-1,:].flatten(),dest=main.rank_connect[3],tag=main.mpi_rank+3*main.num_processes_global)

    #============================

    if (main.rank_connect[2] != main.mpi_rank and ((main.bottomBC.args[1] == -1 and (main.bottomBC.BC_type == 'periodic' or main.bottomBC.BC_type == 'patch') and bottom_face==True) or bottom_face==False)):
      nUD-=1
      tmp = np.zeros(np.size(fD[:,:,:,:,:,0,:]),dtype=fD.dtype)
      main.comm.Recv(tmp,source=main.rank_connect[2],tag=main.rank_connect[2]+3*main.num_processes_global)
      uD[:] = np.reshape(tmp,np.shape(fD[:,:,:,:,:,-1,:]))
    
    #============================

    if (main.rank_connect[3] != main.mpi_rank and main.topBC.args[1] == -1 and (main.topBC.BC_type == 'periodic' or main.topBC.BC_type == 'patch') and top_face==True):
      tmp = np.zeros(np.size(fD[:,:,:,:,:,0,:]),dtype=fD.dtype)
      main.comm.Sendrecv(fU[:,:,:,:,:,-1,:].flatten(),dest=main.rank_connect[3],sendtag=main.mpi_rank+3*main.num_processes_global,\
                         recvbuf=tmp,source=main.rank_connect[3],recvtag=main.rank_connect[3]+3*main.num_processes_global)
      uU[:] = np.reshape(tmp,np.shape(fD[:,:,:,:,:,-1,:]))
    
    #============================

    if (main.BC_rank[3]):
      uU[:] = main.topBC.applyBC(fU[:,:,:,:,:,-1,:],uU,main.topBC.args,main,main.normals[2,:,:,-1,:])

    if (main.BC_rank[2]):
      uD[:] = main.bottomBC.applyBC(fD[:,:,:,:,:,0,:],uD,main.bottomBC.args,main,main.normals[3,:,:,0,:])



  ######################################################################################################################################################



  ## If only using one processor ================
  if (main.rank_connect[4] == main.mpi_rank and main.rank_connect[5] == main.mpi_rank):
    uF[:] = fB[:,:,:,:,:,:, 0]
    uB[:] = fF[:,:,:,:,:,:,-1]
    if (main.frontBC.BC_type == 'patch'):
        if (main.frontBC.args[1] ==  0):
            uF[:] = regionManager.region[main.frontBC.args[0]].a.uB[:,:,:,:,:,:,0]
        if (main.frontBC.args[1] == -1):
            uF[:] = regionManager.region[main.frontBC.args[0]].a.uF[:,:,:,:,:,:,-1]
    
    uF[:] = main.frontBC.applyBC(fF[:,:,:,:,:,:,-1],uF,main.frontBC.args,main,main.normals[4,:,:,:,-1])
    
    if (main.backBC.BC_type == 'patch'):
        if (main.backBC.args[1] == -1):
            uB[:] = regionManager.region[main.backBC.args[0]].a.uF[:,:,:,:,:,:,-1]
        if (main.backBC.args[1] ==  0):
            uB[:] = regionManager.region[main.backBC.args[0]].a.uB[:,:,:,:,:,:,0]
    
    uB[:] = main.backBC.applyBC(fB[:,:,:,:,:,:,0],uB,main.backBC.args,main,main.normals[5,:,:,:,0])
  #======================================================
  else:
    if (main.rank_connect[4] != main.mpi_rank and ((main.backBC.args[1] == -1 and (main.backBC.BC_type == 'periodic' or main.backBC.BC_type == 'patch') and back_face==True) or back_face==False)):
      nFB+=1
      main.comm.Send(fB[:,:,:,:,:,:,0].flatten(),dest=main.rank_connect[4],tag=main.mpi_rank+4*main.num_processes_global)

    #============================

    if (main.rank_connect[5] != main.mpi_rank and ((main.frontBC.args[1] == 0 and (main.frontBC.BC_type == 'periodic' or main.frontBC.BC_type == 'patch') and front_face==True) or front_face==False)):
      nFB-=1
      tmp = np.zeros(np.size(fB[:,:,:,:,:,:,0]),dtype=fB.dtype)
      main.comm.Recv(tmp,source=main.rank_connect[5],tag=main.rank_connect[5]+4*main.num_processes_global)
      uF[:] = np.reshape(tmp,np.shape(fB[:,:,:,:,:,:,0]))

    #============================

    if (main.rank_connect[4] != main.mpi_rank and main.backBC.args[1] == 0 and (main.backBC.BC_type == 'periodic' or main.backBC.BC_type == 'patch') and back_face==True):
      tmp = np.zeros(np.size(fB[:,:,:,:,:,:,0]),dtype=fB.dtype)
      main.comm.Sendrecv(fB[:,:,:,:,:,:,0].flatten(),dest=main.rank_connect[4],sendtag=main.mpi_rank+4*main.num_processes_global,\
                         recvbuf=tmp,source=main.rank_connect[4],recvtag=main.rank_connect[4]+4*main.num_processes_global)
      uB[:] = np.reshape(tmp,np.shape(fB[:,:,:,:,:,:,0]))

    #============================





    if (main.rank_connect[5] != main.mpi_rank and ((main.frontBC.args[1] == 0 and (main.frontBC.BC_type == 'periodic' or main.frontBC.BC_type == 'patch') and front_face==True) or front_face==False)):
      nFB+=1
      main.comm.Send(fF[:,:,:,:,:,:,-1].flatten(),dest=main.rank_connect[5],tag=main.mpi_rank+5*main.num_processes_global)

    #============================

    if (main.rank_connect[4] != main.mpi_rank and ((main.backBC.args[1] == -1 and (main.backBC.BC_type == 'periodic' or main.backBC.BC_type == 'patch') and back_face==True) or back_face==False)):
      nFB-=1
      tmp = np.zeros(np.size(fB[:,:,:,:,:,:,0]),dtype=fB.dtype)
      main.comm.Recv(tmp,source=main.rank_connect[4],tag=main.rank_connect[4]+5*main.num_processes_global)
      uB[:] = np.reshape(tmp,np.shape(fB[:,:,:,:,:,:,-1]))
    
    #============================

    if (main.rank_connect[5] != main.mpi_rank and main.frontBC.args[1] == -1 and (main.frontBC.BC_type == 'periodic' or main.frontBC.BC_type == 'patch') and front_face==True):
      tmp = np.zeros(np.size(fB[:,:,:,:,:,:,0]),dtype=fB.dtype)
      main.comm.Sendrecv(fF[:,:,:,:,:,:,-1].flatten(),dest=main.rank_connect[5],sendtag=main.mpi_rank+5*main.num_processes_global,\
                         recvbuf=tmp,source=main.rank_connect[5],recvtag=main.rank_connect[5]+5*main.num_processes_global)
      uF[:] = np.reshape(tmp,np.shape(fB[:,:,:,:,:,:,-1]))
    
    #============================

    if (main.BC_rank[5]):
      uF[:] = main.frontBC.applyBC(fF[:,:,:,:,:,:,-1],uF,main.frontBC.args,main,main.normals[4,:,:,:,-1])

    if (main.BC_rank[4]):
      uB[:] = main.backBC.applyBC(fB[:,:,:,:,:,:,0],uB,main.backBC.args,main,main.normals[5,:,:,:,0])



  return uR,uL,uU,uD,uF,uB


##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
##########################################################################################################################################################################################################


def sendEdgesGeneralSlab_Derivs(fL,fR,fD,fU,fB,fF,main,regionManager):
  uR = np.zeros(np.shape(fL[:,:,:,:,0,:,:]),dtype=fL.dtype)
  uL = np.zeros(np.shape(fR[:,:,:,:,0,:,:]),dtype=fR.dtype)
  uU = np.zeros(np.shape(fD[:,:,:,:,:,0,:]),dtype=fD.dtype)
  uD = np.zeros(np.shape(fU[:,:,:,:,:,0,:]),dtype=fU.dtype)
  uF = np.zeros(np.shape(fB[:,:,:,:,:,:,0]),dtype=fB.dtype)
  uB = np.zeros(np.shape(fF[:,:,:,:,:,:,0]),dtype=fF.dtype)
 
  left_face   = ((main.mpi_rank-main.starting_rank)%main.procx==0)
  right_face  = ((main.mpi_rank-main.starting_rank)%main.procx==main.procx-1)
  
  bottom_face = (((main.mpi_rank-main.starting_rank)%(main.procx*main.procy))/main.procx==0)
  top_face    = (((main.mpi_rank-main.starting_rank)%(main.procx*main.procy))/main.procx==main.procy-1)

  back_face   = ((main.mpi_rank-main.starting_rank)/(main.procx*main.procy)==0)
  front_face  = ((main.mpi_rank-main.starting_rank)/(main.procx*main.procy)==main.procz-1)

  ####################################################################################################################################



  if (main.rank_connect[0] == main.mpi_rank and main.rank_connect[1] == main.mpi_rank):
    uR[:] = fL[:,:,:,:,0, :,:]
    uL[:] = fR[:,:,:,:,-1,:,:]

    if (main.rightBC.BC_type != 'periodic' and main.rightBC.BC_type != 'patch'):
      uR[:] = fR[:,:,:,:,-1,:,:]
    
    if (main.leftBC.BC_type != 'periodic' and main.leftBC.BC_type != 'patch'):
      uL[:] = fL[:,:,:,:,0 ,:,:]
    
    if (main.rightBC.BC_type == 'patch'):
        if (main.rightBC.args[1] ==  0):
            uR[:] = regionManager.region[main.rightBC.args[0]].b.uL[:,:,:,:, 0,:,:]
        if (main.rightBC.args[1] == -1):
            uR[:] = regionManager.region[main.rightBC.args[0]].b.uR[:,:,:,:,-1,:,:]
        uR[:] = main.rightBC.applyBC(fR[:,:,:,:,-1,:,:],uR,main.rightBC.args,main,main.normals[0,:,-1,:,:])
    
    if (main.leftBC.BC_type == 'patch'):
        if (main.leftBC.args[1] == -1):
            uL[:] = regionManager.region[main.leftBC.args[0]].b.uR[:,:,:,:,-1,:,:]
        if (main.leftBC.args[1] ==  0):
            uL[:] = regionManager.region[main.leftBC.args[0]].b.uL[:,:,:,:, 0,:,:]
        uL[:] = main.leftBC.applyBC(fL[:,:,:,:,0,:,:],uL,main.leftBC.args,main,main.normals[1,:,0,:,:])

  else:
    
    if (main.rank_connect[0] != main.mpi_rank and ((main.leftBC.args[1] == -1 and (main.leftBC.BC_type == 'periodic' or main.leftBC.BC_type == 'patch') and left_face==True) or left_face==False)):
      main.comm.Send(fL[:,:,:,:,0,:,:].flatten(),dest=main.rank_connect[0],tag=main.mpi_rank)

    #============================

    if (main.rank_connect[1] != main.mpi_rank and ((main.rightBC.args[1] == 0 and (main.rightBC.BC_type == 'periodic' or main.rightBC.BC_type == 'patch') and right_face==True) or right_face==False)):
      tmp = np.zeros(np.size(fL[:,:,:,:,0,:,:]))
      main.comm.Recv(tmp,source=main.rank_connect[1],tag=main.rank_connect[1])
      uR[:] = np.reshape(tmp,np.shape(fL[:,:,:,:,0,:,:]))

    #============================

    if (main.rank_connect[0] != main.mpi_rank and main.leftBC.args[1] == 0 and (main.leftBC.BC_type == 'periodic' or main.leftBC.BC_type == 'patch') and left_face==True):
      tmp = np.zeros(np.size(fL[:,:,:,:,0,:,:]))
      main.comm.Sendrecv(fL[:,:,:,:,0,:,:].flatten(),dest=main.rank_connect[0],sendtag=main.mpi_rank,\
                         recvbuf=tmp,source=main.rank_connect[0],recvtag=main.rank_connect[0])
      uL[:] = np.reshape(tmp,np.shape(fL[:,:,:,:,0,:,:]))

    #============================





    if (main.rank_connect[1] != main.mpi_rank and ((main.rightBC.args[1] == 0 and (main.rightBC.BC_type == 'periodic' or main.rightBC.BC_type == 'patch') and right_face==True) or right_face==False)):
      main.comm.Send(fR[:,:,:,:,-1,:,:].flatten(),dest=main.rank_connect[1],tag=main.mpi_rank+main.num_processes_global)

    #============================

    if (main.rank_connect[0] != main.mpi_rank and ((main.leftBC.args[1] == -1 and (main.leftBC.BC_type == 'periodic' or main.leftBC.BC_type == 'patch') and left_face==True) or left_face==False)):
      tmp = np.zeros(np.size(fL[:,:,:,:,0,:,:]))
      main.comm.Recv(tmp,source=main.rank_connect[0],tag=main.rank_connect[0]+main.num_processes_global)
      uL[:] = np.reshape(tmp,np.shape(fL[:,:,:,:,-1,:,:]))
    
    #============================

    if (main.rank_connect[1] != main.mpi_rank and main.rightBC.args[1] == -1 and (main.rightBC.BC_type == 'periodic' or main.rightBC.BC_type == 'patch') and right_face==True):
      tmp = np.zeros(np.size(fL[:,:,:,:,0,:,:]))
      main.comm.Sendrecv(fR[:,:,:,:,-1,:,:].flatten(),dest=main.rank_connect[1],sendtag=main.mpi_rank+main.num_processes_global,\
                         recvbuf=tmp,source=main.rank_connect[1],recvtag=main.rank_connect[1]+main.num_processes_global)
      uR[:] = np.reshape(tmp,np.shape(fR[:,:,:,:,-1,:,:]))
    
    #============================





    if (main.BC_rank[1] and main.rightBC.BC_type != 'periodic' and main.rightBC.BC_type != 'patch'):
      uR[:] = fR[:,:,:,:,-1,:,:]
    elif (main.BC_rank[1] and (main.rightBC.BC_type == 'periodic' or main.leftBC.BC_type == 'patch')):
      uR[:] = main.rightBC.applyBC(fR[:,:,:,:,-1,:,:],uR,main.rightBC.args,main,main.normals[0,:,-1,:,:])
    
    if (main.BC_rank[0] and main.leftBC.BC_type != 'periodic' and main.leftBC.BC_type != 'patch'):
      uL[:] = fL[:,:,:,:,0 ,:,:]
    elif (main.BC_rank[0] and (main.rightBC.BC_type == 'periodic' or main.leftBC.BC_type == 'patch')):
      uL[:] = main.leftBC.applyBC(fL[:,:,:,:,0,:,:],uL,main.leftBC.args,main,main.normals[1,:,0,:,:])


  
  ####################################################################################################################################
  


  if (main.rank_connect[2] == main.mpi_rank and main.rank_connect[3] == main.mpi_rank):
    uU[:] = fD[:,:,:,:,:,0 ,:]
    uD[:] = fU[:,:,:,:,:,-1,:]
    
    if (main.topBC.BC_type != 'periodic' and main.topBC.BC_type != 'patch'): 
      uU[:] = fU[:,:,:,:,:,-1,:]
    
    if (main.bottomBC.BC_type != 'periodic' and main.bottomBC.BC_type != 'patch'):
      uD[:] = fD[:,:,:,:,:,0,:]

    if (main.topBC.BC_type == 'patch'):
        if (main.topBC.args[1] ==  0):
            uU[:] = regionManager.region[main.topBC.args[0]].b.uD[:,:,:,:,:, 0,:,:]
            #uU[:,::main.topBC.xdir,:,:,::main.topBC.xdir,:,:] = regionManager.region[main.topBC.args[0]].b.uD[:,:,:,:,:, 0,:,:]
        if (main.topBC.args[1] == -1):
            uU[:] = regionManager.region[main.topBC.args[0]].b.uU[:,:,:,:,:,-1,:,:]
            #uU[:,::main.topBC.xdir,:,:,::main.topBC.xdir,:,:] = regionManager.region[main.topBC.args[0]].b.uU[:,:,:,:,:,-1,:,:]

        uU[:] = main.topBC.applyBC(fU[:,:,:,:,:,-1,:],uU,main.topBC.args,main,main.normals[2,:,:,-1,:])
    
    if (main.bottomBC.BC_type == 'patch'):
        if (main.bottomBC.args[1] == -1):
            uD[:] = regionManager.region[main.bottomBC.args[0]].b.uU[:,:,:,:,:,-1,:,:]
            #uD[:,::main.bottomBC.xdir,::main.bottomBC.zdir,:,::main.bottomBC.xdir,::main.bottomBC.zdir,:] = regionManager.region[main.bottomBC.args[0]].b.uU[:,:,:,:,:,-1,:,:] 
        if (main.bottomBC.args[1] ==  0):
            uD[:] = regionManager.region[main.bottomBC.args[0]].b.uD[:,:,:,:,:, 0,:,:]
            #uD[:,::main.bottomBC.xdir,::main.bottomBC.zdir,:,::main.bottomBC.xdir,::main.bottomBC.zdir,:] = regionManager.region[main.bottomBC.args[0]].b.uD[:,:,:,:,:, 0,:,:] 

        uD[:] = main.bottomBC.applyBC(fD[:,:,:,:,:,0,:],uD,main.bottomBC.args,main,main.normals[3,:,:,0,:])

  else:
    
    if (main.rank_connect[2] != main.mpi_rank and ((main.bottomBC.args[1] == -1 and (main.bottomBC.BC_type == 'periodic' or main.bottomBC.BC_type == 'patch') and bottom_face==True) or bottom_face==False)):
      main.comm.Send(fD[:,:,:,:,:,0,:].flatten(),dest=main.rank_connect[2],tag=main.mpi_rank+2*main.num_processes_global)

    #============================

    if (main.rank_connect[3] != main.mpi_rank and ((main.topBC.args[1] == 0 and (main.topBC.BC_type == 'periodic' or main.topBC.BC_type == 'patch') and top_face==True) or top_face==False)):
      tmp = np.zeros(np.size(fD[:,:,:,:,:,0,:]))
      main.comm.Recv(tmp,source=main.rank_connect[3],tag=main.rank_connect[3]+2*main.num_processes_global)
      uU[:] = np.reshape(tmp,np.shape(fD[:,:,:,:,:,0,:]))

    #============================

    if (main.rank_connect[2] != main.mpi_rank and main.bottomBC.args[1] == 0 and (main.bottomBC.BC_type == 'periodic' or main.bottomBC.BC_type == 'patch') and bottom_face==True):
      tmp = np.zeros(np.size(fD[:,:,:,:,:,0,:]))
      main.comm.Sendrecv(fD[:,:,:,:,:,0,:].flatten(),dest=main.rank_connect[2],sendtag=main.mpi_rank+2*main.num_processes_global,\
                         recvbuf=tmp,source=main.rank_connect[2],recvtag=main.rank_connect[2]+2*main.num_processes_global)
      uD[:] = np.reshape(tmp,np.shape(fD[:,:,:,:,:,0,:]))

    #============================





    if (main.rank_connect[3] != main.mpi_rank and ((main.topBC.args[1] == 0 and (main.topBC.BC_type == 'periodic' or main.topBC.BC_type == 'patch') and top_face==True) or top_face==False)):
      main.comm.Send(fU[:,:,:,:,:,-1,:].flatten(),dest=main.rank_connect[3],tag=main.mpi_rank+3*main.num_processes_global)

    #============================

    if (main.rank_connect[2] != main.mpi_rank and ((main.bottomBC.args[1] == -1 and (main.bottomBC.BC_type == 'periodic' or main.bottomBC.BC_type == 'patch') and bottom_face==True) or bottom_face==False)):
      tmp = np.zeros(np.size(fD[:,:,:,:,:,0,:]))
      main.comm.Recv(tmp,source=main.rank_connect[2],tag=main.rank_connect[2]+3*main.num_processes_global)
      uD[:] = np.reshape(tmp,np.shape(fD[:,:,:,:,:,-1,:]))
    
    #============================

    if (main.rank_connect[3] != main.mpi_rank and main.topBC.args[1] == -1 and (main.topBC.BC_type == 'periodic' or main.topBC.BC_type == 'patch') and top_face==True):
      tmp = np.zeros(np.size(fD[:,:,:,:,:,0,:]))
      main.comm.Sendrecv(fU[:,:,:,:,:,-1,:].flatten(),dest=main.rank_connect[3],sendtag=main.mpi_rank+3*main.num_processes_global,\
                         recvbuf=tmp,source=main.rank_connect[3],recvtag=main.rank_connect[3]+3*main.num_processes_global)
      uU[:] = np.reshape(tmp,np.shape(fD[:,:,:,:,:,-1,:]))
    
    #============================





    if (main.BC_rank[3] and main.topBC.BC_type != 'periodic' and main.topBC.BC_type != 'patch'): 
      uU[:] = fU[:,:,:,:,:,-1,:]
    elif (main.BC_rank[3] and (main.topBC.BC_type == 'periodic' or main.bottomBC.BC_type == 'patch')):
      uU[:] = main.topBC.applyBC(fU[:,:,:,:,:,-1,:],uU,main.topBC.args,main,main.normals[2,:,:,-1,:])

    if (main.BC_rank[2] and main.bottomBC.BC_type != 'periodic' and main.bottomBC.BC_type != 'patch'):
      uD[:] = fD[:,:,:,:,:,0,:]
    elif (main.BC_rank[2] and (main.topBC.BC_type == 'periodic' or main.bottomBC.BC_type == 'patch')):
      uD[:] = main.bottomBC.applyBC(fD[:,:,:,:,:,0,:],uD,main.bottomBC.args,main,main.normals[3,:,:,0,:])



  ####################################################################################################################################


    
  if (main.rank_connect[4] == main.mpi_rank and main.rank_connect[5] == main.mpi_rank):
    uF[:] = fB[:,:,:,:,:,:, 0]
    uB[:] = fF[:,:,:,:,:,:,-1]
    
    if (main.frontBC.BC_type != 'periodic' and main.frontBC.BC_type != 'patch'): 
      uF[:] = fF[:,:,:,:,:,:,-1]
    
    if (main.backBC.BC_type != 'periodic' and main.backBC.BC_type != 'patch'):
      uB[:] = fB[:,:,:,:,:,:, 0]

    if (main.frontBC.BC_type == 'patch'):
        if (main.frontBC.args[1] ==  0):
            uF[:] = regionManager.region[main.frontBC.args[0]].b.uB[:,:,:,:,:,:,0]
        if (main.frontBC.args[1] == -1):
            uF[:] = regionManager.region[main.frontBC.args[0]].b.uF[:,:,:,:,:,:,-1]
        uF[:] = main.frontBC.applyBC(fF[:,:,:,:,:,:,-1],uF,main.frontBC.args,main,main.normals[4,:,:,:,-1])
    
    if (main.backBC.BC_type == 'patch'):
        if (main.backBC.args[1] == -1):
            uB[:] = regionManager.region[main.backBC.args[0]].b.uF[:,:,:,:,:,:,-1]
        if (main.backBC.args[1] ==  0):
            uB[:] = regionManager.region[main.backBC.args[0]].b.uB[:,:,:,:,:,:,0]
        uB[:] = main.backBC.applyBC(fB[:,:,:,:,:,:,0,:],uB,main.backBC.args,main,main.normals[5,:,:,:,0])

  else:
    
    if (main.rank_connect[4] != main.mpi_rank and ((main.backBC.args[1] == -1 and (main.backBC.BC_type == 'periodic' or main.backBC.BC_type == 'patch') and back_face==True) or back_face==False)):
        main.comm.Send(fB[:,:,:,:,:,:,0].flatten(),dest=main.rank_connect[4],tag=main.mpi_rank+4*main.num_processes_global)

    #============================

    if (main.rank_connect[5] != main.mpi_rank and ((main.frontBC.args[1] == 0 and (main.frontBC.BC_type == 'periodic' or main.frontBC.BC_type == 'patch') and front_Face==True) or front_face==False)):
      tmp = np.zeros(np.size(fB[:,:,:,:,:,:,0]))
      main.comm.Recv(tmp,source=main.rank_connect[5],tag=main.rank_connect[5]+4*main.num_processes_global)
      uF[:] = np.reshape(tmp,np.shape(fB[:,:,:,:,:,:,0]))

    #============================

    if (main.rank_connect[4] != main.mpi_rank and main.backBC.args[1] == 0 and (main.backBC.BC_type == 'periodic' or main.backBC.BC_type == 'patch') and back_face==True):
      tmp = np.zeros(np.size(fB[:,:,:,:,:,:,0]))
      main.comm.Sendrecv(fB[:,:,:,:,:,:,0].flatten(),dest=main.rank_connect[4],sendtag=main.mpi_rank+4*main.num_processes_global,\
                         recvbuf=tmp,source=main.rank_connect[4],recvtag=main.rank_connect[4]+4*main.num_processes_global)
      uB[:] = np.reshape(tmp,np.shape(fB[:,:,:,:,:,:,0]))

    #============================





    if (main.rank_connect[5] != main.mpi_rank and ((main.frontBC.args[1] == 0 and (main.frontBC.BC_type == 'periodic' or main.frontBC.BC_type == 'patch') and front_face==True) or front_face==False)):
      main.comm.Send(fF[:,:,:,:,:,:,-1].flatten(),dest=main.rank_connect[5],tag=main.mpi_rank+5*main.num_processes_global)

    #============================

    if (main.rank_connect[4] != main.mpi_rank and ((main.backBC.args[1] == -1 and (main.backBC.BC_type == 'periodic' or main.backBC.BC_type == 'patch') and back_face==True) or back_face==False)):
      tmp = np.zeros(np.size(fB[:,:,:,:,:,:,0]))
      main.comm.Recv(tmp,source=main.rank_connect[4],tag=main.rank_connect[4]+5*main.num_processes_global)
      uB[:] = np.reshape(tmp,np.shape(fB[:,:,:,:,:,:,-1]))
    
    #============================

    if (main.rank_connect[5] != main.mpi_rank and main.frontBC.args[1] == -1 and (main.frontBC.BC_type == 'periodic' or main.frontBC.BC_type == 'patch') and front_face==True):
      tmp = np.zeros(np.size(fB[:,:,:,:,:,:,0]))
      main.comm.Sendrecv(fF[:,:,:,:,:,:,-1].flatten(),dest=main.rank_connect[5],sendtag=main.mpi_rank+5*main.num_processes_global,\
                         recvbuf=tmp,source=main.rank_connect[5],recvtag=main.rank_connect[5]+5*main.num_processes_global)
      uF[:] = np.reshape(tmp,np.shape(fB[:,:,:,:,:,:,-1]))
    
    #============================





    if (main.BC_rank[5] and main.frontBC.BC_type != 'periodic' and main.frontBC.BC_type != 'patch'): 
      uF[:] = fF[:,:,:,:,:,:,-1]
    elif (main.BC_rank[5] and (main.frontBC.BC_type == 'periodic' or main.backBC.BC_type == 'patch')):
      uF[:] = main.frontBC.applyBC(fF[:,:,:,:,:,:,-1],uF,main.frontBC.args,main,main.normals[4,:,:,:,-1])

    if (main.BC_rank[4] and main.backBC.BC_type != 'periodic'  and main.backBC.BC_type != 'patch'):
      uB[:] = fB[:,:,:,:,:,:,0]
    elif (main.BC_rank[4] and (main.fontBC.BC_type == 'periodic' or main.backBC.BC_type == 'patch')):
      uB[:] = main.backBC.applyBC(fB[:,:,:,:,:,:,0],uB,main.backBC.args,main,main.normals[5,:,:,:,0])
    
  
  
  return uR,uL,uU,uD,uF,uB





##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
##########################################################################################################################################################################################################





def gatherSolScalar(main,u):
  if (main.mpi_rank == main.starting_rank):
    
    uG = np.zeros((main.quadpoints[0],main.quadpoints[1],main.quadpoints[2],main.quadpoints[3],main.Nel[0],main.Nel[1],main.Nel[2],main.Nel[3]))
    uG[:,:,:,:,0:main.Npx,0:main.Npy,0:main.Npz] = u[:]
    
    for i in main.all_mpi_ranks[1::]:
      loc_rank = i - main.starting_rank
      data = np.zeros(np.shape(u)).flatten()
      main.comm.Recv(data,source=loc_rank + main.starting_rank,tag = loc_rank + main.starting_rank)
      
      xL = (int(loc_rank)%int(main.procx*main.procy))%int(main.procx)*main.Npx
      xR = (int(loc_rank)%int(main.procx*main.procy))%int(main.procx)*main.Npx + main.Npx
      yD = (int(loc_rank)%int(main.procx*main.procy))//int(main.procx)*main.Npy
      yU = (int(loc_rank)%int(main.procx*main.procy))//int(main.procx)*main.Npy + main.Npy
      zB = (int(loc_rank)//int(main.procx*main.procy))*main.Npz
      zF = (int(loc_rank)//int(main.procx*main.procy))*main.Npz + main.Npz
      
      uG[:,:,:,:,xL:xR,yD:yU,zB:zF] = np.reshape(data,np.shape(u))
    
    return uG

  else:
    main.comm.Send(u.flatten(),dest=main.starting_rank,tag=main.mpi_rank)


def gatherMassMatrix(main,M):
  if (main.mpi_rank == main.starting_rank):
    MG = np.zeros((main.order[0],main.order[1],main.order[2],main.order[3],main.order[0],main.order[1],main.order[2],main.order[3],main.Nel[0],main.Nel[1],main.Nel[2],main.Nel[3]))
    MG[:,:,:,:,:,:,:,:,0:main.Npx,0:main.Npy,0:main.Npz] = M[:]
    for i in main.all_mpi_ranks[1::]:
      loc_rank = i - main.starting_rank
      data = np.zeros(np.shape(M)).flatten()
      main.comm.Recv(data,source=loc_rank + main.starting_rank,tag = loc_rank + main.starting_rank)
      xL = (int(loc_rank)%int(main.procx*main.procy))%int(main.procx)*main.Npx
      xR = (int(loc_rank)%int(main.procx*main.procy))%int(main.procx)*main.Npx + main.Npx
      yD = (int(loc_rank)%int(main.procx*main.procy))//int(main.procx)*main.Npy
      yU = (int(loc_rank)%int(main.procx*main.procy))//int(main.procx)*main.Npy + main.Npy
      zB = (int(loc_rank)//int(main.procx*main.procy))*main.Npz
      zF = (int(loc_rank)//int(main.procx*main.procy))*main.Npz + main.Npz
      MG[:,:,:,:,:,:,:,:,xL:xR,yD:yU,zB:zF] = np.reshape(data,(main.order[0],main.order[1],main.order[2],main.order[3],main.order[0],main.order[1],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    return MG
  else:
    main.comm.Send(M.flatten(),dest=main.starting_rank,tag=main.mpi_rank)





def gatherSolSlab(main,eqns,var):
  if (main.mpi_rank == main.starting_rank):
    
    uG = np.zeros((var.nvars,var.quadpoints[0],var.quadpoints[1],var.quadpoints[2],var.quadpoints[3],main.Nel[0],main.Nel[1],main.Nel[2],main.Nel[3]))
    uG[:,:,:,:,:,0:main.Npx,0:main.Npy,0:main.Npz,:] = var.u[:]

    for i in main.all_mpi_ranks[1::]:
      loc_rank = i - main.starting_rank
      data = np.zeros(np.shape(var.u)).flatten()
      main.comm.Recv(data,source=loc_rank + main.starting_rank,tag = loc_rank + main.starting_rank)
      xL = (int(loc_rank)%int(main.procx*main.procy))%int(main.procx)*main.Npx
      xR = (int(loc_rank)%int(main.procx*main.procy))%int(main.procx)*main.Npx + main.Npx
      yD = (int(loc_rank)%int(main.procx*main.procy))//int(main.procx)*main.Npy
      yU = (int(loc_rank)%int(main.procx*main.procy))//int(main.procx)*main.Npy + main.Npy
      zB = (int(loc_rank)//int(main.procx*main.procy))*main.Npz
      zF = (int(loc_rank)//int(main.procx*main.procy))*main.Npz + main.Npz
      uG[:,:,:,:,:,xL:xR,yD:yU,zB:zF] = np.reshape(data,np.shape(main.a.u))
    return uG
  
  else:
    main.comm.Send(var.u.flatten(),dest=main.starting_rank,tag=main.mpi_rank)






def gatherSolSpectral(a,main):
  if (main.mpi_rank == main.starting_rank):
    
    aG = np.zeros((main.nvars,main.order[0],main.order[1],main.order[2],main.order[3],main.Nel[0],main.Nel[1],main.Nel[2],main.Nel[3]))
    aG[:,:,:,:,:,0:main.Npx,0:main.Npy,:] = a[:]
    
    for i in main.all_mpi_ranks[1::]:
      loc_rank = i - main.starting_rank
      data = np.zeros(np.shape(a)).flatten()
      main.comm.Recv(data,source=loc_rank + main.starting_rank,tag = loc_rank + main.starting_rank)

      xL = (int(loc_rank)%int(main.procx*main.procy))%int(main.procx)*main.Npx
      xR = (int(loc_rank)%int(main.procx*main.procy))%int(main.procx)*main.Npx + main.Npx
      yD = (int(loc_rank)%int(main.procx*main.procy))//int(main.procx)*main.Npy
      yU = (int(loc_rank)%int(main.procx*main.procy))//int(main.procx)*main.Npy + main.Npy
      zB = (int(loc_rank)//int(main.procx*main.procy))*main.Npz
      zF = (int(loc_rank)//int(main.procx*main.procy))*main.Npz + main.Npz
      
      aG[:,:,:,:,:,xL:xR,yD:yU,zB:zF] = np.reshape(data,(main.nvars,main.order[0],main.order[1],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    
    return aG
  
  else:
    main.comm.Send(a.flatten(),dest=main.starting_rank,tag=main.mpi_rank)






def regionConnector(regionManager):
  reg_counter = 0
  for reg_index in regionManager.mpi_regions_owned:
    region = regionManager.region[reg_counter]
    reg_counter += 1
    #region = regionManager.region[2]

    #=============================================================================================================================
    
    if (region.leftBC.BC_type == 'patch' and region.BC_rank[0]):


      
      region_connect         = region.leftBC.args[0]
      rel_x		     = region.leftBC.args[1]
      rel_y		     = region.leftBC.args[2]
      rel_z		     = region.leftBC.args[3]

    #=========== Do stuff to learn orientations    
#      if (np.allclose(region.y[0,:,:] , regionManager.region[region_connect].y[rel_x,:,:] ) ):
#        print('Y: Regions ' + str(region.region_number) + ' and ' + str(region_connect) + ' aligned' ) 
#      else:
#        print('Y: Regions ' + str(region.region_number) + ' and ' + str(region_connect) + ' not aligned' ) 
#      if (np.allclose(region.z[0,:,:] , regionManager.region[region_connect].z[rel_x,:,:] ) ):
#        print('Z: Regions ' + str(region.region_number) + ' and ' + str(region_connect) + ' aligned' ) 
#      else:
#        print('Z: Regions ' + str(region.region_number) + ' and ' + str(region_connect) + ' not aligned' ) 

      sr                     = regionManager.starting_rank[region_connect]
      px                     = regionManager.procx[region_connect]
      py                     = regionManager.procy[region_connect]
      pz                     = regionManager.procz[region_connect]

      rowx                   = ((int(region.mpi_rank) - int(region.starting_rank)) % int(region.procx*region.procy)) % int(region.procx)
      rowy                   = ((int(region.mpi_rank) - int(region.starting_rank)) % int(region.procx*region.procy)) // int(region.procx)
      rowz                   =  (int(region.mpi_rank) - int(region.starting_rank)) // int(region.procx*region.procy)
      
      if (rel_x==-1):
	
        offset = px-1

      elif (rel_x==0):

        offset = 0

      if (rel_y==0):
        region.leftBC.ydir = 1
        pos_y  = rowy

      elif (rel_y==1):
        region.leftBC.ydir = -1
        pos_y  = py-rowy-1

      if (rel_z==0):
        region.leftBC.zdir = 1
        pos_z  = rowz

      elif (rel_z==1):
        region.leftBC.zdir = -1
        pos_z  = pz-rowz-1

      region.rank_connect[0] = sr + px*((pos_z*py) + pos_y) + offset
    #=============================================================================================================================

    if (region.rightBC.BC_type == 'patch' and region.BC_rank[1]):
      
      region_connect         = region.rightBC.args[0]
      rel_x		     = region.rightBC.args[1]
      rel_y		     = region.rightBC.args[2]
      rel_z		     = region.rightBC.args[3]


#      if (np.allclose(region.y[-1,:,:] , regionManager.region[region_connect].y[rel_x,:,:] ) ):
#        print('Y: Regions ' + str(region.region_number) + ' and ' + str(region_connect) + ' aligned' ) 
#      else:
#        print('Y: Regions ' + str(region.region_number) + ' and ' + str(region_connect) + ' not aligned' ) 
#      if (np.allclose(region.z[-1,:,:] , regionManager.region[region_connect].z[rel_x,:,:] ) ):
#        print('Z: Regions ' + str(region.region_number) + ' and ' + str(region_connect) + ' aligned' ) 
#      else:
#        print('Z: Regions ' + str(region.region_number) + ' and ' + str(region_connect) + ' not aligned' ) 

      sr                     = regionManager.starting_rank[region_connect]
      px                     = regionManager.procx[region_connect]
      py                     = regionManager.procy[region_connect]
      pz                     = regionManager.procz[region_connect]

      rowx                   = ((int(region.mpi_rank) - int(region.starting_rank)) % int(region.procx*region.procy)) % int(region.procx)
      rowy                   = ((int(region.mpi_rank) - int(region.starting_rank)) % int(region.procx*region.procy)) // int(region.procx)
      rowz                   =  (int(region.mpi_rank) - int(region.starting_rank)) // int(region.procx*region.procy)
      
      if (rel_x==-1):
	
        offset = px-1

      elif (rel_x==0):

        offset = 0

      if (rel_y==0):
        region.rightBC.ydir = 1
        pos_y  = rowy

      elif (rel_y==1):
        region.rightBC.ydir = -1
        pos_y  = py-rowy-1

      if (rel_z==0):
        region.rightBC.zdir = 1
        pos_z  = rowz

      elif (rel_z==1):
        region.rightBC.zdir = -1
        pos_z  = pz-rowz-1

      region.rank_connect[1] = sr + px*((pos_z*py) + pos_y) + offset

    #=============================================================================================================================
    
    if (region.bottomBC.BC_type == 'patch' and region.BC_rank[2]):
      
      region_connect         = region.bottomBC.args[0]
      rel_y		     = region.bottomBC.args[1]
      rel_x		     = region.bottomBC.args[2]
      rel_z		     = region.bottomBC.args[3]

#      if (np.allclose(region.x[:,0,:] , regionManager.region[region_connect].x[:,rel_y,:] ) ):
#        print('X: Regions ' + str(region.region_number) + ' and ' + str(region_connect) + ' aligned' ) 
#      else:
#        print('X: Regions ' + str(region.region_number) + ' and ' + str(region_connect) + ' not aligned' ) 
#      if (np.allclose(region.z[:,0,:] , regionManager.region[region_connect].z[:,rel_y,:] ) ):
#        print('Z: Regions ' + str(region.region_number) + ' and ' + str(region_connect) + ' aligned' ) 
#      else:
#        print('Z: Regions ' + str(region.region_number) + ' and ' + str(region_connect) + ' not aligned' ) 


      sr                     = regionManager.starting_rank[region_connect]
      px                     = regionManager.procx[region_connect]
      py                     = regionManager.procy[region_connect]
      pz                     = regionManager.procz[region_connect]

      rowx                   = ((int(region.mpi_rank) - int(region.starting_rank)) % int(region.procx*region.procy)) % int(region.procx)
      rowy                   = ((int(region.mpi_rank) - int(region.starting_rank)) % int(region.procx*region.procy)) // int(region.procx)
      rowz                   =  (int(region.mpi_rank) - int(region.starting_rank)) // int(region.procx*region.procy)
      
      if (rel_y==-1):
	
        offset = px*py-px

      elif (rel_y==0):

        offset = 0

      if (rel_x==0):
        region.bottomBC.xdir = 1
        pos_x  = rowx

      elif (rel_x==1):
        region.bottomBC.xdir = -1 
        pos_x  = px-rowx-1

      if (rel_z==0):
        pos_z  = rowz
        region.bottomBC.zdir = 1

      elif (rel_z==1):
        pos_z  = pz-rowz-1
        region.bottomBC.zdir = -1

      region.rank_connect[2] = sr + (pos_z*(px*py) + pos_x) + offset
    
    #=============================================================================================================================
    
    if (region.topBC.BC_type == 'patch' and region.BC_rank[3]):
      
      region_connect         = region.topBC.args[0]
      rel_y		     = region.topBC.args[1]
      rel_x		     = region.topBC.args[2]
      rel_z		     = region.topBC.args[3]

#      if (np.allclose(region.x[:,-1,:] , regionManager.region[region_connect].x[:,rel_y,:] ) ):
#        print('X: Regions ' + str(region.region_number) + ' and ' + str(region_connect) + ' aligned' ) 
#      else:
#        print('X: Regions ' + str(region.region_number) + ' and ' + str(region_connect) + ' not aligned' ) 
#      if (np.allclose(region.z[:,-1,:] , regionManager.region[region_connect].z[:,rel_y,:] ) ):
#        print('Z: Regions ' + str(region.region_number) + ' and ' + str(region_connect) + ' aligned' ) 
#      else:
#        print('Z: Regions ' + str(region.region_number) + ' and ' + str(region_connect) + ' not aligned' ) 
#

      sr                     = regionManager.starting_rank[region_connect]
      px                     = regionManager.procx[region_connect]
      py                     = regionManager.procy[region_connect]
      pz                     = regionManager.procz[region_connect]

      rowx                   = ((int(region.mpi_rank) - int(region.starting_rank)) % int(region.procx*region.procy)) % int(region.procx)
      rowy                   = ((int(region.mpi_rank) - int(region.starting_rank)) % int(region.procx*region.procy)) // int(region.procx)
      rowz                   =  (int(region.mpi_rank) - int(region.starting_rank)) // int(region.procx*region.procy)
      
      if (rel_y==-1):
	
        offset = px*py-px

      elif (rel_y==0):

        offset = 0

      if (rel_x==0):
        region.topBC.xdir = 1
        pos_x  = rowx

      elif (rel_x==1):
        region.topBC.xdir = -1
        pos_x  = px-rowx-1

      if (rel_z==0):
        region.topBC.zdir = 1
        pos_z  = rowz

      elif (rel_z==1):
        region.topBC.zdir = -1
        pos_z  = pz-rowz-1

      region.rank_connect[3] = sr + (pos_z*(px*py) + pos_x) + offset
    
    #=============================================================================================================================
    
    if (region.backBC.BC_type == 'patch' and region.BC_rank[4]):
      
      region_connect         = region.backBC.args[0]
      rel_z		     = region.backBC.args[1]
      rel_x		     = region.backBC.args[2]
      rel_y		     = region.backBC.args[3]

#      if (np.allclose(region.x[:,:,0] , regionManager.region[region_connect].x[:,:,rel_z] ) ):
#        print('X: Regions ' + str(region.region_number) + ' and ' + str(region_connect) + ' aligned' ) 
#      else:
#        print('X: Regions ' + str(region.region_number) + ' and ' + str(region_connect) + ' not aligned' ) 
#      if (np.allclose(region.y[:,:,0] , regionManager.region[region_connect].y[:,:,rel_z] ) ):
#        print('Y: Regions ' + str(region.region_number) + ' and ' + str(region_connect) + ' aligned' ) 
#      else:
#        print('Y: Regions ' + str(region.region_number) + ' and ' + str(region_connect) + ' not aligned' ) 


      sr                     = regionManager.starting_rank[region_connect]
      px                     = regionManager.procx[region_connect]
      py                     = regionManager.procy[region_connect]
      pz                     = regionManager.procz[region_connect]

      rowx                   = ((int(region.mpi_rank) - int(region.starting_rank)) % int(region.procx*region.procy)) % int(region.procx)
      rowy                   = ((int(region.mpi_rank) - int(region.starting_rank)) % int(region.procx*region.procy)) // int(region.procx)
      rowz                   =  (int(region.mpi_rank) - int(region.starting_rank)) // int(region.procx*region.procy)
      
      if (rel_z==-1):
	
        offset = px*py*pz - px*py

      elif (rel_z==0):

        offset = 0

      if (rel_x==0):
        region.backBC.xdir = 1
        pos_x  = rowx

      elif (rel_x==1):
        region.backBC.xdir = -1
        pos_x  = px-rowx-1

      if (rel_y==0):
        region.backBC.ydir = 1
        pos_y  = rowy

      elif (rel_y==1):
        region.backBC.ydir = -1
        pos_y  = py-rowy-1

      region.rank_connect[4] = sr + (pos_y*px) + pos_x + offset
    
    #=============================================================================================================================
    
    if (region.frontBC.BC_type == 'patch' and region.BC_rank[5]):
      
      region_connect         = region.frontBC.args[0]
      rel_z		     = region.frontBC.args[1]
      rel_x		     = region.frontBC.args[2]
      rel_y		     = region.frontBC.args[3]

#      if (np.allclose(region.x[:,:,-1] , regionManager.region[region_connect].x[:,:,rel_z] ) ):
#        print('X: Regions ' + str(region.region_number) + ' and ' + str(region_connect) + ' aligned' ) 
#      else:
#        print('X: Regions ' + str(region.region_number) + ' and ' + str(region_connect) + ' not aligned' ) 
#      if (np.allclose(region.y[:,:,-1] , regionManager.region[region_connect].y[:,:,rel_z] ) ):
#        print('Y: Regions ' + str(region.region_number) + ' and ' + str(region_connect) + ' aligned' ) 
#      else:
#        print('Y: Regions ' + str(region.region_number) + ' and ' + str(region_connect) + ' not aligned' ) 
#

      sr                     = regionManager.starting_rank[region_connect]
      px                     = regionManager.procx[region_connect]
      py                     = regionManager.procy[region_connect]
      pz                     = regionManager.procz[region_connect]

      rowx                   = ((int(region.mpi_rank) - int(region.starting_rank)) % int(region.procx*region.procy)) % int(region.procx)
      rowy                   = ((int(region.mpi_rank) - int(region.starting_rank)) % int(region.procx*region.procy)) // int(region.procx)
      rowz                   =  (int(region.mpi_rank) - int(region.starting_rank)) // int(region.procx*region.procy)
      
      if (rel_z==-1):
	
        offset = px*py*pz - px*py

      elif (rel_z==0):

        offset = 0

      if (rel_x==0):
        region.frontBC.xdir = 1
        pos_x  = rowx

      elif (rel_x==1):
        region.frontBC.xdir = -1
        pos_x  = px-rowx-1

      if (rel_y==0):
        region.frontBC.ydir = 1
        pos_y  = rowy

      elif (rel_y==1):
        region.frontBC.ydir = -1
        pos_y  = py-rowy-1

      region.rank_connect[5] = sr + (pos_y*px) + pos_x + offset
    
    #=============================================================================================================================

    if ( (region.BC_rank[0] and region.leftBC.BC_type != 'patch') and region.leftBC.BC_type != 'periodic' ):
      region.rank_connect[0] = region.mpi_rank
    
    if ( (region.BC_rank[1] and region.rightBC.BC_type != 'patch') and region.rightBC.BC_type != 'periodic' ):
      region.rank_connect[1] = region.mpi_rank

    if ( (region.BC_rank[2] and region.bottomBC.BC_type != 'patch') and region.bottomBC.BC_type != 'periodic' ):
      region.rank_connect[2] = region.mpi_rank

    if ( (region.BC_rank[3] and region.topBC.BC_type != 'patch') and region.topBC.BC_type != 'periodic' ):
      region.rank_connect[3] = region.mpi_rank

    if ( (region.BC_rank[4] and region.backBC.BC_type != 'patch') and region.bottomBC.BC_type != 'periodic' ):
      region.rank_connect[4] = region.mpi_rank

    if ( (region.BC_rank[5] and region.frontBC.BC_type != 'patch') and region.topBC.BC_type != 'periodic' ):
      region.rank_connect[5] = region.mpi_rank






def getRankConnectionsSlab(mpi_rank,num_processes,procx,procy,procz,starting_rank):
  ##============== MPI INFORMATION ===================
#  if (procx*procy != num_processes):
#    if (mpi_rank == 0):
#      print('Error, correct x/y proc decomposition, now quitting!')
#    sys.exit()
  
  rank_connect = np.zeros((6)) ##I'm an idiot and ordering for this is left right bottom top
  
  rank_connect[0] = mpi_rank-1
  rank_connect[1] = mpi_rank+1
  rank_connect[2] = mpi_rank-procx
  rank_connect[3] = mpi_rank+procx
  rank_connect[4] = mpi_rank-procx*procy
  rank_connect[5] = mpi_rank+procx*procy

  BC_rank = [False,False,False,False,False,False] #ordering is right, top, left, bottom  
  
  if ((mpi_rank-starting_rank)%procx==0):                                   #left face 
    rank_connect[0] = mpi_rank + procx - 1
    BC_rank[0] = True
  
  if ((mpi_rank-starting_rank)%procx==procx-1):                             #right face 
    rank_connect[1] = mpi_rank - procx + 1
    BC_rank[1] = True
  
  if (((mpi_rank-starting_rank)%(procx*procy))//procx==0):                   #bottom face
    rank_connect[2] = mpi_rank + procx*procy - procx
    BC_rank[2] = True

  if (((mpi_rank-starting_rank)%(procx*procy))//procx==procy-1):             #top face
    rank_connect[3] = mpi_rank - procx*procy + procx
    BC_rank[3] = True

  if ((mpi_rank-starting_rank)//(procx*procy)==0):                           #back face
    rank_connect[4] = mpi_rank + procx*procy*procz - procx*procy
    BC_rank[4] = True
  
  if ((mpi_rank-starting_rank)//(procx*procy)==procz-1):                     #front face
    rank_connect[5] = mpi_rank - procx*procy*procz + procx*procy
    BC_rank[5] = True

  return rank_connect,BC_rank
