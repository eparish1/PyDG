import numpy as np
import mpi4py as MPI

#========================================================================================

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

#========================================================================================

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


##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
##########################################################################################################################################################################################################


def sendEdgesGeneralSlab(fL,fR,fD,fU,fB,fF,main,regionManager):
  uR = np.zeros(np.shape(fL[:,:,:,:,0,:,:]))
  uL = np.zeros(np.shape(fR[:,:,:,:,0,:,:]))
  uU = np.zeros(np.shape(fD[:,:,:,:,:,0,:]))
  uD = np.zeros(np.shape(fU[:,:,:,:,:,0,:]))
  uF = np.zeros(np.shape(fB[:,:,:,:,:,:,0]))
  uB = np.zeros(np.shape(fF[:,:,:,:,:,:,0]))

  left_face   = ((main.mpi_rank-main.starting_rank)%main.procx==0)
  right_face  = ((main.mpi_rank-main.starting_rank)%main.procx==main.procx-1)
  
  bottom_face = (((main.mpi_rank-main.starting_rank)%(main.procx*main.procy))/main.procx==0)
  top_face    = (((main.mpi_rank-main.starting_rank)%(main.procx*main.procy))/main.procx==main.procy-1)

  back_face   = ((main.mpi_rank-main.starting_rank)/(main.procx*main.procy)==0)
  front_face  = ((main.mpi_rank-main.starting_rank)/(main.procx*main.procy)==main.procz-1)
  

  ######################################################################################################################################################



  ## If only using one processor ================
  
  if (main.rank_connect[0] == main.mpi_rank and main.rank_connect[1] == main.mpi_rank):
    uR[:] = fL[:,:,:,:, 0,:,:]
    uL[:] = fR[:,:,:,:,-1,:,:]
    
    if (main.rightBC.BC_type == 'patch'):
        if (main.rightBC.args[1] ==  0):
            uR[:] = regionManager.region[main.rightBC.args[0]].a.uL[:,:,:,:, 0,:,:]
        if (main.rightBC.args[1] == -1):
            uR[:] = regionManager.region[main.rightBC.args[0]].a.uR[:,:,:,:,-1,:,:]
    
    uR[:] = main.rightBC.applyBC(fR[:,:,:,:,-1,:,:],uR,main.rightBC.args,main)
    
    if (main.leftBC.BC_type == 'patch'):
        if (main.leftBC.args[1] == -1):
            uL[:] = regionManager.region[main.leftBC.args[0]].a.uR[:,:,:,:,-1,:,:]
        if (main.leftBC.args[1] ==  0):
            uL[:] = regionManager.region[main.leftBC.args[0]].a.uL[:,:,:,:, 0,:,:]
    
    uL[:] = main.leftBC.applyBC(fL[:,:,:,:, 0,:,:],uL,main.leftBC.args,main)
 
  #======================================================
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
      uR[:] = np.reshape(tmp,np.shape(fL[:,:,:,:,-1,:,:]))
    
    #============================

    if (main.BC_rank[1]):
      uR[:] = main.rightBC.applyBC(fR[:,:,:,:,-1,:,:],uR,main.rightBC.args,main)

    if (main.BC_rank[0]):
      uL[:] = main.leftBC.applyBC(fL[:,:,:,:,0 ,:,:],uL,main.leftBC.args,main)



  ######################################################################################################################################################



  ## If only using one processor ================
  if (main.rank_connect[2] == main.mpi_rank and main.rank_connect[3] == main.mpi_rank):
    uU[:] = fD[:,:,:,:,:,0 ,:]
    uD[:] = fU[:,:,:,:,:,-1,:]
    
    if (main.topBC.BC_type == 'patch'):
        if (main.topBC.args[1] ==  0):
            uU[:] = regionManager.region[main.topBC.args[0]].a.uD[:,:,:,:, 0,:,:]
        if (main.topBC.args[1] == -1):
            uU[:] = regionManager.region[main.topBC.args[0]].a.uU[:,:,:,:,-1,:,:]
    
    uU[:] = main.topBC.applyBC(fU[:,:,:,:,:,-1,:],uU,main.topBC.args,main)
    
    if (main.bottomBC.BC_type == 'patch'):
        if (main.bottomBC.args[1] == -1):
            uD[:] = regionManager.region[main.bottomBC.args[0]].a.uU[:,:,:,:,-1,:,:]
        if (main.bottomBC.args[1] ==  0):
            uD[:] = regionManager.region[main.bottomBC.args[0]].a.uD[:,:,:,:, 0,:,:]
    
    uD[:] = main.bottomBC.applyBC(fD[:,:,:,:,:,0,:],uD,main.bottomBC.args,main)

  #======================================================
  else:
    
    if (main.rank_connect[2] != main.mpi_rank and ((main.bottomBC.args[1] == -1 and (main.bottomBC.BC_type == 'periodic' or main.bottomBC.BC_type == 'patch') and bottom_face==True) or bottom_face==False)):
      main.comm.Send(fD[:,:,:,:,:,0,:].flatten(),dest=main.rank_connect[2],tag=main.mpi_rank)

    #============================

    if (main.rank_connect[3] != main.mpi_rank and ((main.topBC.args[1] == 0 and (main.topBC.BC_type == 'periodic' or main.topBC.BC_type == 'patch') and top_face==True) or top_face==False)):
      tmp = np.zeros(np.size(fD[:,:,:,:,:,0,:]))
      main.comm.Recv(tmp,source=main.rank_connect[3],tag=main.rank_connect[3])
      uU[:] = np.reshape(tmp,np.shape(fD[:,:,:,:,:,0,:]))

    #============================

    if (main.rank_connect[2] != main.mpi_rank and main.bottomBC.args[1] == 0 and (main.bottomBC.BC_type == 'periodic' or main.bottomBC.BC_type == 'patch') and bottom_face==True):
      tmp = np.zeros(np.size(fD[:,:,:,:,:,0,:]))
      main.comm.Sendrecv(fD[:,:,:,:,:,0,:].flatten(),dest=main.rank_connect[2],sendtag=main.mpi_rank,\
                         recvbuf=tmp,source=main.rank_connect[2],recvtag=main.rank_connect[2])
      uD[:] = np.reshape(tmp,np.shape(fD[:,:,:,:,:,0,:]))

    #============================





    if (main.rank_connect[3] != main.mpi_rank and ((main.topBC.args[1] == 0 and (main.topBC.BC_type == 'periodic' or main.topBC.BC_type == 'patch') and top_face==True) or top_face==False)):
      main.comm.Send(fU[:,:,:,:,:,-1,:].flatten(),dest=main.rank_connect[3],tag=main.mpi_rank+main.num_processes_global)

    #============================

    if (main.rank_connect[2] != main.mpi_rank and ((main.bottomBC.args[1] == -1 and (main.bottomBC.BC_type == 'periodic' or main.bottomBC.BC_type == 'patch') and bottom_face==True) or bottom_face==False)):
      tmp = np.zeros(np.size(fD[:,:,:,:,:,0,:]))
      main.comm.Recv(tmp,source=main.rank_connect[2],tag=main.rank_connect[2]+main.num_processes_global)
      uD[:] = np.reshape(tmp,np.shape(fD[:,:,:,:,:,-1,:]))
    
    #============================

    if (main.rank_connect[3] != main.mpi_rank and main.topBC.args[1] == -1 and (main.topBC.BC_type == 'periodic' or main.topBC.BC_type == 'patch') and top_face==True):
      tmp = np.zeros(np.size(fD[:,:,:,:,:,0,:]))
      main.comm.Sendrecv(fU[:,:,:,:,:,-1,:].flatten(),dest=main.rank_connect[3],sendtag=main.mpi_rank+main.num_processes_global,\
                         recvbuf=tmp,source=main.rank_connect[3],recvtag=main.rank_connect[3]+main.num_processes_global)
      uU[:] = np.reshape(tmp,np.shape(fD[:,:,:,:,:,-1,:]))
    
    #============================

    if (main.BC_rank[3]):
      uU[:] = main.topBC.applyBC(fU[:,:,:,:,:,-1,:],uU,main.topBC.args,main)

    if (main.BC_rank[2]):
      uD[:] = main.bottomBC.applyBC(fD[:,:,:,:,:,0,:],uD,main.bottomBC.args,main)



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
    
    uF[:] = main.frontBC.applyBC(fF[:,:,:,:,:,:,-1],uF,main.frontBC.args,main)
    
    if (main.backBC.BC_type == 'patch'):
        if (main.backBC.args[1] == -1):
            uB[:] = regionManager.region[main.backBC.args[0]].a.uF[:,:,:,:,:,:,-1]
        if (main.backBC.args[1] ==  0):
            uB[:] = regionManager.region[main.backBC.args[0]].a.uB[:,:,:,:,:,:,0]
    
    uB[:] = main.backBC.applyBC(fB[:,:,:,:,:,0,:],uB,main.backBC.args,main)

  #======================================================
  else:
    
    if (main.rank_connect[4] != main.mpi_rank and ((main.backBC.args[1] == -1 and (main.backBC.BC_type == 'periodic' or main.backBC.BC_type == 'patch') and back_face==True) or back_face==False)):
        main.comm.Send(fB[:,:,:,:,:,:,0].flatten(),dest=main.rank_connect[4],tag=main.mpi_rank)

    #============================

    if (main.rank_connect[5] != main.mpi_rank and ((main.frontBC.args[1] == 0 and (main.frontBC.BC_type == 'periodic' or main.frontBC.BC_type == 'patch') and front_Face==True) or front_face==False)):
      tmp = np.zeros(np.size(fB[:,:,:,:,:,:,0]))
      main.comm.Recv(tmp,source=main.rank_connect[5],tag=main.rank_connect[5])
      uF[:] = np.reshape(tmp,np.shape(fB[:,:,:,:,:,:,0]))

    #============================

    if (main.rank_connect[4] != main.mpi_rank and main.backBC.args[1] == 0 and (main.backBC.BC_type == 'periodic' or main.backBC.BC_type == 'patch') and back_face==True):
      tmp = np.zeros(np.size(fB[:,:,:,:,:,:,0]))
      main.comm.Sendrecv(fB[:,:,:,:,:,:,0].flatten(),dest=main.rank_connect[4],sendtag=main.mpi_rank,\
                         recvbuf=tmp,source=main.rank_connect[4],recvtag=main.rank_connect[4])
      uB[:] = np.reshape(tmp,np.shape(fB[:,:,:,:,:,:,0]))

    #============================





    if (main.rank_connect[5] != main.mpi_rank and ((main.frontBC.args[1] == 0 and (main.frontBC.BC_type == 'periodic' or main.frontBC.BC_type == 'patch') and front_face==True) or front_face==False)):
      main.comm.Send(fF[:,:,:,:,:,:,-1].flatten(),dest=main.rank_connect[5],tag=main.mpi_rank+main.num_processes_global)

    #============================

    if (main.rank_connect[4] != main.mpi_rank and ((main.backBC.args[1] == -1 and (main.backBC.BC_type == 'periodic' or main.backBC.BC_type == 'patch') and back_face==True) or back_face==False)):
      tmp = np.zeros(np.size(fB[:,:,:,:,:,:,0]))
      main.comm.Recv(tmp,source=main.rank_connect[4],tag=main.rank_connect[4]+main.num_processes_global)
      uB[:] = np.reshape(tmp,np.shape(fB[:,:,:,:,:,:,-1]))
    
    #============================

    if (main.rank_connect[5] != main.mpi_rank and main.frontBC.args[1] == -1 and (main.frontBC.BC_type == 'periodic' or main.frontBC.BC_type == 'patch') and front_face==True):
      tmp = np.zeros(np.size(fB[:,:,:,:,:,:,0]))
      main.comm.Sendrecv(fF[:,:,:,:,:,:,-1].flatten(),dest=main.rank_connect[5],sendtag=main.mpi_rank+main.num_processes_global,\
                         recvbuf=tmp,source=main.rank_connect[5],recvtag=main.rank_connect[5]+main.num_processes_global)
      uF[:] = np.reshape(tmp,np.shape(fB[:,:,:,:,:,:,-1]))
    
    #============================

    if (main.BC_rank[5]):
      uF[:] = main.frontBC.applyBC(fF[:,:,:,:,:,:,-1],uF,main.frontBC.args,main)

    if (main.BC_rank[4]):
      uB[:] = main.backBC.applyBC(fB[:,:,:,:,:,:,0],uB,main.backBC.args,main)


  return uR,uL,uU,uD,uF,uB


##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
##########################################################################################################################################################################################################


def sendEdgesGeneralSlab_Derivs(fL,fR,fD,fU,fB,fF,main,regionManager):
  uR = np.zeros(np.shape(fL[:,:,:,:,0,:,:]))
  uL = np.zeros(np.shape(fR[:,:,:,:,0,:,:]))
  uU = np.zeros(np.shape(fD[:,:,:,:,:,0,:]))
  uD = np.zeros(np.shape(fU[:,:,:,:,:,0,:]))
  uF = np.zeros(np.shape(fB[:,:,:,:,:,:,0]))
  uB = np.zeros(np.shape(fF[:,:,:,:,:,:,0]))
 
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
        uR[:] = main.rightBC.applyBC(fR[:,:,:,:,-1,:,:],uR,main.rightBC.args,main)
    
    if (main.leftBC.BC_type == 'patch'):
        if (main.leftBC.args[1] == -1):
            uL[:] = regionManager.region[main.leftBC.args[0]].b.uR[:,:,:,:,-1,:,:]
        if (main.leftBC.args[1] ==  0):
            uL[:] = regionManager.region[main.leftBC.args[0]].b.uL[:,:,:,:, 0,:,:]
        uL[:] = main.leftBC.applyBC(fL[:,:,:,:,0,:,:],uL,main.leftBC.args,main)

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
      uR[:] = np.reshape(tmp,np.shape(fL[:,:,:,:,-1,:,:]))
    
    #============================





    if (main.BC_rank[1] and main.rightBC.BC_type != 'periodic' and main.rightBC.BC_type != 'patch'):
      uR[:] = fR[:,:,:,:,-1,:,:]
    elif (main.BC_rank[1] and (main.rightBC.BC_type == 'periodic' or main.leftBC.BC_type == 'patch')):
      uR[:] = main.rightBC.applyBC(fR[:,:,:,:,-1,:,:],uR,main.rightBC.args,main)
    
    if (main.BC_rank[0] and main.leftBC.BC_type != 'periodic' and main.leftBC.BC_type != 'patch'):
      uL[:] = fL[:,:,:,:,0 ,:,:]
    elif (main.BC_rank[0] and (main.rightBC.BC_type == 'periodic' or main.leftBC.BC_type == 'patch')):
      uL[:] = main.leftBC.applyBC(fL[:,:,:,:,0,:,:],uL,main.leftBC.args,main)


  
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
            uU[:] = regionManager.region[main.topBC.args[0]].a.uD[:,:,:,:, 0,:,:]
        if (main.topBC.args[1] == -1):
            uU[:] = regionManager.region[main.topBC.args[0]].a.uU[:,:,:,:,-1,:,:]
        uU[:] = main.topBC.applyBC(fU[:,:,:,:,:,-1,:],uU,main.topBC.args,main)
    
    if (main.bottomBC.BC_type == 'patch'):
        if (main.bottomBC.args[1] == -1):
            uD[:] = regionManager.region[main.bottomBC.args[0]].a.uU[:,:,:,:,-1,:,:]
        if (main.bottomBC.args[1] ==  0):
            uD[:] = regionManager.region[main.bottomBC.args[0]].a.uD[:,:,:,:, 0,:,:]
        uD[:] = main.bottomBC.applyBC(fD[:,:,:,:,:,0,:],uD,main.bottomBC.args,main)

  else:
    
    if (main.rank_connect[2] != main.mpi_rank and ((main.bottomBC.args[1] == -1 and (main.bottomBC.BC_type == 'periodic' or main.bottomBC.BC_type == 'patch') and bottom_face==True) or bottom_face==False)):
      main.comm.Send(fD[:,:,:,:,:,0,:].flatten(),dest=main.rank_connect[2],tag=main.mpi_rank)

    #============================

    if (main.rank_connect[3] != main.mpi_rank and ((main.topBC.args[1] == 0 and (main.topBC.BC_type == 'periodic' or main.topBC.BC_type == 'patch') and top_face==True) or top_face==False)):
      tmp = np.zeros(np.size(fD[:,:,:,:,:,0,:]))
      main.comm.Recv(tmp,source=main.rank_connect[3],tag=main.rank_connect[3])
      uU[:] = np.reshape(tmp,np.shape(fD[:,:,:,:,:,0,:]))

    #============================

    if (main.rank_connect[2] != main.mpi_rank and main.bottomBC.args[1] == 0 and (main.bottomBC.BC_type == 'periodic' or main.bottomBC.BC_type == 'patch') and bottom_face==True):
      tmp = np.zeros(np.size(fD[:,:,:,:,:,0,:]))
      main.comm.Sendrecv(fD[:,:,:,:,:,0,:].flatten(),dest=main.rank_connect[2],sendtag=main.mpi_rank,\
                         recvbuf=tmp,source=main.rank_connect[2],recvtag=main.rank_connect[2])
      uD[:] = np.reshape(tmp,np.shape(fD[:,:,:,:,:,0,:]))

    #============================





    if (main.rank_connect[3] != main.mpi_rank and ((main.topBC.args[1] == 0 and (main.topBC.BC_type == 'periodic' or main.topBC.BC_type == 'patch') and top_face==True) or top_face==False)):
      main.comm.Send(fU[:,:,:,:,:,-1,:].flatten(),dest=main.rank_connect[3],tag=main.mpi_rank+main.num_processes_global)

    #============================

    if (main.rank_connect[2] != main.mpi_rank and ((main.bottomBC.args[1] == -1 and (main.bottomBC.BC_type == 'periodic' or main.bottomBC.BC_type == 'patch') and bottom_face==True) or bottom_face==False)):
      tmp = np.zeros(np.size(fD[:,:,:,:,:,0,:]))
      main.comm.Recv(tmp,source=main.rank_connect[2],tag=main.rank_connect[2]+main.num_processes_global)
      uD[:] = np.reshape(tmp,np.shape(fD[:,:,:,:,:,-1,:]))
    
    #============================

    if (main.rank_connect[3] != main.mpi_rank and main.topBC.args[1] == -1 and (main.topBC.BC_type == 'periodic' or main.topBC.BC_type == 'patch') and top_face==True):
      tmp = np.zeros(np.size(fD[:,:,:,:,:,0,:]))
      main.comm.Sendrecv(fU[:,:,:,:,:,-1,:].flatten(),dest=main.rank_connect[3],sendtag=main.mpi_rank+main.num_processes_global,\
                         recvbuf=tmp,source=main.rank_connect[3],recvtag=main.rank_connect[3]+main.num_processes_global)
      uU[:] = np.reshape(tmp,np.shape(fD[:,:,:,:,:,-1,:]))
    
    #============================





    if (main.BC_rank[3] and main.topBC.BC_type != 'periodic' and main.topBC.BC_type != 'patch'): 
      uU[:] = fU[:,:,:,:,:,-1,:]
    elif (main.BC_rank[3] and (main.rightBC.BC_type == 'periodic' or main.leftBC.BC_type == 'patch')):
      uU[:] = main.topBC.applyBC(fU[:,:,:,:,:,-1,:],uU,main.topBC.args,main)

    if (main.BC_rank[2] and main.bottomBC.BC_type != 'periodic' and main.bottomBC.BC_type != 'patch'):
      uD[:] = fD[:,:,:,:,:,0,:]
    elif (main.BC_rank[2] and (main.rightBC.BC_type == 'periodic' or main.leftBC.BC_type == 'patch')):
      uD[:] = main.bottomBC.applyBC(fD[:,:,:,:,:,0,:],uD,main.bottomBC.args,main)



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
            uF[:] = regionManager.region[main.frontBC.args[0]].a.uB[:,:,:,:,:,:,0]
        if (main.frontBC.args[1] == -1):
            uF[:] = regionManager.region[main.frontBC.args[0]].a.uF[:,:,:,:,:,:,-1]
        uF[:] = main.frontBC.applyBC(fF[:,:,:,:,:,:,-1],uF,main.frontBC.args,main)
    
    if (main.backBC.BC_type == 'patch'):
        if (main.backBC.args[1] == -1):
            uB[:] = regionManager.region[main.backBC.args[0]].a.uF[:,:,:,:,:,:,-1]
        if (main.backBC.args[1] ==  0):
            uB[:] = regionManager.region[main.backBC.args[0]].a.uB[:,:,:,:,:,:,0]
        uB[:] = main.backBC.applyBC(fB[:,:,:,:,:,0,:],uB,main.backBC.args,main)

  else:
    
    if (main.rank_connect[4] != main.mpi_rank and ((main.backBC.args[1] == -1 and (main.backBC.BC_type == 'periodic' or main.backBC.BC_type == 'patch') and back_face==True) or back_face==False)):
        main.comm.Send(fB[:,:,:,:,:,:,0].flatten(),dest=main.rank_connect[4],tag=main.mpi_rank)

    #============================

    if (main.rank_connect[5] != main.mpi_rank and ((main.frontBC.args[1] == 0 and (main.frontBC.BC_type == 'periodic' or main.frontBC.BC_type == 'patch') and front_Face==True) or front_face==False)):
      tmp = np.zeros(np.size(fB[:,:,:,:,:,:,0]))
      main.comm.Recv(tmp,source=main.rank_connect[5],tag=main.rank_connect[5])
      uF[:] = np.reshape(tmp,np.shape(fB[:,:,:,:,:,:,0]))

    #============================

    if (main.rank_connect[4] != main.mpi_rank and main.backBC.args[1] == 0 and (main.backBC.BC_type == 'periodic' or main.backBC.BC_type == 'patch') and back_face==True):
      tmp = np.zeros(np.size(fB[:,:,:,:,:,:,0]))
      main.comm.Sendrecv(fB[:,:,:,:,:,:,0].flatten(),dest=main.rank_connect[4],sendtag=main.mpi_rank,\
                         recvbuf=tmp,source=main.rank_connect[4],recvtag=main.rank_connect[4])
      uB[:] = np.reshape(tmp,np.shape(fB[:,:,:,:,:,:,0]))

    #============================





    if (main.rank_connect[5] != main.mpi_rank and ((main.frontBC.args[1] == 0 and (main.frontBC.BC_type == 'periodic' or main.frontBC.BC_type == 'patch') and front_face==True) or front_face==False)):
      main.comm.Send(fF[:,:,:,:,:,:,-1].flatten(),dest=main.rank_connect[5],tag=main.mpi_rank+main.num_processes_global)

    #============================

    if (main.rank_connect[4] != main.mpi_rank and ((main.backBC.args[1] == -1 and (main.backBC.BC_type == 'periodic' or main.backBC.BC_type == 'patch') and back_face==True) or back_face==False)):
      tmp = np.zeros(np.size(fB[:,:,:,:,:,:,0]))
      main.comm.Recv(tmp,source=main.rank_connect[4],tag=main.rank_connect[4]+main.num_processes_global)
      uB[:] = np.reshape(tmp,np.shape(fB[:,:,:,:,:,:,-1]))
    
    #============================

    if (main.rank_connect[5] != main.mpi_rank and main.frontBC.args[1] == -1 and (main.frontBC.BC_type == 'periodic' or main.frontBC.BC_type == 'patch') and front_face==True):
      tmp = np.zeros(np.size(fB[:,:,:,:,:,:,0]))
      main.comm.Sendrecv(fF[:,:,:,:,:,:,-1].flatten(),dest=main.rank_connect[5],sendtag=main.mpi_rank+main.num_processes_global,\
                         recvbuf=tmp,source=main.rank_connect[5],recvtag=main.rank_connect[5]+main.num_processes_global)
      uF[:] = np.reshape(tmp,np.shape(fB[:,:,:,:,:,:,-1]))
    
    #============================





    if (main.BC_rank[5] and main.frontBC.BC_type != 'periodic' and main.frontBC.BC_type != 'patch'): 
      uF[:] = fF[:,:,:,:,:,:,-1]
    elif (main.BC_rank[5] and (main.rightBC.BC_type == 'periodic' or main.leftBC.BC_type == 'patch')):
      uF[:] = main.frontBC.applyBC(fF[:,:,:,:,:,:,-1],uF,main.frontBC.args,main)

    if (main.BC_rank[4] and main.backBC.BC_type != 'periodic'  and main.backBC.BC_type != 'patch'):
      uB[:] = fB[:,:,:,:,:,:,0]
    elif (main.BC_rank[4] and (main.rightBC.BC_type == 'periodic' or main.leftBC.BC_type == 'patch')):
      uB[:] = main.backBC.applyBC(fB[:,:,:,:,:,:,0],uB,main.backBC.args,main)
    
  
  
  return uR,uL,uU,uD,uF,uB





##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
##########################################################################################################################################################################################################





def gatherSolScalar(main,u):
  if (main.mpi_rank == main.starting_rank):
    
    uG = np.zeros((main.quadpoints[0],main.quadpoints[1],main.quadpoints[2],main.quadpoints[3],main.Nel[0],main.Nel[1],main.Nel[2],main.Nel[3]))
    uG[:,:,:,:,0:main.Npx,0:main.Npy,:] = u[:]
    
    for i in main.all_mpi_ranks[1::]:
      loc_rank = i - main.starting_rank
      data = np.zeros(np.shape(u)).flatten()
      main.comm.Recv(data,source=loc_rank + main.starting_rank,tag = loc_rank + main.starting_rank)
      
      xL = (int(loc_rank)%int(main.procx*main.procy))%int(main.procx)*main.Npx
      xR = (int(loc_rank)%int(main.procx*main.procy))%int(main.procx)*main.Npx + main.Npx
      yD = (int(loc_rank)%int(main.procx*main.procy))/int(main.procx)*main.Npy
      yU = (int(loc_rank)%int(main.procx*main.procy))/int(main.procx)*main.Npy + main.Npy
      zB = (int(loc_rank)/int(main.procx*main.procy))*main.Npz
      zF = (int(loc_rank)/int(main.procx*main.procy))*main.Npz + main.Npz
      
      uG[:,:,:,:,xL:xR,yD:yU,zB:zF] = np.reshape(data,np.shape(u))
    
    return uG

  else:
    main.comm.Send(u.flatten(),dest=main.starting_rank,tag=main.mpi_rank)






def gatherSolSlab(main,eqns,var):
  if (main.mpi_rank == main.starting_rank):
    
    uG = np.zeros((var.nvars,var.quadpoints[0],var.quadpoints[1],var.quadpoints[2],var.quadpoints[3],main.Nel[0],main.Nel[1],main.Nel[2],main.Nel[3]))
    uG[:,:,:,:,:,0:main.Npx,0:main.Npy,:] = var.u[:]
    
    for i in main.all_mpi_ranks[1::]:
      loc_rank = i - main.starting_rank
      data = np.zeros(np.shape(var.u)).flatten()
      main.comm.Recv(data,source=loc_rank + main.starting_rank,tag = loc_rank + main.starting_rank)
      
      xL = (int(loc_rank)%int(main.procx*main.procy))%int(main.procx)*main.Npx
      xR = (int(loc_rank)%int(main.procx*main.procy))%int(main.procx)*main.Npx + main.Npx
      yD = (int(loc_rank)%int(main.procx*main.procy))/int(main.procx)*main.Npy
      yU = (int(loc_rank)%int(main.procx*main.procy))/int(main.procx)*main.Npy + main.Npy
      zB = (int(loc_rank)/int(main.procx*main.procy))*main.Npz
      zF = (int(loc_rank)/int(main.procx*main.procy))*main.Npz + main.Npz
      
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
      yD = (int(loc_rank)%int(main.procx*main.procy))/int(main.procx)*main.Npy
      yU = (int(loc_rank)%int(main.procx*main.procy))/int(main.procx)*main.Npy + main.Npy
      zB = (int(loc_rank)/int(main.procx*main.procy))*main.Npz
      zF = (int(loc_rank)/int(main.procx*main.procy))*main.Npz + main.Npz
      
      aG[:,:,:,:,:,xL:xR,yD:yU,zB:zF] = np.reshape(data,(main.nvars,main.order[0],main.order[1],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    
    return aG
  
  else:
    main.comm.Send(a.flatten(),dest=main.starting_rank,tag=main.mpi_rank)






def regionConnector(regionManager):
    
    region = regionManager.region[0]

    #=============================================================================================================================
    
    if (region.leftBC.BC_type == 'patch' and region.BC_rank[0]):
      
      region_connect         = region.leftBC.args[0]

      sr                     = regionManager.starting_rank[region_connect]
      px                     = regionManager.procx[region_connect]
      py                     = regionManager.procy[region_connect]
      pz                     = regionManager.procz[region_connect]

      rowx                   = ((int(region.mpi_rank) - int(region.starting_rank)) % int(region.procx*region.procy)) % int(region.procx)
      rowy                   = ((int(region.mpi_rank) - int(region.starting_rank)) % int(region.procx*region.procy)) / int(region.procx)
      rowz                   =  (int(region.mpi_rank) - int(region.starting_rank)) / int(region.procx*region.procy)
      
      region.rank_connect[0] = sr + px*((rowz*py) + rowy + 1) - 1

    #=============================================================================================================================

    if (region.rightBC.BC_type == 'patch' and region.BC_rank[1]):
      
      region_connect         = region.rightBC.args[0]

      sr                     = regionManager.starting_rank[region_connect]
      px                     = regionManager.procx[region_connect]
      py                     = regionManager.procy[region_connect]
      pz                     = regionManager.procz[region_connect]

      rowx                   = ((int(region.mpi_rank) - int(region.starting_rank)) % int(region.procx*region.procy)) % int(region.procx)
      rowy                   = ((int(region.mpi_rank) - int(region.starting_rank)) % int(region.procx*region.procy)) / int(region.procx)
      rowz                   =  (int(region.mpi_rank) - int(region.starting_rank)) / int(region.procx*region.procy)
      
      region.rank_connect[1] = sr + px*((rowz*py) + rowy)

    #=============================================================================================================================
    
    if (region.bottomBC.BC_type == 'patch' and region.BC_rank[2]):
      
      region_connect         = region.bottomBC.args[0]

      sr                     = regionManager.starting_rank[region_connect]
      px                     = regionManager.procx[region_connect]
      py                     = regionManager.procy[region_connect]
      pz                     = regionManager.procz[region_connect]

      rowx                   = ((int(region.mpi_rank) - int(region.starting_rank)) % int(region.procx*region.procy)) % int(region.procx)
      rowy                   = ((int(region.mpi_rank) - int(region.starting_rank)) % int(region.procx*region.procy)) / int(region.procx)
      rowz                   =  (int(region.mpi_rank) - int(region.starting_rank)) / int(region.procx*region.procy)
      
      region.rank_connect[2] = sr + (rowz*(px*py) + rowx) + (px*py) - px
    
    #=============================================================================================================================
    
    if (region.topBC.BC_type == 'patch' and region.BC_rank[3]):
      
      region_connect         = region.topBC.args[0]

      sr                     = regionManager.starting_rank[region_connect]
      px                     = regionManager.procx[region_connect]
      py                     = regionManager.procy[region_connect]
      pz                     = regionManager.procz[region_connect]

      rowx                   = ((int(region.mpi_rank) - int(region.starting_rank)) % int(region.procx*region.procy)) % int(region.procx)
      rowy                   = ((int(region.mpi_rank) - int(region.starting_rank)) % int(region.procx*region.procy)) / int(region.procx)
      rowz                   =  (int(region.mpi_rank) - int(region.starting_rank)) / int(region.procx*region.procy)
      
      region.rank_connect[3] = sr + (rowz*(px*py) + rowx)
    
    #=============================================================================================================================
    
    if (region.backBC.BC_type == 'patch' and region.BC_rank[4]):
      
      region_connect         = region.backBC.args[0]

      sr                     = regionManager.starting_rank[region_connect]
      px                     = regionManager.procx[region_connect]
      py                     = regionManager.procy[region_connect]
      pz                     = regionManager.procz[region_connect]

      rowx                   = ((int(region.mpi_rank) - int(region.starting_rank)) % int(region.procx*region.procy)) % int(region.procx)
      rowy                   = ((int(region.mpi_rank) - int(region.starting_rank)) % int(region.procx*region.procy)) / int(region.procx)
      rowz                   =  (int(region.mpi_rank) - int(region.starting_rank)) / int(region.procx*region.procy)
      
      region.rank_connect[4] = sr + (rowy*px) + rowx + (pz-1)*px*py
    
    #=============================================================================================================================
    
    if (region.frontBC.BC_type == 'patch' and region.BC_rank[5]):
      
      region_connect         = region.frontBC.args[0]

      sr                     = regionManager.starting_rank[region_connect]
      px                     = regionManager.procx[region_connect]
      py                     = regionManager.procy[region_connect]
      pz                     = regionManager.procz[region_connect]

      rowx                   = ((int(region.mpi_rank) - int(region.starting_rank)) % int(region.procx*region.procy)) % int(region.procx)
      rowy                   = ((int(region.mpi_rank) - int(region.starting_rank)) % int(region.procx*region.procy)) / int(region.procx)
      rowz                   =  (int(region.mpi_rank) - int(region.starting_rank)) / int(region.procx*region.procy)
      
      region.rank_connect[5] = sr + (rowy*px) + rowx
    
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
  
  if (((mpi_rank-starting_rank)%(procx*procy))/procx==0):                   #bottom face
    rank_connect[2] = mpi_rank + procx*procy - procx
    BC_rank[2] = True

  if (((mpi_rank-starting_rank)%(procx*procy))/procx==procy-1):             #top face
    rank_connect[3] = mpi_rank - procx*procy + procx
    BC_rank[3] = True

  if ((mpi_rank-starting_rank)/(procx*procy)==0):                           #back face
    rank_connect[4] = mpi_rank + procx*procy*procz - procx*procy
    BC_rank[4] = True
  
  if ((mpi_rank-starting_rank)/(procx*procy)==procz-1):                     #front face
    rank_connect[5] = mpi_rank - procx*procy*procz + procx*procy
    BC_rank[5] = True

  return rank_connect,BC_rank
