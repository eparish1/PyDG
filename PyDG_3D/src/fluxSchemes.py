import numpy as np
from MPI_functions import *
from tensor_products import *
def centralFluxGeneral(fR,fL,fU,fD,fF,fB,fR_edge,fL_edge,fU_edge,fD_edge,fF_edge,fB_edge):
  fRS = np.zeros(np.shape(fR))
  fLS = np.zeros(np.shape(fL))
  fUS = np.zeros(np.shape(fU))
  fDS = np.zeros(np.shape(fD))
  fFS = np.zeros(np.shape(fF))
  fBS = np.zeros(np.shape(fB))

  fRS[:,:,:,0:-1,:,:] = 0.5*(fR[:,:,:,0:-1,:,:] + fL[:,:,:,1::,:,:])
  fRS[:,:,:,  -1,:,:] = 0.5*(fR[:,:,:,  -1,:,:] + fR_edge)
  fLS[:,:,:,1:: ,:,:] = fRS[:,:,:,0:-1,:,:]
  fLS[:,:,:,0   ,:,:] = 0.5*(fL[:,:,:,0,:,:]    + fL_edge)
  fUS[:,:,:,:,0:-1,:] = 0.5*(fU[:,:,:,:,0:-1,:] + fD[:,:,:,:,1::,:])
  fUS[:,:,:,:,  -1,:] = 0.5*(fU[:,:,:,:,  -1,:] + fU_edge)
  fDS[:,:,:,:,1:: ,:] = fUS[:,:,:,:,0:-1,:]
  fDS[:,:,:,:,0   ,:] = 0.5*(fD[:,:,:,:,   0,:] + fD_edge)
  fFS[:,:,:,:,:,0:-1] = 0.5*(fF[:,:,:,:,:,0:-1] + fB[:,:,:,:,:,1::])
  fFS[:,:,:,:,:,  -1] = 0.5*(fF[:,:,:,:,:,  -1] + fF_edge)
  fBS[:,:,:,:,:,1:: ] = fFS[:,:,:,:,:,0:-1]
  fBS[:,:,:,:,:,0   ] = 0.5*(fB[:,:,:,:,:,   0] + fB_edge)
  return fRS,fLS,fUS,fDS,fFS,fBS

def inviscidFlux(main,eqns,fluxVar,var,args=None):
  nx = np.array([1,0,0])
  ny = np.array([0,1,0])
  nz = np.array([0,0,1])
  fluxVar.fRS[:,:,:,0:-1,:,:] = eqns.inviscidFlux(var.uR[:,:,:,0:-1,:,:],var.uL[:,:,:,1::,:,:],nx)
  fluxVar.fRS[:,:,:,  -1,:,:] = eqns.inviscidFlux(var.uR[:,:,:,  -1,:,:],var.uR_edge,nx)
  fluxVar.fLS[:,:,:,1:: ,:,:] = fluxVar.fRS[:,:,:,0:-1,:,:]
  fluxVar.fLS[:,:,:,0   ,:,:] = eqns.inviscidFlux(var.uL_edge,var.uL[:,:,:,0,:,:],nx)
  fluxVar.fUS[:,:,:,:,0:-1,:] = eqns.inviscidFlux(var.uU[:,:,:,:,0:-1,:],var.uD[:,:,:,:,1::,:],ny )
  fluxVar.fUS[:,:,:,:,  -1,:] = eqns.inviscidFlux(var.uU[:,:,:,:,  -1,:],var.uU_edge,ny)
  fluxVar.fDS[:,:,:,:,1:: ,:] = fluxVar.fUS[:,:,:,:,0:-1,:] 
  fluxVar.fDS[:,:,:,:,0   ,:] = eqns.inviscidFlux(var.uD_edge,var.uD[:,:,:,:,0,:],ny)
  #print(np.amax(fluxVar.fDS[1,:,:,:,0,:]),np.amax(np.abs(var.uD_edge[1])), np.amax(np.abs(var.uD[1,:,:,:,0,:]/var.uD[0,:,:,:,0,:] ) )  )
  fluxVar.fFS[:,:,:,:,:,0:-1] = eqns.inviscidFlux(var.uF[:,:,:,:,:,0:-1],var.uB[:,:,:,:,:,1::],nz )
  fluxVar.fFS[:,:,:,:,:,  -1] = eqns.inviscidFlux(var.uF[:,:,:,:,:,  -1],var.uF_edge,nz)
  fluxVar.fBS[:,:,:,:,:,1:: ] = fluxVar.fFS[:,:,:,:,:,0:-1] 
  fluxVar.fBS[:,:,:,:,:,0   ] = eqns.inviscidFlux(var.uB_edge,var.uB[:,:,:,:,:,0],nz)

def inviscidFluxTwoArg(main,eqns,fluxVar,var,args):
  up = args[0]
  upR,upL,upU,upD,upF,upB = reconstructEdgesGeneral(up,main)
  upR_edge,upL_edge,upU_edge,upD_edge,upF_edge,upB_edge = sendEdgesGeneralSlab(upL,upR,upD,upU,upB,upF,main)


  nx = np.array([1,0,0])
  ny = np.array([0,1,0])
  nz = np.array([0,0,1])
  fluxVar.fRS[:,:,:,0:-1,:,:] = eqns.inviscidFlux(var.uR[:,:,:,0:-1,:,:],var.uL[:,:,:,1::,:,:],upR[:,:,:,0:-1,:,:],upL[:,:,:,1::,:,:],nx)
  fluxVar.fRS[:,:,:,  -1,:,:] = eqns.inviscidFlux(var.uR[:,:,:,  -1,:,:],var.uR_edge,upR[:,:,:,  -1,:,:],upR_edge,nx)
  fluxVar.fLS[:,:,:,1:: ,:,:] = fluxVar.fRS[:,:,:,0:-1,:,:]
  fluxVar.fLS[:,:,:,0   ,:,:] = eqns.inviscidFlux(var.uL_edge,var.uL[:,:,:,0,:,:],upL_edge,upL[:,:,:,0,:,:],nx)
  fluxVar.fUS[:,:,:,:,0:-1,:] = eqns.inviscidFlux(var.uU[:,:,:,:,0:-1,:],var.uD[:,:,:,:,1::,:],upU[:,:,:,:,0:-1,:],upD[:,:,:,:,1::,:],ny )
  fluxVar.fUS[:,:,:,:,  -1,:] = eqns.inviscidFlux(var.uU[:,:,:,:,  -1,:],var.uU_edge,upU[:,:,:,:,  -1,:],upU_edge,ny)
  fluxVar.fDS[:,:,:,:,1:: ,:] = fluxVar.fUS[:,:,:,:,0:-1,:] 
  fluxVar.fDS[:,:,:,:,0   ,:] = eqns.inviscidFlux(var.uD_edge,var.uD[:,:,:,:,0,:],upD_edge,upD[:,:,:,:,0,:],ny)
  #print(np.amax(fluxVar.fDS[1,:,:,:,0,:]),np.amax(np.abs(var.uD_edge[1])), np.amax(np.abs(var.uD[1,:,:,:,0,:]/var.uD[0,:,:,:,0,:] ) )  )
  fluxVar.fFS[:,:,:,:,:,0:-1] = eqns.inviscidFlux(var.uF[:,:,:,:,:,0:-1],var.uB[:,:,:,:,:,1::],upF[:,:,:,:,:,0:-1],upB[:,:,:,:,:,1::],nz )
  fluxVar.fFS[:,:,:,:,:,  -1] = eqns.inviscidFlux(var.uF[:,:,:,:,:,  -1],var.uF_edge,upF[:,:,:,:,:,  -1],upF_edge,nz)
  fluxVar.fBS[:,:,:,:,:,1:: ] = fluxVar.fFS[:,:,:,:,:,0:-1] 
  fluxVar.fBS[:,:,:,:,:,0   ] = eqns.inviscidFlux(var.uB_edge,var.uB[:,:,:,:,:,0],upB_edge,upB[:,:,:,:,:,0],nz)

def inviscidFluxGen(main,eqns,fluxVar,var,args):
  nx = np.array([1,0,0])
  ny = np.array([0,1,0])
  nz = np.array([0,0,1])
  nargs = np.shape(args)[0]
  argsR,argsL,argsU,argsD,argsF,argsB = [],[],[],[],[],[]
  argsR_edge,argsL_edge,argsU_edge,argsD_edge,argsF_edge,argsB_edge = [],[],[],[],[],[]
  for i in range(0,nargs):
    tmpR,tmpL,tmpU,tmpD,tmpF,tmpB = reconstructEdgesGeneral(args[i],main)
    tmpR_edge,tmpL_edge,tmpU_edge,tmpD_edge,tmpF_edge,tmpB_edge = sendEdgesGeneralSlab(tmpL,tmpR,tmpD,tmpU,tmpB,tmpF,main)
    argsR.append(tmpR)
    argsL.append(tmpL)
    argsU.append(tmpU)
    argsD.append(tmpD)
    argsF.append(tmpF)
    argsB.append(tmpB)
    argsR_edge.append(tmpR_edge)
    argsL_edge.append(tmpL_edge)
    argsU_edge.append(tmpU_edge)
    argsD_edge.append(tmpD_edge)
    argsF_edge.append(tmpF_edge)
    argsB_edge.append(tmpB_edge)

  ## Get left and right fluxes
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsR[i][:,:,:,0:-1,:,:])
    fluxArgs.append(argsL[i][:,:,:,1::,:,: ])
  fluxVar.fRS[:,:,:,0:-1,:,:] = eqns.inviscidFlux(var.uR[:,:,:,0:-1,:,:],var.uL[:,:,:,1::,:,:],nx,fluxArgs)
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsR[i][:,:,:,-1,:,:])
    fluxArgs.append(argsR_edge[i])
  fluxVar.fRS[:,:,:,  -1,:,:] = eqns.inviscidFlux(var.uR[:,:,:,  -1,:,:],var.uR_edge,nx,fluxArgs)
  fluxVar.fLS[:,:,:,1:: ,:,:] = fluxVar.fRS[:,:,:,0:-1,:,:] ##same as the right flux but shifted one
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsL_edge[i])
    fluxArgs.append(argsL[i][:,:,:,0,:,:])
  fluxVar.fLS[:,:,:,0   ,:,:] = eqns.inviscidFlux(var.uL_edge,var.uL[:,:,:,0,:,:],nx,fluxArgs)
  ## Get the up and down fluxes
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsU[i][:,:,:,:,0:-1,:])
    fluxArgs.append(argsD[i][:,:,:,:,1::,:])
  fluxVar.fUS[:,:,:,:,0:-1,:] = eqns.inviscidFlux(var.uU[:,:,:,:,0:-1,:],var.uD[:,:,:,:,1::,:],ny,fluxArgs)
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsU[i][:,:,:,:,  -1,:])
    fluxArgs.append(argsU_edge[i])
  fluxVar.fUS[:,:,:,:,  -1,:] = eqns.inviscidFlux(var.uU[:,:,:,:,  -1,:],var.uU_edge,ny,fluxArgs)
  fluxVar.fDS[:,:,:,:,1:: ,:] = fluxVar.fUS[:,:,:,:,0:-1,:] 
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsD_edge[i])
    fluxArgs.append(argsD[i][:,:,:,:,0,:])
  fluxVar.fDS[:,:,:,:,0   ,:] = eqns.inviscidFlux(var.uD_edge,var.uD[:,:,:,:,0,:],ny,fluxArgs)
  ## Get the front and back fluxes
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsF[i][:,:,:,:,:,0:-1])
    fluxArgs.append(argsB[i][:,:,:,:,:,1::])
  fluxVar.fFS[:,:,:,:,:,0:-1] = eqns.inviscidFlux(var.uF[:,:,:,:,:,0:-1],var.uB[:,:,:,:,:,1::],nz,fluxArgs)
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsF[i][:,:,:,:,:,-1])
    fluxArgs.append(argsF_edge[i])
  fluxVar.fFS[:,:,:,:,:,  -1] = eqns.inviscidFlux(var.uF[:,:,:,:,:,  -1],var.uF_edge,nz,fluxArgs)
  fluxVar.fBS[:,:,:,:,:,1:: ] = fluxVar.fFS[:,:,:,:,:,0:-1] 
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsB_edge[i])
    fluxArgs.append(argsB[i][:,:,:,:,:,0])
  fluxVar.fBS[:,:,:,:,:,0   ] = eqns.inviscidFlux(var.uB_edge,var.uB[:,:,:,:,:,0],nz,fluxArgs)

