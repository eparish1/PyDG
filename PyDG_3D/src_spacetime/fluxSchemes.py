import numpy as np
from MPI_functions import *
from tensor_products import *
from navier_stokes_reacting import *
def centralFluxGeneral(fR,fL,fU,fD,fF,fB,fR_edge,fL_edge,fU_edge,fD_edge,fF_edge,fB_edge):
  fRS = np.zeros(np.shape(fR))
  fLS = np.zeros(np.shape(fL))
  fUS = np.zeros(np.shape(fU))
  fDS = np.zeros(np.shape(fD))
  fFS = np.zeros(np.shape(fF))
  fBS = np.zeros(np.shape(fB))
  fRS[:,:,:,:,0:-1,:,:] = 0.5*(fR[:,:,:,:,0:-1,:,:] + fL[:,:,:,:,1::,:,:])
  fRS[:,:,:,:,  -1,:,:] = 0.5*(fR[:,:,:,:,  -1,:,:] + fR_edge)
  fLS[:,:,:,:,1:: ,:,:] = fRS[:,:,:,:,0:-1,:,:]
  fLS[:,:,:,:,0   ,:,:] = 0.5*(fL[:,:,:,:,0,:,:]    + fL_edge)
  fUS[:,:,:,:,:,0:-1,:] = 0.5*(fU[:,:,:,:,:,0:-1,:] + fD[:,:,:,:,:,1::,:])
  fUS[:,:,:,:,:,  -1,:] = 0.5*(fU[:,:,:,:,:,  -1,:] + fU_edge)
  fDS[:,:,:,:,:,1:: ,:] = fUS[:,:,:,:,:,0:-1,:]
  fDS[:,:,:,:,:,0   ,:] = 0.5*(fD[:,:,:,:,:,   0,:] + fD_edge)
  fFS[:,:,:,:,:,:,0:-1] = 0.5*(fF[:,:,:,:,:,:,0:-1] + fB[:,:,:,:,:,:,1::])
  fFS[:,:,:,:,:,:,  -1] = 0.5*(fF[:,:,:,:,:,:,  -1] + fF_edge)
  fBS[:,:,:,:,:,:,1:: ] = fFS[:,:,:,:,:,:,0:-1]
  fBS[:,:,:,:,:,:,0   ] = 0.5*(fB[:,:,:,:,:,:,   0] + fB_edge)
  return fRS,fLS,fUS,fDS,fFS,fBS

def centralFluxGeneral2(fRLS,fUDS,fFBS,fR,fL,fU,fD,fF,fB,fR_edge,fL_edge,fU_edge,fD_edge,fF_edge,fB_edge):
  fRLS[:,:,:,:,1:-1,:,:] = 0.5*(fR[:,:,:,:,0:-1,:,:] + fL[:,:,:,:,1::,:,:])
  fRLS[:,:,:,:,  -1,:,:] = 0.5*(fR[:,:,:,:,  -1,:,:] + fR_edge)
  fRLS[:,:,:,:,0   ,:,:] =  0.5*(fL[:,:,:,:,0,:,:]    + fL_edge)
  fUDS[:,:,:,:,:,1:-1,:] = 0.5*(fU[:,:,:,:,:,0:-1,:] + fD[:,:,:,:,:,1::,:])
  fUDS[:,:,:,:,:,  -1,:] = 0.5*(fU[:,:,:,:,:,  -1,:] + fU_edge)
  fUDS[:,:,:,:,:,0   ,:] = 0.5*(fD[:,:,:,:,:,   0,:] + fD_edge)
  fFBS[:,:,:,:,:,:,1:-1] = 0.5*(fF[:,:,:,:,:,:,0:-1] + fB[:,:,:,:,:,:,1::])
  fFBS[:,:,:,:,:,:,  -1] = 0.5*(fF[:,:,:,:,:,:,  -1] + fF_edge)
  fFBS[:,:,:,:,:,:,0   ] = 0.5*(fB[:,:,:,:,:,:,   0] + fB_edge)

def generalFlux(main,eqns,fluxVar,var,fluxFunction,args=None):
  fluxVar.fRLS[:,:,:,:,1:-1,:,:] = fluxFunction(main,var.uR[:,:,:,:,0:-1,:,:],var.uL[:,:,:,:,1::,:,:],main.normals[0][:,None,None,None,0:-1,:,:,None])
  fluxVar.fRLS[:,:,:,:,  -1,:,:] = fluxFunction(main,var.uR[:,:,:,:,  -1,:,:],var.uR_edge,main.normals[0][:,None,None,None,-1,:,:,None])
  fluxVar.fRLS[:,:,:,:,0   ,:,:] = fluxFunction(main,var.uL_edge,var.uL[:,:,:,:,0,:,:],-main.normals[1][:,None,None,None,0,:,:,None])

  fluxVar.fUDS[:,:,:,:,:,1:-1,:] = fluxFunction(main,var.uU[:,:,:,:,:,0:-1,:],var.uD[:,:,:,:,:,1::,:],main.normals[2][:,None,None,None,:,0:-1,:,None])
  fluxVar.fUDS[:,:,:,:,:,  -1,:] = fluxFunction(main,var.uU[:,:,:,:,:,  -1,:],var.uU_edge,main.normals[2][:,None,None,None,:,-1,:,None])
  fluxVar.fUDS[:,:,:,:,:,0   ,:] = fluxFunction(main,var.uD_edge,var.uD[:,:,:,:,:,0,:],-main.normals[3][:,None,None,None,:,0,:,None])

  fluxVar.fFBS[:,:,:,:,:,:,1:-1] = fluxFunction(main,var.uF[:,:,:,:,:,:,0:-1],var.uB[:,:,:,:,:,:,1::],main.normals[4][:,None,None,None,:,:,0:-1,None])
  fluxVar.fFBS[:,:,:,:,:,:,  -1] = fluxFunction(main,var.uF[:,:,:,:,:,:,  -1],var.uF_edge,main.normals[4][:,None,None,None,:,:,-1,None])
  fluxVar.fFBS[:,:,:,:,:,:,0   ] = fluxFunction(main,var.uB_edge,var.uB[:,:,:,:,:,:,0],-main.normals[5][:,None,None,None,:,:,0,None])

def inviscidFlux(main,eqns,fluxVar,var,args=None):
  #nx = np.array([1,0,0])
  #ny = np.array([0,1,0])
  #nz = np.array([0,0,1])
  fluxVar.fRLS[:,:,:,:,1:-1,:,:] = eqns.inviscidFlux(main,var.uR[:,:,:,:,0:-1,:,:],var.uL[:,:,:,:,1::,:,:],main.normals[0][:,None,None,None,0:-1,:,:,None])
  fluxVar.fRLS[:,:,:,:,  -1,:,:] = eqns.inviscidFlux(main,var.uR[:,:,:,:,  -1,:,:],var.uR_edge,main.normals[0][:,None,None,None,-1,:,:,None])
  fluxVar.fRLS[:,:,:,:,0   ,:,:] = eqns.inviscidFlux(main,var.uL_edge,var.uL[:,:,:,:,0,:,:],-main.normals[1][:,None,None,None,0,:,:,None])

  fluxVar.fUDS[:,:,:,:,:,1:-1,:] = eqns.inviscidFlux(main,var.uU[:,:,:,:,:,0:-1,:],var.uD[:,:,:,:,:,1::,:],main.normals[2][:,None,None,None,:,0:-1,:,None])
  fluxVar.fUDS[:,:,:,:,:,  -1,:] = eqns.inviscidFlux(main,var.uU[:,:,:,:,:,  -1,:],var.uU_edge,main.normals[2][:,None,None,None,:,-1,:,None])
  fluxVar.fUDS[:,:,:,:,:,0   ,:] = eqns.inviscidFlux(main,var.uD_edge,var.uD[:,:,:,:,:,0,:],-main.normals[3][:,None,None,None,:,0,:,None])

  fluxVar.fFBS[:,:,:,:,:,:,1:-1] = eqns.inviscidFlux(main,var.uF[:,:,:,:,:,:,0:-1],var.uB[:,:,:,:,:,:,1::],main.normals[4][:,None,None,None,:,:,0:-1,None])
  fluxVar.fFBS[:,:,:,:,:,:,  -1] = eqns.inviscidFlux(main,var.uF[:,:,:,:,:,:,  -1],var.uF_edge,main.normals[4][:,None,None,None,:,:,-1,None])
  fluxVar.fFBS[:,:,:,:,:,:,0   ] = eqns.inviscidFlux(main,var.uB_edge,var.uB[:,:,:,:,:,:,0],-main.normals[5][:,None,None,None,:,:,0,None])


def inviscidFlux_DOUBLEFLUX(main,eqns,fluxVar,var,args=None):
  nx = np.array([1,0,0])
  ny = np.array([0,1,0])
  nz = np.array([0,0,1])
  fluxVar.fRS[:,:,:,:,0:-1,:,:] = HLLCFlux_reacting_doublefluxR(main,var.uR[:,:,:,:,0:-1,:,:],var.uL[:,:,:,:,1::,:,:],main.a.pR[:,:,:,0:-1,:,:],main.a.pL[:,:,:,1::,:,:],main.a.rh0[0,:,:,:,0:-1,:,:],main.a.gamma_star[0,:,:,:,0:-1,:,:],nx)
  fluxVar.fRS[:,:,:,:,  -1,:,:] = HLLCFlux_reacting_doublefluxR(main,var.uR[:,:,:,:,  -1,:,:],var.uR_edge,main.a.pR[:,:,:,-1,:,:],main.a.pR_edge,main.a.rh0[0,:,:,:,-1,:,:],main.a.gamma_star[0,:,:,:,-1,:,:],nx)
  fluxVar.fLS[:,:,:,:,1:: ,:,:] = HLLCFlux_reacting_doublefluxL(main,var.uR[:,:,:,:,0:-1,:,:],var.uL[:,:,:,:,1::,:,:],main.a.pR[:,:,:,0:-1,:,:],main.a.pL[:,:,:,1::,:,:],main.a.rh0[0,:,:,:,1::,:,:],main.a.gamma_star[0,:,:,:,1::,:,:],nx)
  fluxVar.fLS[:,:,:,:,0   ,:,:] = HLLCFlux_reacting_doublefluxL(main,var.uL_edge,var.uL[:,:,:,:,0,:,:],main.a.pL_edge,main.a.pL[:,:,:,0,:,:],main.a.rh0[0,:,:,:,0,:,:],main.a.gamma_star[0,:,:,:,0,:,:],nx)

  fluxVar.fUS[:,:,:,:,:,0:-1,:] = eqns.inviscidFlux(main,var.uU[:,:,:,:,:,0:-1,:],var.uD[:,:,:,:,:,1::,:],main.a.pU[:,:,:,:,0:-1,:],main.a.pD[:,:,:,:,1::,:],main.a.rh0[:,0,:,:,:,0:-1,:],main.a.gamma_star[:,0,:,:,:,0:-1,:],ny )
  fluxVar.fUS[:,:,:,:,:,  -1,:] = eqns.inviscidFlux(main,var.uU[:,:,:,:,:,  -1,:],var.uU_edge,main.a.pU[:,:,:,:,  -1,:],main.a.pU_edge,main.a.rh0[:,0,:,:,  -1,:],main.a.gamma_star[:,0,:,:,  -1,:],ny)
  fluxVar.fDS[:,:,:,:,:,1:: ,:] = eqns.inviscidFlux(main,var.uU[:,:,:,:,:,0:-1,:],var.uD[:,:,:,:,:,1::,:],main.a.pU[:,:,:,:,0:-1,:],main.a.pD[:,:,:,:,1::,:],main.a.rh0[:,0,:,:,:,1::,:],main.a.gamma_star[:,0,:,:,:,1::,:],ny )
  fluxVar.fDS[:,:,:,:,:,0   ,:] = eqns.inviscidFlux(main,var.uD_edge,var.uD[:,:,:,:,:,0,:],main.a.pD_edge,main.a.pD[:,:,:,:,0,:],main.a.rh0[:,0,:,:,:,0,:],main.a.gamma_star[:,0,:,:,:,0,:],ny)

  fluxVar.fFS[:,:,:,:,:,:,0:-1] = eqns.inviscidFlux(main,var.uF[:,:,:,:,:,:,0:-1],var.uB[:,:,:,:,:,:,1::],main.a.pF[:,:,:,:,:,0:-1],main.a.pB[:,:,:,:,:,1::],main.a.rh0[:,0,:,:,:,:,0:-1],main.a.gamma_star[:,0,:,:,:,:,0:-1],nz )
  fluxVar.fFS[:,:,:,:,:,:,  -1] = eqns.inviscidFlux(main,var.uF[:,:,:,:,:,:,  -1],var.uF_edge,main.a.pF[:,:,:,:,:,  -1],main.a.pF_edge,main.a.rh0[:,0,:,:,:,:,  -1],main.a.gamma_star[:,0,:,:,:,:,  -1],nz)
  fluxVar.fBS[:,:,:,:,:,:,1:: ] = eqns.inviscidFlux(main,var.uF[:,:,:,:,:,:,0:-1],var.uB[:,:,:,:,:,:,1::],main.a.pF[:,:,:,:,:,0:-1],main.a.pB[:,:,:,:,:,1::],main.a.rh0[:,0,:,:,:,:,1::],main.a.gamma_star[:,0,:,:,:,:,1::],nz )
  fluxVar.fBS[:,:,:,:,:,:,0   ] = eqns.inviscidFlux(main,var.uB_edge,var.uB[:,:,:,:,:,:,0],main.a.pB_edge,main.a.pB[:,:,:,:,:,0],main.a.rh0[:,0,:,:,:,:,0],main.a.gamma_star[:,0,:,:,:,:,0],nz)


def inviscidFlux_DOUBLEFLUX2(main,eqns,fluxVar,var,args=None):
  nx = np.array([1,0,0])
  ny = np.array([0,1,0])
  nz = np.array([0,0,1])
  fluxVar.fRS[:,:,:,:,0:-1,:,:] = HLLCFlux_reacting_doubleflux_minus(main,var.uR[:,:,:,:,0:-1,:,:],var.uL[:,:,:,:,1::,:,:],main.a.pR[:,:,:,0:-1,:,:],main.a.pL[:,:,:,1::,:,:],main.a.gamma_star[0,:,:,:,0:-1,:,:],main.a.gamma_star[0,:,:,:,1::,:,:],nx)
  fluxVar.fRS[:,:,:,:,  -1,:,:] = HLLCFlux_reacting_doubleflux_minus(main,var.uR[:,:,:,:,  -1,:,:],var.uR_edge,main.a.pR[:,:,:,-1,:,:],main.a.pR_edge,main.a.gamma_star[0,:,:,:,-1,:,:],main.a.gamma_star[0,:,:,:,0,:,:],nx)
  fluxVar.fLS[:,:,:,:,1:: ,:,:] = HLLCFlux_reacting_doubleflux_plus(main,var.uR[:,:,:,:,0:-1,:,:],var.uL[:,:,:,:,1::,:,:],main.a.pR[:,:,:,0:-1,:,:],main.a.pL[:,:,:,1::,:,:],main.a.gamma_star[0,:,:,:,0:-1,:,:],main.a.gamma_star[0,:,:,:,1::,:,:],nx)
  fluxVar.fLS[:,:,:,:,0   ,:,:] = HLLCFlux_reacting_doubleflux_plus(main,var.uL_edge,var.uL[:,:,:,:,0,:,:],main.a.pL_edge,main.a.pL[:,:,:,0,:,:],main.a.gamma_star[0,:,:,:,-1,:,:],main.a.gamma_star[0,:,:,:,0,:,:],nx)

  fluxVar.fUS[:,:,:,:,:,0:-1,:] = eqns.inviscidFlux(main,var.uU[:,:,:,:,:,0:-1,:],var.uD[:,:,:,:,:,1::,:],main.a.pU[:,:,:,:,0:-1,:],main.a.pD[:,:,:,:,1::,:],ny )
  fluxVar.fUS[:,:,:,:,:,  -1,:] = eqns.inviscidFlux(main,var.uU[:,:,:,:,:,  -1,:],var.uU_edge,main.a.pU[:,:,:,:,  -1,:],main.a.pU_edge,ny)
  fluxVar.fDS[:,:,:,:,:,1:: ,:] = fluxVar.fUS[:,:,:,:,:,0:-1,:] 
  fluxVar.fDS[:,:,:,:,:,0   ,:] = eqns.inviscidFlux(main,var.uD_edge,var.uD[:,:,:,:,:,0,:],main.a.pD_edge,main.a.pD[:,:,:,:,0,:],ny)

  fluxVar.fFS[:,:,:,:,:,:,0:-1] = eqns.inviscidFlux(main,var.uF[:,:,:,:,:,:,0:-1],var.uB[:,:,:,:,:,:,1::],main.a.pF[:,:,:,:,:,0:-1],main.a.pB[:,:,:,:,:,1::],nz )
  fluxVar.fFS[:,:,:,:,:,:,  -1] = eqns.inviscidFlux(main,var.uF[:,:,:,:,:,:,  -1],var.uF_edge,main.a.pF[:,:,:,:,:,  -1],main.a.pF_edge,nz)
  fluxVar.fBS[:,:,:,:,:,:,1:: ] = fluxVar.fFS[:,:,:,:,:,:,0:-1] 
  fluxVar.fBS[:,:,:,:,:,:,0   ] = eqns.inviscidFlux(main,var.uB_edge,var.uB[:,:,:,:,:,:,0],main.a.pB_edge,main.a.pB[:,:,:,:,:,0],nz)



def generalFluxTwoArg(main,eqns,fluxVar,var,var2,fluxFunction,args):
  fluxVar.fRLS[:,:,:,:,1:-1,:,:] = fluxFunction(main,var.uR[:,:,:,:,0:-1,:,:],var.uL[:,:,:,:,1::,:,:],var2.uR[:,:,:,:,0:-1,:,:],var2.uL[:,:,:,:,1::,:,:],main.normals[0][:,None,None,None,0:-1,:,:,None])
  fluxVar.fRLS[:,:,:,:,  -1,:,:] = fluxFunction(main,var.uR[:,:,:,:,  -1,:,:],var.uR_edge,var2.uR[:,:,:,:,  -1,:,:],var2.uR_edge,main.normals[0][:,None,None,None,-1,:,:,None])
  fluxVar.fRLS[:,:,:,:,0   ,:,:] = fluxFunction(main,var.uL_edge,var.uL[:,:,:,:,0,:,:],var2.uL_edge,var2.uL[:,:,:,:,0,:,:],-main.normals[1][:,None,None,None,0,:,:,None])
  fluxVar.fUDS[:,:,:,:,:,1:-1,:] = fluxFunction(main,var.uU[:,:,:,:,:,0:-1,:],var.uD[:,:,:,:,:,1::,:],var2.uU[:,:,:,:,:,0:-1,:],var2.uD[:,:,:,:,:,1::,:],main.normals[2][:,None,None,None,:,0:-1,:,None])
  fluxVar.fUDS[:,:,:,:,:,  -1,:] = fluxFunction(main,var.uU[:,:,:,:,:,  -1,:],var.uU_edge,var2.uU[:,:,:,:,:,  -1,:],var2.uU_edge,main.normals[2][:,None,None,None,:,-1,:,None])
  fluxVar.fUDS[:,:,:,:,:,0   ,:] = fluxFunction(main,var.uD_edge,var.uD[:,:,:,:,:,0,:],var2.uD_edge,var2.uD[:,:,:,:,:,0,:],-main.normals[3][:,None,None,None,:,0,:,None])
  fluxVar.fFBS[:,:,:,:,:,:,1:-1] = fluxFunction(main,var.uF[:,:,:,:,:,:,0:-1],var.uB[:,:,:,:,:,:,1::],var2.uF[:,:,:,:,:,:,0:-1],var2.uB[:,:,:,:,:,:,1::],main.normals[4][:,None,None,None,:,:,0:-1,None])
  fluxVar.fFBS[:,:,:,:,:,:,  -1] = fluxFunction(main,var.uF[:,:,:,:,:,:,  -1],var.uF_edge,var2.uF[:,:,:,:,:,:,  -1],var2.uF_edge,main.normals[4][:,None,None,None,:,:,-1,None])
  fluxVar.fFBS[:,:,:,:,:,:,0   ] = fluxFunction(main,var.uB_edge,var.uB[:,:,:,:,:,:,0],var2.uB_edge,var2.uB[:,:,:,:,:,:,0],-main.normals[5][:,None,None,None,:,:,0,None])


def generalFluxGen2(region,eqns,fluxVar,var,fluxFunction,args):
  nargs = np.shape(args)[0]
  argsR,argsL,argsU,argsD,argsF,argsB = [],[],[],[],[],[]
  argsR_edge,argsL_edge,argsU_edge,argsD_edge,argsF_edge,argsB_edge = [],[],[],[],[],[]
  for i in range(0,nargs):
    #main.a.uR[:],main.a.uL[:],main.a.uU[:],main.a.uD[:],main.a.uF[:],main.a.uB[:] = main.basis.reconstructEdgesGeneral(main.a.a,main)   
    #tmpR,tmpL,tmpU,tmpD,tmpF,tmpB = main.basis.reconstructEdgesGeneral(args[i],main)
    #tmpR_edge,tmpL_edge,tmpU_edge,tmpD_edge,tmpF_edge,tmpB_edge = sendEdgesGeneralSlab(tmpL,tmpR,tmpD,tmpU,tmpB,tmpF,main)
    argsR.append(args[i].uR)
    argsL.append(args[i].uL)
    argsU.append(args[i].uU)
    argsD.append(args[i].uD)
    argsF.append(args[i].uF)
    argsB.append(args[i].uB)
    argsR_edge.append(args[i].uR_edge)
    argsL_edge.append(args[i].uL_edge)
    argsU_edge.append(args[i].uU_edge)
    argsD_edge.append(args[i].uD_edge)
    argsF_edge.append(args[i].uF_edge)
    argsB_edge.append(args[i].uB_edge)

  ## Get left and right fluxes
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsR[i][:,:,:,:,0:-1,:,:])
    fluxArgs.append(argsL[i][:,:,:,:,1::,:,: ])

  tmpuR = np.append(var.uL_edge[:,:,:,:,None],var.uR,axis=4)
  tmpuL = np.append(var.uL,var.uR_edge[:,:,:,:,None],axis=4)
  tmp_normals = np.append(-region.normals[1][:,0:1,:,:], region.normals[0][:,:,:,:],axis=1)


  fluxFunction(eqns,fluxVar.fRLS[:,:,:,:,1:-1,:,:],region,var.uR[:,:,:,:,0:-1,:,:],var.uL[:,:,:,:,1::,:,:],region.normals[0][:,None,None,None,0:-1,:,:,None],fluxArgs)
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsR[i][:,:,:,:,-1,:,:])
    fluxArgs.append(argsR_edge[i])

  fluxFunction(eqns,fluxVar.fRLS[:,:,:,:,  -1,:,:],region,var.uR[:,:,:,:,  -1,:,:],var.uR_edge,region.normals[0][:,None,None,None,-1,:,:,None],fluxArgs)
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsL_edge[i])
    fluxArgs.append(argsL[i][:,:,:,:,0,:,:])
  fluxFunction(eqns,fluxVar.fRLS[:,:,:,:,0   ,:,:],region,var.uL_edge,var.uL[:,:,:,:,0,:,:],-region.normals[1][:,None,None,None,0,:,:,None],fluxArgs)

  ## Get the up and down fluxes
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsU[i][:,:,:,:,:,0:-1,:])
    fluxArgs.append(argsD[i][:,:,:,:,:,1::,:])
  fluxFunction(eqns,fluxVar.fUDS[:,:,:,:,:,1:-1,:],region,var.uU[:,:,:,:,:,0:-1,:],var.uD[:,:,:,:,:,1::,:],region.normals[2][:,None,None,None,:,0:-1,:,None],fluxArgs)

  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsU[i][:,:,:,:,:,  -1,:])
    fluxArgs.append(argsU_edge[i])
  fluxFunction(eqns,fluxVar.fUDS[:,:,:,:,:,  -1,:],region,var.uU[:,:,:,:,:,  -1,:],var.uU_edge,region.normals[2][:,None,None,None,:,-1,:,None],fluxArgs)

  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsD_edge[i])
    fluxArgs.append(argsD[i][:,:,:,:,:,0,:])
  fluxFunction(eqns,fluxVar.fUDS[:,:,:,:,:,0   ,:],region,var.uD_edge,var.uD[:,:,:,:,:,0,:],-region.normals[3][:,None,None,None,:,0,:,None],fluxArgs)

  ## Get the front and back fluxes
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsF[i][:,:,:,:,:,:,0:-1])
    fluxArgs.append(argsB[i][:,:,:,:,:,:,1::])
  fluxFunction(eqns,fluxVar.fFBS[:,:,:,:,:,:,1:-1],region,var.uF[:,:,:,:,:,:,0:-1],var.uB[:,:,:,:,:,:,1::],region.normals[4][:,None,None,None,:,:,0:-1,None],fluxArgs)

  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsF[i][:,:,:,:,:,:,-1])
    fluxArgs.append(argsF_edge[i])
  fluxFunction(eqns,fluxVar.fFBS[:,:,:,:,:,:,  -1],region,var.uF[:,:,:,:,:,:,  -1],var.uF_edge,region.normals[4][:,None,None,None,:,:,-1,None],fluxArgs)

  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsB_edge[i])
    fluxArgs.append(argsB[i][:,:,:,:,:,:,0])
  fluxFunction(eqns,fluxVar.fFBS[:,:,:,:,:,:,0   ],region,var.uB_edge,var.uB[:,:,:,:,:,:,0],-region.normals[5][:,None,None,None,:,:,0,None],fluxArgs)



### Flux function that computes the fluxes at a given set of element indices
def generalFluxGen3(region,eqns,fluxVar,var,fluxFunction,args):
  nargs = np.shape(args)[0]
  argsR,argsL,argsU,argsD,argsF,argsB = [],[],[],[],[],[]
  argsR_edge,argsL_edge,argsU_edge,argsD_edge,argsF_edge,argsB_edge = [],[],[],[],[],[]
  for i in range(0,nargs):
    #main.a.uR[:],main.a.uL[:],main.a.uU[:],main.a.uD[:],main.a.uF[:],main.a.uB[:] = main.basis.reconstructEdgesGeneral(main.a.a,main)   
    #tmpR,tmpL,tmpU,tmpD,tmpF,tmpB = main.basis.reconstructEdgesGeneral(args[i],main)
    #tmpR_edge,tmpL_edge,tmpU_edge,tmpD_edge,tmpF_edge,tmpB_edge = sendEdgesGeneralSlab(tmpL,tmpR,tmpD,tmpU,tmpB,tmpF,main)
    argsR.append(args[i].uR)
    argsL.append(args[i].uL)
    argsU.append(args[i].uU)
    argsD.append(args[i].uD)
    argsF.append(args[i].uF)
    argsB.append(args[i].uB)
    argsR_edge.append(args[i].uR_edge)
    argsL_edge.append(args[i].uL_edge)
    argsU_edge.append(args[i].uU_edge)
    argsD_edge.append(args[i].uD_edge)
    argsF_edge.append(args[i].uF_edge)
    argsB_edge.append(args[i].uB_edge)

  ## Get left and right fluxes
  fluxArgs = []
#  for i in range(0,nargs):
#    fluxArgs.append(argsR[i][:,:,:,:,0:-1,:,:])
#    fluxArgs.append(argsL[i][:,:,:,:,1::,:,: ])

  ## Generate indices for the faces
  ## 
  indx_x = np.append(region.indx_x,region.indx_x[-1] + 1) ## append this since we have one extra value in the left-right direction
  tmpuR = np.append(var.uL_edge[:,:,:,:,None],var.uR,axis=4)
  tmpuL = np.append(var.uL,var.uR_edge[:,:,:,:,None],axis=4)
  tmp_normals = np.append(-region.normals[1][:,0:1,:,:], region.normals[0][:,:,:,:],axis=1)
  indx_var = range(0,np.shape(var.uR)[0])
  indices_flux = np.ix_(region.indx_var,region.indx_y_ord,region.indx_z_ord,region.indx_t_ord,indx_x,region.indx_y,region.indx_z)
  tmp = np.zeros(np.shape(tmpuR[indices_flux]))
  fluxFunction(eqns,tmp,region,tmpuR[indices_flux],tmpuL[indices_flux],  tmp_normals[np.ix_(np.array(range(0,3)),indx_x,region.indx_y,region.indx_z)][:,None,None,None,:,:,:,None],fluxArgs)
  fluxVar.fRLS = tmp[:]*1.

  ## Get the up and down fluxes
  fluxArgs = []
#  for i in range(0,nargs):
#    fluxArgs.append(argsU[i][:,:,:,:,:,0:-1,:])
#    fluxArgs.append(argsD[i][:,:,:,:,:,1::,:])

  indx_y = np.append(region.indx_y,region.indx_y[-1] + 1)
  tmpuU = np.append(var.uD_edge[:,:,:,:,:,None],var.uU,axis=5)
  tmpuD = np.append(var.uD,var.uU_edge[:,:,:,:,:,None],axis=5)
  tmp_normals = np.append(-region.normals[3][:,:,0:1,:], region.normals[2][:,:,:,:],axis=2)
  indices_flux = np.ix_(indx_var,region.indx_x_ord,region.indx_z_ord,region.indx_t_ord,region.indx_x,indx_y,region.indx_z)
  tmp = np.zeros(np.shape(tmpuU[indices_flux]))
  fluxFunction(eqns,tmp,region,tmpuU[indices_flux],tmpuD[indices_flux],tmp_normals[np.ix_(np.array(range(0,3)),region.indx_x,indx_y,region.indx_z)][:,None,None,None,:,:,:,None],fluxArgs)
  fluxVar.fUDS = tmp[:]*1.

  ## Get the front and back fluxes
  fluxArgs = []
#  for i in range(0,nargs):
#    fluxArgs.append(argsF[i][:,:,:,:,:,:,0:-1])
#    fluxArgs.append(argsB[i][:,:,:,:,:,:,1::])
  indx_z = np.append(region.indx_z,region.indx_z[-1] + 1)
  tmpuF = np.append(var.uB_edge[:,:,:,:,:,:,None],var.uF,axis=6)
  tmpuB = np.append(var.uB,var.uF_edge[:,:,:,:,:,:,None],axis=6)
  tmp_normals = np.append(-region.normals[5][:,:,:,0:1], region.normals[4][:,:,:,:],axis=3)
  indices_flux = np.ix_(region.indx_var,region.indx_x_ord,region.indx_z_ord,region.indx_t_ord,region.indx_x,region.indx_y,indx_z)
  tmp = np.zeros(np.shape(tmpuF[indices_flux]))
  fluxFunction(eqns,tmp,region,tmpuF[indices_flux],tmpuB[indices_flux],tmp_normals[np.ix_(np.array(range(0,3)),region.indx_x,region.indx_y,indx_z)][:,None,None,None,:,:,:,None],fluxArgs)
  fluxVar.fFBS = tmp[:]*1.


### formulation of flux evaluation to do hyper reduction
def generalFluxGen_hyper(region,eqns,fluxVar,var,fluxFunction,cell_ijk,args):
  nargs = np.shape(args)[0]
#  argsR,argsL,argsU,argsD,argsF,argsB = [],[],[],[],[],[]
#  argsR_edge,argsL_edge,argsU_edge,argsD_edge,argsF_edge,argsB_edge = [],[],[],[],[],[]
#  for i in range(0,nargs):
#    argsR.append(args[i].uR[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
#    argsL.append(args[i].uL[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
#    argsU.append(args[i].uU[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
#    argsD.append(args[i].uD[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
#    argsF.append(args[i].uF[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
#    argsB.append(args[i].uB[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
#    argsR_edge.append(args[i].uR_edge[:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
#    argsL_edge.append(args[i].uL_edge[:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
#    argsU_edge.append(args[i].uU_edge[:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
#    argsD_edge.append(args[i].uD_edge[:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
#    argsF_edge.append(args[i].uF_edge[:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
#    argsB_edge.append(args[i].uB_edge[:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])

  ## Cells
  #cell_ijk = region.cell_ijk
  fluxArgs = []
  for i in range(0,nargs):
    print('Not yet implemented for args in generalFluxGen_hyper')
    #fluxArgs.append(args[i].uR[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
    #fluxArgs.append(args[i].uL[:,:,:,:,(cell_ijk[5][0]+1)%region.Npx,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])

  ftmp = np.zeros(np.shape( fluxVar.fRS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] ) ,dtype=fluxVar.fRS.dtype)
  fluxFunction(eqns,ftmp,region,var.uR[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],var.uL[:,:,:,:, (cell_ijk[5][0]+1)%region.Npx,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],region.normals[0][:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0] ],fluxArgs)
  fluxVar.fRS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] = ftmp[:]*1.
  ##now to same thing for fL
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(args[i].uR[:,:,:,:,cell_ijk[5][0]-1,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
    fluxArgs.append(args[i].uL[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
  ftmp = np.zeros(np.shape( fluxVar.fLS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] ) ,dtype=fluxVar.fLS.dtype)
  fluxFunction(eqns,ftmp,region,var.uR[:,:,:,:,cell_ijk[5][0]-1,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],var.uL[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],-region.normals[1][:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0] ],fluxArgs)
  fluxVar.fLS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] = ftmp[:]*1.
  ## now get contributions from boundary
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(args[i].uR[:,:,:,:,-1,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
    fluxArgs.append(args[i].uR_edge[:,:,:,:,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
  ftmp = np.zeros(np.shape(fluxVar.fRS[:,:,:,:,  -1,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] ) ,dtype=fluxVar.fRS.dtype)
  fluxFunction(eqns,ftmp,region,var.uR[:,:,:,:,  -1,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],var.uR_edge[:,:,:,:,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],region.normals[0][:,-1,cell_ijk[6][0],cell_ijk[7][0] ],fluxArgs)
  fluxVar.fRS[:,:,:,:,-1,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] = ftmp[:]*1.
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(args[i].uL_edge[:,:,:,:,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
    fluxArgs.append(args[i].uL[:,:,:,:,0,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
  ftmp = np.zeros(np.shape(fluxVar.fLS[:,:,:,:,  0,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] ) ,dtype=fluxVar.fLS.dtype)
  fluxFunction(eqns,ftmp,region,var.uL_edge[:,:,:,:,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],var.uL[:,:,:,:,0,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],-region.normals[1][:,0,cell_ijk[6][0],cell_ijk[7][0] ],fluxArgs)
  fluxVar.fLS[:,:,:,:,0 ,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] = ftmp[:]*1.

  ## Get the up and down fluxes
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(args[i].uU[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
    fluxArgs.append(args[i].uD[:,:,:,:,cell_ijk[5][0],(cell_ijk[6][0]+1)%region.Npy,cell_ijk[7][0],cell_ijk[8][0]])
  ftmp = np.zeros(np.shape( fluxVar.fUS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] ) ,dtype=fluxVar.fUS.dtype)
  fluxFunction(eqns,ftmp,region,var.uU[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],var.uD[:,:,:,:, cell_ijk[5][0],(cell_ijk[6][0]+1)%region.Npy,cell_ijk[7][0],cell_ijk[8][0]],region.normals[2][:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0] ],fluxArgs)
  fluxVar.fUS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] = ftmp[:]*1.

  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(args[i].uU[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0]-1,cell_ijk[7][0],cell_ijk[8][0]])
    fluxArgs.append(args[i].uD[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
  ftmp = np.zeros(np.shape( fluxVar.fDS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] ) ,dtype=fluxVar.fDS.dtype)
  fluxFunction(eqns,ftmp,region,var.uU[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0]-1,cell_ijk[7][0],cell_ijk[8][0]],var.uD[:,:,:,:, cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],-region.normals[3][:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0] ],fluxArgs)
  fluxVar.fDS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] = ftmp[:]*1.
  ## now get contributions from the boundary
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(args[i].uU[:,:,:,:,cell_ijk[5][0],-1,cell_ijk[7][0],cell_ijk[8][0]])
    fluxArgs.append(args[i].uU_edge[:,:,:,:,cell_ijk[5][0],cell_ijk[7][0],cell_ijk[8][0]])
  ftmp = np.zeros(np.shape(fluxVar.fUS[:,:,:,:,cell_ijk[5][0],-1,cell_ijk[7][0],cell_ijk[8][0]] ) ,dtype=fluxVar.fUS.dtype)
  fluxFunction(eqns,ftmp,region,var.uU[:,:,:,:,cell_ijk[5][0],  -1,cell_ijk[7][0],cell_ijk[8][0]],var.uU_edge[:,:,:,:,cell_ijk[5][0],cell_ijk[7][0],cell_ijk[8][0]],region.normals[2][:,cell_ijk[5][0],-1,cell_ijk[7][0] ],fluxArgs)
  fluxVar.fUS[:,:,:,:,cell_ijk[5][0],-1,cell_ijk[7][0],cell_ijk[8][0]] = ftmp[:]*1.

  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(args[i].uD_edge[:,:,:,:,cell_ijk[5][0],cell_ijk[7][0],cell_ijk[8][0]])
    fluxArgs.append(args[i].uD[:,:,:,:,cell_ijk[5][0],0,cell_ijk[7][0],cell_ijk[8][0]])
  ftmp = np.zeros(np.shape(fluxVar.fDS[:,:,:,:,cell_ijk[5][0],0,cell_ijk[7][0],cell_ijk[8][0]] ) ,dtype=fluxVar.fDS.dtype)
  fluxFunction(eqns,ftmp,region,var.uD_edge[:,:,:,:,cell_ijk[5][0],cell_ijk[7][0],cell_ijk[8][0]],var.uD[:,:,:,:,cell_ijk[5][0],0,cell_ijk[7][0],cell_ijk[8][0]],-region.normals[3][:,cell_ijk[5][0],0,cell_ijk[7][0] ],fluxArgs)
  fluxVar.fDS[:,:,:,:,cell_ijk[5][0],0,cell_ijk[7][0],cell_ijk[8][0]] = ftmp[:]*1.

  # now get up and down
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(args[i].uF[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
    fluxArgs.append(args[i].uB[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],(cell_ijk[7][0]+1)%region.Npz,cell_ijk[8][0]])
  ftmp = np.zeros(np.shape( fluxVar.fFS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] ) ,dtype=fluxVar.fFS.dtype)
  fluxFunction(eqns,ftmp,region,var.uF[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],var.uB[:,:,:,:, cell_ijk[5][0],cell_ijk[6][0],(cell_ijk[7][0]+1)%region.Npz,cell_ijk[8][0]],region.normals[4][:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0] ],fluxArgs)
  fluxVar.fFS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] = ftmp[:]*1.

  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(args[i].uF[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]-1,cell_ijk[8][0]])
    fluxArgs.append(args[i].uB[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
  ftmp = np.zeros(np.shape( fluxVar.fBS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] ) ,dtype=fluxVar.fBS.dtype)
  fluxFunction(eqns,ftmp,region,var.uF[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]-1,cell_ijk[8][0]],var.uB[:,:,:,:, cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],-region.normals[5][:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0] ],fluxArgs)
  fluxVar.fBS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] = ftmp[:]*1.
  ## now get contributions from the boundary
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(args[i].uF[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],-1,cell_ijk[8][0]])
    fluxArgs.append(args[i].uF_edge[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[8][0]])
  ftmp = np.zeros(np.shape(fluxVar.fFS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],-1,cell_ijk[8][0]] ) ,dtype=fluxVar.fFS.dtype)
  fluxFunction(eqns,ftmp,region,var.uF[:,:,:,:,cell_ijk[5][0], cell_ijk[6][0],-1,cell_ijk[8][0]],var.uF_edge[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[8][0]],region.normals[4][:,cell_ijk[5][0],cell_ijk[6][0],-1 ],fluxArgs)
  fluxVar.fFS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],-1,cell_ijk[8][0]] = ftmp[:]*1.

  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(args[i].uB_edge[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[8][0]])
    fluxArgs.append(args[i].uB[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],0,cell_ijk[8][0]])
  fluxFunction(eqns,ftmp,region,var.uB_edge[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[8][0]],var.uB[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],0,cell_ijk[8][0]],-region.normals[5][:,cell_ijk[5][0],cell_ijk[6][0],0 ],fluxArgs)
  fluxVar.fBS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],0,cell_ijk[8][0]] = ftmp[:]*1.


def generalFluxGen_SWE(region,eqns,fluxVar,var,fluxFunction,args):
  nargs = np.shape(args)[0]
  argsR,argsL,argsU,argsD,argsF,argsB = [],[],[],[],[],[]
  argsR_edge,argsL_edge,argsU_edge,argsD_edge,argsF_edge,argsB_edge = [],[],[],[],[],[]
  for i in range(0,nargs):
    #main.a.uR[:],main.a.uL[:],main.a.uU[:],main.a.uD[:],main.a.uF[:],main.a.uB[:] = main.basis.reconstructEdgesGeneral(main.a.a,main)   
    #tmpR,tmpL,tmpU,tmpD,tmpF,tmpB = main.basis.reconstructEdgesGeneral(args[i],main)
    #tmpR_edge,tmpL_edge,tmpU_edge,tmpD_edge,tmpF_edge,tmpB_edge = sendEdgesGeneralSlab(tmpL,tmpR,tmpD,tmpU,tmpB,tmpF,main)
    argsR.append(args[i].uR)
    argsL.append(args[i].uL)
    argsU.append(args[i].uU)
    argsD.append(args[i].uD)
    argsF.append(args[i].uF)
    argsB.append(args[i].uB)
    argsR_edge.append(args[i].uR_edge)
    argsL_edge.append(args[i].uL_edge)
    argsU_edge.append(args[i].uU_edge)
    argsD_edge.append(args[i].uD_edge)
    argsF_edge.append(args[i].uF_edge)
    argsB_edge.append(args[i].uB_edge)

  ## Get left and right fluxes
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsR[i][:,:,:,:,0:-1,:,:])
    fluxArgs.append(argsL[i][:,:,:,:,1::,:,: ])
  fluxFunction(eqns,fluxVar.fRLS[:,:,:,:,1:-1,:,:],region,var.uR[:,:,:,:,0:-1,:,:],var.uL[:,:,:,:,1::,:,:],region.normals[0][:,None,None,None,0:-1,:,:,None],[region.surface_R[:,:,:,:,0:-1,:,:],region.surface_L[:,:,:,:,1::,:,:]])
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsR[i][:,:,:,:,-1,:,:])
    fluxArgs.append(argsR_edge[i])

  fluxFunction(eqns,fluxVar.fRLS[:,:,:,:,  -1,:,:],region,var.uR[:,:,:,:,  -1,:,:],var.uR_edge,region.normals[0][:,None,None,None,-1,:,:,None],[var.surface_R[:,:,:,:,  -1,:,:],var.surface_R[:,:,:,:,  -1,:,:]])
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsL_edge[i])
    fluxArgs.append(argsL[i][:,:,:,:,0,:,:])
  fluxFunction(eqns,fluxVar.fRLS[:,:,:,:,0   ,:,:],region,var.uL_edge,var.uL[:,:,:,:,0,:,:],-region.normals[1][:,None,None,None,0,:,:,None],[var.surface_L[:,:,:,:,0,:,:],var.surface_L[:,:,:,:,0,:,:] ])

  ## Get the up and down fluxes
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsU[i][:,:,:,:,:,0:-1,:])
    fluxArgs.append(argsD[i][:,:,:,:,:,1::,:])
  fluxFunction(eqns,fluxVar.fUDS[:,:,:,:,:,1:-1,:],region,var.uU[:,:,:,:,:,0:-1,:],var.uD[:,:,:,:,:,1::,:],region.normals[2][:,None,None,None,:,0:-1,:,None],[var.surface_U[:,:,:,:,:,0:-1,:],var.surface_D[:,:,:,:,:,1::,:]])

  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsU[i][:,:,:,:,:,  -1,:])
    fluxArgs.append(argsU_edge[i])
  fluxFunction(eqns,fluxVar.fUDS[:,:,:,:,:,  -1,:],region,var.uU[:,:,:,:,:,  -1,:],var.uU_edge,region.normals[2][:,None,None,None,:,-1,:,None],fluxArgs)

  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsD_edge[i])
    fluxArgs.append(argsD[i][:,:,:,:,:,0,:])
  fluxFunction(eqns,fluxVar.fUDS[:,:,:,:,:,0   ,:],region,var.uD_edge,var.uD[:,:,:,:,:,0,:],-region.normals[3][:,None,None,None,:,0,:,None],fluxArgs)

  ## Get the front and back fluxes
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsF[i][:,:,:,:,:,:,0:-1])
    fluxArgs.append(argsB[i][:,:,:,:,:,:,1::])
  fluxFunction(eqns,fluxVar.fFBS[:,:,:,:,:,:,1:-1],region,var.uF[:,:,:,:,:,:,0:-1],var.uB[:,:,:,:,:,:,1::],region.normals[4][:,None,None,None,:,:,0:-1,None],fluxArgs)

  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsF[i][:,:,:,:,:,:,-1])
    fluxArgs.append(argsF_edge[i])
  fluxFunction(eqns,fluxVar.fFBS[:,:,:,:,:,:,  -1],region,var.uF[:,:,:,:,:,:,  -1],var.uF_edge,region.normals[4][:,None,None,None,:,:,-1,None],fluxArgs)

  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsB_edge[i])
    fluxArgs.append(argsB[i][:,:,:,:,:,:,0])
  fluxFunction(eqns,fluxVar.fFBS[:,:,:,:,:,:,0   ],region,var.uB_edge,var.uB[:,:,:,:,:,:,0],-region.normals[5][:,None,None,None,:,:,0,None],fluxArgs)




def generalFluxGen(region,eqns,fluxVar,var,fluxFunction,args):
  nargs = np.shape(args)[0]
  argsR,argsL,argsU,argsD,argsF,argsB = [],[],[],[],[],[]
  argsR_edge,argsL_edge,argsU_edge,argsD_edge,argsF_edge,argsB_edge = [],[],[],[],[],[]
  for i in range(0,nargs):
    #main.a.uR[:],main.a.uL[:],main.a.uU[:],main.a.uD[:],main.a.uF[:],main.a.uB[:] = main.basis.reconstructEdgesGeneral(main.a.a,main)   
    #tmpR,tmpL,tmpU,tmpD,tmpF,tmpB = main.basis.reconstructEdgesGeneral(args[i],main)
    #tmpR_edge,tmpL_edge,tmpU_edge,tmpD_edge,tmpF_edge,tmpB_edge = sendEdgesGeneralSlab(tmpL,tmpR,tmpD,tmpU,tmpB,tmpF,main)
    argsR.append(args[i].uR)
    argsL.append(args[i].uL)
    argsU.append(args[i].uU)
    argsD.append(args[i].uD)
    argsF.append(args[i].uF)
    argsB.append(args[i].uB)
    argsR_edge.append(args[i].uR_edge)
    argsL_edge.append(args[i].uL_edge)
    argsU_edge.append(args[i].uU_edge)
    argsD_edge.append(args[i].uD_edge)
    argsF_edge.append(args[i].uF_edge)
    argsB_edge.append(args[i].uB_edge)

  ## Get left and right fluxes
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsR[i][:,:,:,:,0:-1,:,:])
    fluxArgs.append(argsL[i][:,:,:,:,1::,:,: ])
  fluxFunction(eqns,fluxVar.fRLS[:,:,:,:,1:-1,:,:],region,var.uR[:,:,:,:,0:-1,:,:],var.uL[:,:,:,:,1::,:,:],region.normals[0][:,None,None,None,0:-1,:,:,None],fluxArgs)
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsR[i][:,:,:,:,-1,:,:])
    fluxArgs.append(argsR_edge[i])

  fluxFunction(eqns,fluxVar.fRLS[:,:,:,:,  -1,:,:],region,var.uR[:,:,:,:,  -1,:,:],var.uR_edge,region.normals[0][:,None,None,None,-1,:,:,None],fluxArgs)
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsL_edge[i])
    fluxArgs.append(argsL[i][:,:,:,:,0,:,:])
  fluxFunction(eqns,fluxVar.fRLS[:,:,:,:,0   ,:,:],region,var.uL_edge,var.uL[:,:,:,:,0,:,:],-region.normals[1][:,None,None,None,0,:,:,None],fluxArgs)

  ## Get the up and down fluxes
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsU[i][:,:,:,:,:,0:-1,:])
    fluxArgs.append(argsD[i][:,:,:,:,:,1::,:])
  fluxFunction(eqns,fluxVar.fUDS[:,:,:,:,:,1:-1,:],region,var.uU[:,:,:,:,:,0:-1,:],var.uD[:,:,:,:,:,1::,:],region.normals[2][:,None,None,None,:,0:-1,:,None],fluxArgs)

  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsU[i][:,:,:,:,:,  -1,:])
    fluxArgs.append(argsU_edge[i])
  fluxFunction(eqns,fluxVar.fUDS[:,:,:,:,:,  -1,:],region,var.uU[:,:,:,:,:,  -1,:],var.uU_edge,region.normals[2][:,None,None,None,:,-1,:,None],fluxArgs)

  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsD_edge[i])
    fluxArgs.append(argsD[i][:,:,:,:,:,0,:])
  fluxFunction(eqns,fluxVar.fUDS[:,:,:,:,:,0   ,:],region,var.uD_edge,var.uD[:,:,:,:,:,0,:],-region.normals[3][:,None,None,None,:,0,:,None],fluxArgs)

  ## Get the front and back fluxes
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsF[i][:,:,:,:,:,:,0:-1])
    fluxArgs.append(argsB[i][:,:,:,:,:,:,1::])
  fluxFunction(eqns,fluxVar.fFBS[:,:,:,:,:,:,1:-1],region,var.uF[:,:,:,:,:,:,0:-1],var.uB[:,:,:,:,:,:,1::],region.normals[4][:,None,None,None,:,:,0:-1,None],fluxArgs)

  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsF[i][:,:,:,:,:,:,-1])
    fluxArgs.append(argsF_edge[i])
  fluxFunction(eqns,fluxVar.fFBS[:,:,:,:,:,:,  -1],region,var.uF[:,:,:,:,:,:,  -1],var.uF_edge,region.normals[4][:,None,None,None,:,:,-1,None],fluxArgs)

  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsB_edge[i])
    fluxArgs.append(argsB[i][:,:,:,:,:,:,0])
  fluxFunction(eqns,fluxVar.fFBS[:,:,:,:,:,:,0   ],region,var.uB_edge,var.uB[:,:,:,:,:,:,0],-region.normals[5][:,None,None,None,:,:,0,None],fluxArgs)






def generalFluxGenStrong(region,eqns,fluxVar,var,fluxFunction,args):
  nargs = np.shape(args)[0]
  argsR,argsL,argsU,argsD,argsF,argsB = [],[],[],[],[],[]
  argsR_edge,argsL_edge,argsU_edge,argsD_edge,argsF_edge,argsB_edge = [],[],[],[],[],[]
  for i in range(0,nargs):
    argsR.append(args[i].uR)
    argsL.append(args[i].uL)
    argsU.append(args[i].uU)
    argsD.append(args[i].uD)
    argsF.append(args[i].uF)
    argsB.append(args[i].uB)
    argsR_edge.append(args[i].uR_edge)
    argsL_edge.append(args[i].uL_edge)
    argsU_edge.append(args[i].uU_edge)
    argsD_edge.append(args[i].uD_edge)
    argsF_edge.append(args[i].uF_edge)
    argsB_edge.append(args[i].uB_edge)

  ## Get left and right fluxes
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsR[i][:,:,:,:,0:-1,:,:])
    fluxArgs.append(argsL[i][:,:,:,:,1::,:,: ])
  fluxFunction(eqns,fluxVar.fRLS[:,:,:,:,1:-1,:,:],region,var.uR[:,:,:,:,0:-1,:,:],var.uL[:,:,:,:,1::,:,:],region.normals[0][:,None,None,None,0:-1,:,:,None],fluxArgs)
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsR[i][:,:,:,:,-1,:,:])
    fluxArgs.append(argsR_edge[i])

  fluxFunction(eqns,fluxVar.fRLS[:,:,:,:,  -1,:,:],region,var.uR[:,:,:,:,  -1,:,:],var.uR_edge,region.normals[0][:,None,None,None,-1,:,:,None],fluxArgs)
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsL_edge[i])
    fluxArgs.append(argsL[i][:,:,:,:,0,:,:])
  fluxFunction(eqns,fluxVar.fRLS[:,:,:,:,0   ,:,:],region,var.uL_edge,var.uL[:,:,:,:,0,:,:],-region.normals[1][:,None,None,None,0,:,:,None],fluxArgs)

  fluxVar.fRS[:] = fluxVar.fRLS[:,:,:,:,1::] - eqns.basicFlux(eqns,region,var.uR,region.normals[0][:,None,None,None,:,:,:,None],fluxArgs)
  fluxVar.fLS[:] = fluxVar.fRLS[:,:,:,:,0:-1] - eqns.basicFlux(eqns,region,var.uL,-region.normals[1][:,None,None,None,:,:,:,None],fluxArgs)

  ## Get the up and down fluxes
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsU[i][:,:,:,:,:,0:-1,:])
    fluxArgs.append(argsD[i][:,:,:,:,:,1::,:])
  fluxFunction(eqns,fluxVar.fUDS[:,:,:,:,:,1:-1,:],region,var.uU[:,:,:,:,:,0:-1,:],var.uD[:,:,:,:,:,1::,:],region.normals[2][:,None,None,None,:,0:-1,:,None],fluxArgs)

  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsU[i][:,:,:,:,:,  -1,:])
    fluxArgs.append(argsU_edge[i])
  fluxFunction(eqns,fluxVar.fUDS[:,:,:,:,:,  -1,:],region,var.uU[:,:,:,:,:,  -1,:],var.uU_edge,region.normals[2][:,None,None,None,:,-1,:,None],fluxArgs)

  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsD_edge[i])
    fluxArgs.append(argsD[i][:,:,:,:,:,0,:])
  fluxFunction(eqns,fluxVar.fUDS[:,:,:,:,:,0   ,:],region,var.uD_edge,var.uD[:,:,:,:,:,0,:],-region.normals[3][:,None,None,None,:,0,:,None],fluxArgs)


  fluxVar.fUS[:] = fluxVar.fUDS[:,:,:,:,:,1::] - eqns.basicFlux(eqns,region,var.uU,region.normals[2][:,None,None,None,:,:,:,None],fluxArgs)
  fluxVar.fDS[:] = fluxVar.fUDS[:,:,:,:,:,0:-1] - eqns.basicFlux(eqns,region,var.uD,-region.normals[3][:,None,None,None,:,:,:,None],fluxArgs)

  ## Get the front and back fluxes
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsF[i][:,:,:,:,:,:,0:-1])
    fluxArgs.append(argsB[i][:,:,:,:,:,:,1::])
  fluxFunction(eqns,fluxVar.fFBS[:,:,:,:,:,:,1:-1],region,var.uF[:,:,:,:,:,:,0:-1],var.uB[:,:,:,:,:,:,1::],region.normals[4][:,None,None,None,:,:,0:-1,None],fluxArgs)

  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsF[i][:,:,:,:,:,:,-1])
    fluxArgs.append(argsF_edge[i])
  fluxFunction(eqns,fluxVar.fFBS[:,:,:,:,:,:,  -1],region,var.uF[:,:,:,:,:,:,  -1],var.uF_edge,region.normals[4][:,None,None,None,:,:,-1,None],fluxArgs)

  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsB_edge[i])
    fluxArgs.append(argsB[i][:,:,:,:,:,:,0])
  fluxFunction(eqns,fluxVar.fFBS[:,:,:,:,:,:,0   ],region,var.uB_edge,var.uB[:,:,:,:,:,:,0],-region.normals[5][:,None,None,None,:,:,0,None],fluxArgs)

  fluxVar.fFS[:] = fluxVar.fFBS[:,:,:,:,:,:,1::] - eqns.basicFlux(eqns,region,var.uF,region.normals[4][:,None,None,None,:,:,:,None],fluxArgs)
  fluxVar.fBS[:] = fluxVar.fFBS[:,:,:,:,:,:,0:-1] - eqns.basicFlux(eqns,region,var.uB,-region.normals[5][:,None,None,None,:,:,:,None],fluxArgs)







def generalFluxGen_element(main,eqns,fluxVar,var,fluxFunction,args):
  ### Flux function that only evaluate the fluxes using quantities local to the element
  # this is a type of double flux function since fL does not equal fR  
  nargs = np.shape(args)[0]
  argsR,argsL,argsU,argsD,argsF,argsB = [],[],[],[],[],[]
  argsR_edge,argsL_edge,argsU_edge,argsD_edge,argsF_edge,argsB_edge = [],[],[],[],[],[]
  for i in range(0,nargs):
    #main.a.uR[:],main.a.uL[:],main.a.uU[:],main.a.uD[:],main.a.uF[:],main.a.uB[:] = main.basis.reconstructEdgesGeneral(main.a.a,main)   
    #tmpR,tmpL,tmpU,tmpD,tmpF,tmpB = main.basis.reconstructEdgesGeneral(args[i],main)
    #tmpR_edge,tmpL_edge,tmpU_edge,tmpD_edge,tmpF_edge,tmpB_edge = sendEdgesGeneralSlab(tmpL,tmpR,tmpD,tmpU,tmpB,tmpF,main)
    argsR.append(args[i].uR)
    argsL.append(args[i].uL)
    argsU.append(args[i].uU)
    argsD.append(args[i].uD)
    argsF.append(args[i].uF)
    argsB.append(args[i].uB)
    argsR_edge.append(args[i].uR_edge)
    argsL_edge.append(args[i].uL_edge)
    argsU_edge.append(args[i].uU_edge)
    argsD_edge.append(args[i].uD_edge)
    argsF_edge.append(args[i].uF_edge)
    argsB_edge.append(args[i].uB_edge)

  ## Get left and right fluxes
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsR[i][:,:,:,:,0:-1,:,:])
    fluxArgs.append(argsL[i][:,:,:,:,1::,:,: ])
  fluxFunction(fluxVar.fR[:,:,:,:,0:-1,:,:],main,var.uR[:,:,:,:,0:-1,:,:],0.*var.uL[:,:,:,:,1::,:,:],main.normals[0][:,None,None,None,0:-1,:,:,None],fluxArgs)
  fluxFunction(fluxVar.fL[:,:,:,:,1::,:,:],main,var.uR[:,:,:,:,0:-1,:,:]*0.,var.uL[:,:,:,:,1::,:,:],main.normals[0][:,None,None,None,0:-1,:,:,None],fluxArgs)

  fluxArgs = []

  for i in range(0,nargs):
    fluxArgs.append(argsR[i][:,:,:,:,-1,:,:])
    fluxArgs.append(argsR_edge[i])

  fluxFunction(fluxVar.fR[:,:,:,:,  -1,:,:],main,var.uR[:,:,:,:,  -1,:,:],0.*var.uR_edge,main.normals[0][:,None,None,None,-1,:,:,None],fluxArgs)
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsL_edge[i])
    fluxArgs.append(argsL[i][:,:,:,:,0,:,:])
  fluxFunction(fluxVar.fL[:,:,:,:,0   ,:,:],main,var.uL_edge*0.,var.uL[:,:,:,:,0,:,:],-main.normals[1][:,None,None,None,0,:,:,None],fluxArgs)

  ## Get the up and down fluxes
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsU[i][:,:,:,:,:,0:-1,:])
    fluxArgs.append(argsD[i][:,:,:,:,:,1::,:])
  fluxFunction(fluxVar.fU[:,:,:,:,:,0:-1,:],main,var.uU[:,:,:,:,:,0:-1,:],0.*var.uD[:,:,:,:,:,1::,:],main.normals[2][:,None,None,None,:,0:-1,:,None],fluxArgs)
  fluxFunction(fluxVar.fD[:,:,:,:,:,1:: ,:],main,var.uU[:,:,:,:,:,0:-1,:]*0.,var.uD[:,:,:,:,:,1::,:],main.normals[2][:,None,None,None,:,0:-1,:,None],fluxArgs)

  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsU[i][:,:,:,:,:,  -1,:])
    fluxArgs.append(argsU_edge[i])
  fluxFunction(fluxVar.fU[:,:,:,:,:,  -1,:],main,var.uU[:,:,:,:,:,  -1,:],var.uU_edge*0.,main.normals[2][:,None,None,None,:,-1,:,None],fluxArgs)

  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsD_edge[i])
    fluxArgs.append(argsD[i][:,:,:,:,:,0,:])
  fluxFunction(fluxVar.fD[:,:,:,:,:,0   ,:],main,var.uD_edge*0.,var.uD[:,:,:,:,:,0,:],-main.normals[3][:,None,None,None,:,0,:,None],fluxArgs)

  ## Get the front and back fluxes
  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsF[i][:,:,:,:,:,:,0:-1])
    fluxArgs.append(argsB[i][:,:,:,:,:,:,1::])
  fluxFunction(fluxVar.fF[:,:,:,:,:,:,0:-1],main,var.uF[:,:,:,:,:,:,0:-1],0.*var.uB[:,:,:,:,:,:,1::],main.normals[4][:,None,None,None,:,:,0:-1,None],fluxArgs)
  fluxFunction(fluxVar.fB[:,:,:,:,:,:,1:: ],main,var.uF[:,:,:,:,:,:,0:-1]*0.,var.uB[:,:,:,:,:,:,1::],main.normals[4][:,None,None,None,:,:,0:-1,None],fluxArgs)

  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsF[i][:,:,:,:,:,:,-1])
    fluxArgs.append(argsF_edge[i])
  fluxFunction(fluxVar.fF[:,:,:,:,:,:,  -1],main,var.uF[:,:,:,:,:,:,  -1],0.*var.uF_edge,main.normals[4][:,None,None,None,:,:,-1,None],fluxArgs)

  fluxArgs = []
  for i in range(0,nargs):
    fluxArgs.append(argsB_edge[i])
    fluxArgs.append(argsB[i][:,:,:,:,:,:,0])
  fluxFunction(fluxVar.fB[:,:,:,:,:,:,0   ],main,var.uB_edge*0.,var.uB[:,:,:,:,:,:,0],-main.normals[5][:,None,None,None,:,:,0,None],fluxArgs)




