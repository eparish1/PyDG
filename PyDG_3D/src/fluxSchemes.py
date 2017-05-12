import numpy as np

def centralFluxGeneral(fR,fL,fU,fD,fF,fB,fR_edge,fL_edge,fU_edge,fD_edge,fF_edge,fB_edge):
  fRS = np.zeros(np.shape(fR))
  fLS = np.zeros(np.shape(fL))
  fUS = np.zeros(np.shape(fU))
  fDS = np.zeros(np.shape(fD))
  fFS = np.zeros(np.shape(fD))
  fBS = np.zeros(np.shape(fD))

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

def inviscidFlux(main,eqns,fluxVar,var):
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
  fluxVar.fFS[:,:,:,:,:,0:-1] = eqns.inviscidFlux(var.uF[:,:,:,:,:,0:-1],var.uB[:,:,:,:,:,1::],nz )
  fluxVar.fFS[:,:,:,:,:,  -1] = eqns.inviscidFlux(var.uF[:,:,:,:,:,  -1],var.uF_edge,nz)
  fluxVar.fBS[:,:,:,:,:,1:: ] = fluxVar.fFS[:,:,:,:,:,0:-1] 
  fluxVar.fBS[:,:,:,:,:,0   ] = eqns.inviscidFlux(var.uB_edge,var.uB[:,:,:,:,:,0],nz)

