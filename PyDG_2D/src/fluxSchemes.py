import numpy as np

def starState(var,main):
  uRLS = np.zeros((var.nvars,var.quadpoints,main.Npx+1,main.Npy))
  uUDS = np.zeros((var.nvars,var.quadpoints,main.Npx,main.Npy+1))
  uRLS[:,:,1:-1,:] = 0.5*(var.uR[:,:,0:-1,:] + var.uL[:,:,1::,:])
  uRLS[:,:,  -1,:] = 0.5*(var.uR[:,:,  -1,:] + var.uR_edge)
  uRLS[:,:,   0,:] = 0.5*(var.uL[:,:,0  ,:]  + var.uL_edge)
  uUDS[:,:,:,1:-1] = 0.5*(var.uU[:,:,:,0:-1] + var.uD[:,:,:,1::])
  uUDS[:,:,:,  -1] = 0.5*(var.uU[:,:,:,  -1] + var.uU_edge)
  uUDS[:,:,:,   0] = 0.5*(var.uD[:,:,:,   0] + var.uD_edge)
  return uRLS,uUDS

#def starState(var):
#  var.u.uRS[:,:,0:-1,:] = 0.5*(var.u.uR[:,:,0:-1,:] + var.u.uL[:,:,1::,:])
#  var.u.uRS[:,:,  -1,:] = 0.5*(var.u.uR[:,:,  -1,:] + var.u.uL[:,:,0  ,:])
#  var.u.uLS[:,:,1:: ,:] = var.u.uRS[:,:,0:-1,:]
#  var.u.uLS[:,:,0   ,:] = var.u.uRS[:,:,  -1,:]
#  var.u.uUS[:,:,:,0:-1] = 0.5*(var.u.uU[:,:,:,0:-1] + var.u.uD[:,:,:,1::])
#  var.u.uUS[:,:,:,  -1] = 0.5*(var.u.uU[:,:,:,  -1] + var.u.uU_edge)
#  var.u.uDS[:,:,:,1:: ] = var.fUS[:,:,:,0:-1]
#  var.u.uDS[:,:,:,0   ] = 0.5*(var.uD[:,:,:,   0] + var.uD_edge)

def centralFluxGeneral(fR,fL,fU,fD,fR_edge,fL_edge,fU_edge,fD_edge):
  fRS = np.zeros(np.shape(fR))
  fLS = np.zeros(np.shape(fL))
  fUS = np.zeros(np.shape(fU))
  fDS = np.zeros(np.shape(fD))
  fRS[:,:,0:-1,:] = 0.5*(fR[:,:,0:-1,:] + fL[:,:,1::,:])
  fRS[:,:,  -1,:] = 0.5*(fR[:,:,  -1,:] + fR_edge)
  fLS[:,:,1:: ,:] = fRS[:,:,0:-1,:]
  fLS[:,:,0   ,:] = 0.5*(fL[:,:,0,:]    + fL_edge)
  fUS[:,:,:,0:-1] = 0.5*(fU[:,:,:,0:-1] + fD[:,:,:,1::])
  fUS[:,:,:,  -1] = 0.5*(fU[:,:,:,  -1] + fU_edge)
  fDS[:,:,:,1:: ] = fUS[:,:,:,0:-1]
  fDS[:,:,:,0   ] = 0.5*(fD[:,:,:,   0] + fD_edge)
  return fRS,fLS,fUS,fDS


def centralFlux(main,eqns,schemes,fluxVar,var):
  fluxVar.fRS[:,:,0:-1,:] = 0.5*(fluxVar.fR[:,:,0:-1,:] + fluxVar.fL[:,:,1::,:])
  fluxVar.fRS[:,:,  -1,:] = 0.5*(fluxVar.fR[:,:,  -1,:] + fluxVar.fR_edge)
  fluxVar.fLS[:,:,1:: ,:] = fluxVar.fRS[:,:,0:-1,:]
  fluxVar.fLS[:,:,0   ,:] = 0.5*(fluxVar.fL[:,:,0,:]    + fluxVar.fL_edge)
  fluxVar.fUS[:,:,:,0:-1] = 0.5*(fluxVar.fU[:,:,:,0:-1] + fluxVar.fD[:,:,:,1::])
  fluxVar.fUS[:,:,:,  -1] = 0.5*(fluxVar.fU[:,:,:,  -1] + fluxVar.fU_edge)
  fluxVar.fDS[:,:,:,1:: ] = fluxVar.fUS[:,:,:,0:-1]
  fluxVar.fDS[:,:,:,0   ] = 0.5*(fluxVar.fD[:,:,:,   0] + fluxVar.fD_edge)


def rusanovFlux(main,eqns,schemes,fluxVar,var):
  uRLS,uUDS = starState(var,main)
  eigsRL,eigsUD = eqns.getEigs(uRLS,uUDS)
  fluxVar.fRS[:,:,0:-1,:] = 0.5*(fluxVar.fR[:,:,0:-1,:] + fluxVar.fL[:,:,1::,:]) - 0.5*eigsRL[:,:,1:-1,:]*(var.uL[:,:,1::,:] - var.uR[:,:,0:-1,:])
  fluxVar.fRS[:,:,  -1,:] = 0.5*(fluxVar.fR[:,:,  -1,:] + fluxVar.fR_edge) - 0.5*eigsRL[:,:,  -1,:]*(var.uR_edge - var.uR[:,:,  -1,:])
  fluxVar.fLS[:,:,1:: ,:] = fluxVar.fRS[:,:,0:-1,:]
  fluxVar.fLS[:,:,0   ,:] = 0.5*(fluxVar.fL_edge + fluxVar.fL[:,:,0,:] )         - 0.5*eigsRL[:,:,0,   :]*(var.uL[:,:,0,:  ] - var.uL_edge) 
  fluxVar.fUS[:,:,:,0:-1] = 0.5*(fluxVar.fU[:,:,:,0:-1] + fluxVar.fD[:,:,:,1::]) - 0.5*eigsUD[:,:,:,1:-1]*(var.uD[:,:,:,1::] - var.uU[:,:,:,0:-1])
  fluxVar.fUS[:,:,:,  -1] = 0.5*(fluxVar.fU[:,:,:,  -1] + fluxVar.fU_edge) -       0.5*eigsUD[:,:,:,  -1]*(var.uU_edge       - var.uU[:,:,:,  -1])
  fluxVar.fDS[:,:,:,1:: ] = 0.5*(fluxVar.fU[:,:,:,0:-1] + fluxVar.fD[:,:,:,1::]) - 0.5*eigsUD[:,:,:,1:-1]*(var.uD[:,:,:,1::] - var.uU[:,:,:,0:-1])
  fluxVar.fDS[:,:,:,0   ] = 0.5*(fluxVar.fD_edge + fluxVar.fD[:,:,:,0])          - 0.5*eigsUD[:,:,:,   0]*(var.uD[:,:,:,0  ] - var.uD_edge)

