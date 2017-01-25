import numpy as np

def starState(var,main):
  uRLS = np.zeros((var.nvars,main.order,main.Npx+1,main.Npy))
  uUDS = np.zeros((var.nvars,main.order,main.Npx,main.Npy+1))
  uRLS[:,:,1:-1,:] = 0.5*(var.uR[:,:,0:-1,:] + var.uL[:,:,1::,:])
  uRLS[:,:,  -1,:] = 0.5*(var.uR[:,:,  -1,:] + var.uL[:,:,0  ,:])
  uRLS[:,:,   0,:] = 0.5*(var.uR[:,:,  -1,:] + var.uL[:,:,0  ,:])
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


def centralFlux(main,eqns,schemes,fluxVar,var):
  fluxVar.fRS[:,:,0:-1,:] = 0.5*(fluxVar.fR[:,:,0:-1,:] + fluxVar.fL[:,:,1::,:])
  fluxVar.fRS[:,:,  -1,:] = 0.5*(fluxVar.fR[:,:,  -1,:] + fluxVar.fL[:,:,0  ,:])
  fluxVar.fLS[:,:,1:: ,:] = fluxVar.fRS[:,:,0:-1,:]
  fluxVar.fLS[:,:,0   ,:] = fluxVar.fRS[:,:,  -1,:]
  fluxVar.fUS[:,:,:,0:-1] = 0.5*(fluxVar.fU[:,:,:,0:-1] + fluxVar.fD[:,:,:,1::])
  fluxVar.fUS[:,:,:,  -1] = 0.5*(fluxVar.fU[:,:,:,  -1] + fluxVar.fU_edge)
  fluxVar.fDS[:,:,:,1:: ] = fluxVar.fUS[:,:,:,0:-1]
  fluxVar.fDS[:,:,:,0   ] = 0.5*(fluxVar.fD[:,:,:,   0] + fluxVar.fD_edge)


def rusanovFlux(main,eqns,schemes,fluxVar,var):
  uRLS,uUDS = starState(var,main)
  eigsRL,eigsUD = eqns.getEigs(uRLS,uUDS)

  fluxVar.fRS[:,:,0:-1,:] = 0.5*(fluxVar.fR[:,:,0:-1,:] + fluxVar.fL[:,:,1::,:]) - 0.5*eigsRL[:,:,1:-1,:]*(var.uL[:,:,1::,:] - var.uR[:,:,0:-1,:])
  fluxVar.fRS[:,:,  -1,:] = 0.5*(fluxVar.fR[:,:,  -1,:] + fluxVar.fL[:,:,0  ,:]) - 0.5*eigsRL[:,:,  -1,:]*(var.uL[:,:,  0,:] - var.uR[:,:,  -1,:])
  fluxVar.fLS[:,:,1:: ,:] = fluxVar.fRS[:,:,0:-1,:]
  fluxVar.fLS[:,:,0   ,:] = fluxVar.fRS[:,:,  -1,:]
  fluxVar.fUS[:,:,:,0:-1] = 0.5*(fluxVar.fU[:,:,:,0:-1] + fluxVar.fD[:,:,:,1::]) - 0.5*eigsUD[:,:,:,1:-1]*(var.uD[:,:,:,1::] - var.uU[:,:,:,0:-1])
  fluxVar.fUS[:,:,:,  -1] = 0.5*(fluxVar.fU[:,:,:,  -1] + fluxVar.fU_edge) -       0.5*eigsUD[:,:,:,  -1]*(var.uU_edge       - var.uU[:,:,:,  -1])
  fluxVar.fDS[:,:,:,1:: ] = 0.5*(fluxVar.fU[:,:,:,0:-1] + fluxVar.fD[:,:,:,1::]) - 0.5*eigsUD[:,:,:,1:-1]*(var.uD[:,:,:,1::] - var.uU[:,:,:,0:-1])
  fluxVar.fDS[:,:,:,0   ] = 0.5*(fluxVar.fD_edge + fluxVar.fD[:,:,:,0])          - 0.5*eigsUD[:,:,:,   0]*(var.uD[:,:,:,0  ] - var.uD_edge)

