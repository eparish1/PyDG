import numpy as np

def starState(uR,uL,uU,uD,uU_edge,uD_edge,uRLS,uUDS):
  uRLS[:,:,1:-1,:] = 0.5*(uR[:,:,0:-1,:] + uL[:,:,1::,:])
  uRLS[:,:,  -1,:] = 0.5*(uR[:,:,  -1,:] + uL[:,:,0  ,:])
  uRLS[:,:,   0,:] = 0.5*(uR[:,:,  -1,:] + uL[:,:,0  ,:])
  uUDS[:,:,:,1:-1] = 0.5*(uU[:,:,:,0:-1] + uD[:,:,:,1::])
  uUDS[:,:,:,  -1] = 0.5*(uU[:,:,:,  -1] + uU_edge)
  uUDS[:,:,:,   0] = 0.5*(uD[:,:,:,   0] + uD_edge)

def centralFluxU(fluxVar):
  fluxVar.uRS[:,:,0:-1,:] = 0.5*(fluxVar.uR[:,:,0:-1,:] + fluxVar.uL[:,:,1::,:])
  fluxVar.uRS[:,:,  -1,:] = 0.5*(fluxVar.uR[:,:,  -1,:] + fluxVar.uL[:,:,0  ,:])
  fluxVar.uLS[:,:,1:: ,:] = fluxVar.uRS[:,:,0:-1,:]
  fluxVar.uLS[:,:,0   ,:] = fluxVar.uRS[:,:,  -1,:]
  fluxVar.uUS[:,:,:,0:-1] = 0.5*(fluxVar.uU[:,:,:,0:-1] + fluxVar.uD[:,:,:,1::])
  fluxVar.uUS[:,:,:,  -1] = 0.5*(fluxVar.uU[:,:,:,  -1] + fluxVar.uU_edge)
  fluxVar.uDS[:,:,:,1:: ] = fluxVar.fUS[:,:,:,0:-1]
  fluxVar.uDS[:,:,:,0   ] = 0.5*(fluxVar.uD[:,:,:,   0] + fluxVar.uD_edge)


def centralFlux(fluxVar):
  fluxVar.fRS[:,:,0:-1,:] = 0.5*(fluxVar.fR[:,:,0:-1,:] + fluxVar.fL[:,:,1::,:])
  fluxVar.fRS[:,:,  -1,:] = 0.5*(fluxVar.fR[:,:,  -1,:] + fluxVar.fL[:,:,0  ,:])
  fluxVar.fLS[:,:,1:: ,:] = fluxVar.fRS[:,:,0:-1,:]
  fluxVar.fLS[:,:,0   ,:] = fluxVar.fRS[:,:,  -1,:]
  fluxVar.fUS[:,:,:,0:-1] = 0.5*(fluxVar.fU[:,:,:,0:-1] + fluxVar.fD[:,:,:,1::])
  fluxVar.fUS[:,:,:,  -1] = 0.5*(fluxVar.fU[:,:,:,  -1] + fluxVar.fU_edge)
  fluxVar.fDS[:,:,:,1:: ] = fluxVar.fUS[:,:,:,0:-1]
  fluxVar.fDS[:,:,:,0   ] = 0.5*(fluxVar.fD[:,:,:,   0] + fluxVar.fD_edge)


def rusanovFlux(main,eqns,schemes):
  eqns.starState(main.uR,main.uL,main.uU,main.uD,main.uU_edge,main.uD_edge,schemes.uRLS,schemes.uUDS)
  eqns.getEigs(schemes.uRLS,schemes.uUDS,schemes.eigsRL,schemes.eigsUD)
  main.fluxVar.fRS[:,:,0:-1,:] = 0.5*(main.fR[:,:,0:-1,:] + main.fL[:,:,1::,:]) - 0.5*schemes.eigsRL[:,:,1:-1,:]*(main.uL[:,:,1::,:] - main.uR[:,:,0:-1,:])
  main.fluxVar.fRS[:,:,  -1,:] = 0.5*(main.fR[:,:,  -1,:] + main.fL[:,:,0  ,:]) - 0.5*schemes.eigsRL[:,:,  -1,:]*(main.uL[:,:,  0,:] - main.uR[:,:,  -1,:])
  main.fluxVar.fLS[:,:,1:: ,:] = main.fRS[:,:,0:-1,:]
  main.fluxVar.fLS[:,:,0   ,:] = main.fRS[:,:,  -1,:]
  main.fluxVar.fUS[:,:,:,0:-1] = 0.5*(main.fU[:,:,:,0:-1] + main.fD[:,:,:,1::]) - 0.5*schemes.eigsUD[:,:,:,1:-1]*(main.uD[:,:,:,1::] - main.uU[:,:,:,0:-1])
  main.fluxVar.fUS[:,:,:,  -1] = 0.5*(main.fU[:,:,:,  -1] + main.fU_edge) -       0.5*scehems.eigsUD[:,:,:,  -1]*(main.uU_edge       - main.uU[:,:,:,  -1])
  main.fluxVar.fDS[:,:,:,1:: ] = 0.5*(main.fU[:,:,:,0:-1] + main.fD[:,:,:,1::]) - 0.5*schemes.eigsUD[:,:,:,1:-1]*(main.uD[:,:,:,1::] - main.uU[:,:,:,0:-1])
  main.fluxVar.fDS[:,:,:,0   ] = 0.5*(main.fD_edge + main.fD[:,:,:,0])          - 0.5*schemes.eigsUD[:,:,:,   0]*(main.uD[:,:,:,0  ] - main.uD_edge)

  fluxVar.fRS[:,:,0:-1,:] = 0.5*(fluxVar.fR[:,:,0:-1,:] + fluxVar.fL[:,:,1::,:]) - 0.5*schemes.eigsRL[:,:,1:-1,:]*(var.u.uL[:,:,1::,:] - var.u.uR[:,:,0:-1,:])
  fluxVar.fRS[:,:,  -1,:] = 0.5*(fluxVar.fR[:,:,  -1,:] + fluxVar.fL[:,:,0  ,:]) - 0.5*schemes.eigsRL[:,:,  -1,:]*(var.u.uL[:,:,  0,:] - var.u.uR[:,:,  -1,:])
  fluxVar.fLS[:,:,1:: ,:] = fluxVar.fRS[:,:,0:-1,:]
  fluxVar.fLS[:,:,0   ,:] = fluxVar.fRS[:,:,  -1,:]
  fluxVar.fUS[:,:,:,0:-1] = 0.5*(fluxVar.fU[:,:,:,0:-1] + fluxVar.fD[:,:,:,1::]) - 0.5*schemes.eigsUD[:,:,:,1:-1]*(var.u.uD[:,:,:,1::] - var.u.uU[:,:,:,0:-1])
  fluxVar.fUS[:,:,:,  -1] = 0.5*(fluxVar.fU[:,:,:,  -1] + fluxVar.fU_edge) -       0.5*scehems.eigsUD[:,:,:,  -1]*(var.u.uU_edge       - var.u.uU[:,:,:,  -1])
  fluxVar.fDS[:,:,:,1:: ] = fluxVar.fUS[:,:,:,0:-1]
  fluxVar.fDS[:,:,:,0   ] = 0.5*(fluxVar.fD[:,:,:,   0] + fluxVar.fD_edge)

