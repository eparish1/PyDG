import numpy as np
from navier_stokes_entropy import entropy_to_conservative 
def constantForcing(main):
  if (main.fsource):
    force = np.zeros(np.shape(main.iFlux.fx))
    for i in range(0,main.nvars):
      force[i] = main.source_mag[i]#*main.a.u[i]
    main.RHS[:] += main.basis.volIntegrateGlob(main, force*main.Jdet[None,:,:,:,None,:,:,:,None] ,main.w0,main.w1,main.w2,main.w3)


def volumetricForcing(main):
  if (main.fsource):
    #U = entropy_to_conservative(main.a.u)
    force = np.zeros(np.shape(main.iFlux.fx))
    for i in range(0,main.nvars):
      force[i] = main.source_mag[i]*main.a.u[1]
    main.RHS[:] += main.basis.volIntegrateGlob(main, force*main.Jdet[None,:,:,:,None,:,:,:,None] ,main.w0,main.w1,main.w2,main.w3)

def combustionForcing(main):
  if (main.fsource):
    force = np.zeros(np.shape(main.iFlux.fx))
    sources = main.cgas_field.net_production_rates[:,0:-1]*main.cgas_field.molecular_weights[None,0:-1]
    for i in range(5,main.nvars):
      force[i] = np.reshape(sources[:,i-5],np.shape(main.a.u[0]))
      force[4] -= force[i]*main.delta_h0[i-5]
    force[4] -= main.delta_h0[-1]*np.reshape(sources[:,-1],np.shape(main.a.u[0]))
    main.RHS[:] += main.basis.volIntegrateGlob(main, force*main.Jdet[None,:,:,:,None,:,:,:,None] ,main.w0,main.w1,main.w2,main.w3)
