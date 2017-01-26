import numpy as np
from DG_functions import getRHS
def tauModel(main,MZ,eqns,schemes):
    eps = 1.e-5
    MZ.a.a[:,0:main.order,0:main.order,:,:] = main.a.a[:]
    getRHS(MZ,eqns,schemes)
    #residual = main.RHS[:] - MZ.RHS[:,0:main.order,0:main.order,:,:] ##get residual while here
    RHS1 = np.zeros(np.shape(MZ.RHS))
    RHS1[:] = MZ.RHS[:]
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order,0:main.order,:,:] = main.a.a[:]
    MZ.a.a[:] = MZ.a.a[:] + eps*MZ.RHS[:]
    getRHS(MZ,eqns,schemes)
    RHS2 = np.zeros(np.shape(MZ.RHS))
    RHS2[:] = MZ.RHS[:]
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order,0:main.order,:,:] = main.a.a[:]
    MZ.a.a[:,0:main.order,0:main.order,:,:] += eps*MZ.RHS[:,0:main.order,0:main.order,:,:]
    getRHS(MZ,eqns,schemes)
    RHS3 = np.zeros(np.shape(MZ.RHS))
    RHS3[:] = MZ.RHS[:]
    PLQLU = (RHS2[:,0:main.order,0:main.order,:,:] - RHS3[:,0:main.order,0:main.order,:,:])/eps
    return MZ.tau*PLQLU
