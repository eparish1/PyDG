import numpy as np
import numpy.polynomial.legendre
def legendreInit(zeta,order):
  L = np.zeros((order,np.size(zeta)))
  Lp = np.zeros((order,np.size(zeta)))
  Lpedge = np.zeros((order,2))
  for i in range(0,order):
    c = np.zeros(i+1)
    c[-1] = 1.
    L[i,:] = numpy.polynomial.legendre.legval(zeta,c)
    Lp[i,:] = numpy.polynomial.legendre.legval(zeta,numpy.polynomial.legendre.legder(c))
    Lpedge[i,:] = numpy.polynomial.legendre.legval(np.array([-1,1]),numpy.polynomial.legendre.legder(c))
  return L[0:order,:],Lp[0:order],Lpedge[0:order]

def gaussPoints(order,quadpoints):
  w = np.zeros((order,quadpoints))
  wp = np.zeros((order,quadpoints))
  wpedge = np.zeros((order,2))
  zeta,weights = numpy.polynomial.legendre.leggauss(quadpoints)
  w[:],wp[:],wpedge[:] = legendreInit(zeta,order)
  return w,wp,wpedge,weights,zeta
