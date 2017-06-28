import numpy as np
import numpy


def computeSpectrum(UG):
  N1,N2,N3 = np.shape(UG[0])
  L1,L2,L3 = 2.*np.pi,2.*np.pi,2.*np.pi
  u = UG[1]/UG[0]
  v = UG[2]/UG[0]
  w = UG[3]/UG[0]

  uhat =  np.fft.rfftn(u) / np.sqrt(N1*N2*N3)
  vhat =  np.fft.rfftn(v) / np.sqrt(N1*N2*N3)
  what =  np.fft.rfftn(w) / np.sqrt(N1*N2*N3)

  print(np.shape(uhat))
  k1 = np.fft.fftshift( np.linspace(-N1/2,N1/2-1,N1) ) * 2.*np.pi / L1
  k2 = np.fft.fftshift( np.linspace(-N2/2,N2/2-1,N2) ) * 2.*np.pi / L2
  k3 = np.linspace( 0,N3/2,N3/2+1 ) * 2. * np.pi / L3

  k2,k1,k3 = np.meshgrid(k2,k1,k3)

  ksqr = (k1*L1/(2.*np.pi))**2 + (k2*L2/(2.*np.pi))**2 + (k3*L3/(2.*np.pi))**2
  k_m, indices1 = np.unique((np.rint(np.sqrt(ksqr[:,:,1:N3/2].flatten()))), return_inverse=True)
  k_0, indices2 = np.unique((np.rint(np.sqrt(ksqr[:,:,0].flatten()))), return_inverse=True)
  #k_m, indices1 = np.unique(np.rint(np.sqrt(ksqr[:,:,:].flatten())), return_inverse=True)
  kmax = np.int(np.round(np.amax(k_m)))
  kdata = np.linspace(0,kmax,kmax+1)
  spectrum = np.zeros((kmax+1,3),dtype='complex')
  spectrum2 = np.zeros((kmax+1,3),dtype='complex')
  np.add.at( spectrum[:,0],np.int8(k_m[indices1]),2*uhat[:,:,1:N3/2].flatten()*np.conj(uhat[:,:,1:N3/2].flatten()))
  np.add.at( spectrum[:,0],np.int8(k_0[indices2]),  uhat[:,:,0].flatten()*          np.conj(uhat[:,:,0].flatten()))
  np.add.at( spectrum[:,1],np.int8(k_m[indices1]),2*vhat[:,:,1:N3/2].flatten()*np.conj(vhat[:,:,1:N3/2].flatten()))
  np.add.at( spectrum[:,1],np.int8(k_0[indices2]),vhat[:,:,0].flatten()*np.conj(vhat[:,:,0].flatten()))
  np.add.at( spectrum[:,2],np.int8(k_m[indices1]),2*what[:,:,1:N3/2].flatten()*np.conj(what[:,:,1:N3/2].flatten()))
  np.add.at( spectrum[:,2],np.int8(k_0[indices2]),what[:,:,0].flatten()*np.conj(what[:,:,0].flatten()))
  spectrum = spectrum/(N1*N2*N3)
  return kdata,spectrum

def reconstructU(w,a):
  tmp =  np.einsum('rn,zpqrijk->zpqnijk',w,a)
  tmp = np.einsum('qm,zpqnijk->zpmnijk',w,tmp)
  return np.einsum('pl,zpmnijk->zlmnijk',w,tmp)

def getGlobU(u):
  nvars,quadpoints,quadpoints,quadpoints,Nelx,Nely,Nelz = np.shape(u)
  uG = np.zeros((nvars,quadpoints*Nelx,quadpoints*Nely,quadpoints*Nelz))
  for i in range(0,Nelx):
    for j in range(0,Nely):
      for k in range(0,Nelz):
        for m in range(0,nvars):
          uG[m,i*quadpoints:(i+1)*quadpoints,j*quadpoints:(j+1)*quadpoints,k*quadpoints:(k+1)*quadpoints] = u[m,:,:,:,i,j,k]
  return uG


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

def uniformPoints(order,quadpoints):
  w = np.zeros((order,quadpoints))
  wp = np.zeros((order,quadpoints))
  wpedge = np.zeros((order,2))
  zeta,weights = numpy.polynomial.legendre.leggauss(quadpoints)
  dzeta = 2./quadpoints
  zeta = np.linspace(-1+dzeta/2,1-dzeta/2,quadpoints)
  w[:],wp[:],wpedge[:] = legendreInit(zeta,order)
  return w,wp,wpedge,weights,zeta


def gaussPoints(order,quadpoints):
  w = np.zeros((order,quadpoints))
  wp = np.zeros((order,quadpoints))
  wpedge = np.zeros((order,2))
  zeta,weights = numpy.polynomial.legendre.leggauss(quadpoints)
  w[:],wp[:],wpedge[:] = legendreInit(zeta,order)
  return w,wp,wpedge,weights,zeta


def volIntegrateScalar(weights,f):
  return  np.einsum('pqrijk->ijk',weights[:,None,None,None,None,None]*weights[None,:,None,None,None,None]*weights[None,None,:,None,None,None]*f[:,:,:,:,:,:])

def volIntegrate(weights,f):
  return  np.einsum('zpqrijk->zijk',weights[None,:,None,None,None,None,None]*weights[None,None,:,None,None,None,None]*weights[None,None,None,:,None,None,None]*f[:,:,:,:,:])


data = np.load('npsol0.npz')
a = data['a']
order = np.shape(a)[1]
quadpoints = order*2
Nelx = np.shape(a)[-3]
Nely = np.shape(a)[-2]
Nelz = np.shape(a)[-1]
dx = 2.*np.pi/Nelx
dy = 2.*np.pi/Nely
dz = 2.*np.pi/Nelz

w,wp,wpedge,weights,zeta = gaussPoints(order,quadpoints)
w2,wp2,wpedge2,weights2,zeta2 = uniformPoints(order,quadpoints)

U = reconstructU(w,a)
U2 = reconstructU(w2,a)

usqr = 1./U[0]**2*(U[1]*U[1] + U[2]*U[2] + U[3]*U[3])
E0 = 1./(2.*np.pi)**3*np.sum(volIntegrateScalar(weights,0.5*U[0]*usqr))*dx*dy*dz/8.
print('E0 = ' + str(E0))

Earray = np.zeros(0)
tarray = np.zeros(0)
for i in range(0,1250,10):
  print(i)
  data = np.load('npsol' + str(i) + '.npz')
  a = data['a']
  U = reconstructU(w,a)
  usqr = 1./U[0]**2*(U[1]*U[1] + U[2]*U[2] + U[3]*U[3])
  E0 = 1./(2.*np.pi)**3*np.sum(volIntegrateScalar(weights,0.5*U[0]*usqr))*dx*dy*dz/8.
  Earray = np.append(Earray,E0)
  tarray = np.append(tarray,data['t'])

U2 = getGlobU(reconstructU(w2,a))
kdata,spectrum = computeSpectrum(U2)
np.savez('stats',E=Earray,t=tarray,k=kdata,spectrum=spectrum)
