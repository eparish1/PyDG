from pylab import *
from evtk.hl import gridToVTK

grid = load('../DGgrid.npz')
x = grid['x']
y = grid['y']
z = grid['z'] 
for i in range(0,3140,20):
  ta = np.ones((np.size(x),np.size(x),np.size(x)))
  data = load('npsol' + str(i) + '.npz')
  string = 'PVsol' + str(i)
  gridToVTK(string, x[:,None,None]*ta,y[None,:,None]*ta,z[None,None,:]*ta, pointData = {"u" : np.real(data['U'][0].transpose())})

