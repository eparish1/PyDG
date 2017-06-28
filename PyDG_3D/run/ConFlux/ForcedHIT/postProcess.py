from pylab import *
axis_font = {'size':'20'}
datatmp = load('npspec9.npz')
data = load('npspec22800.npz')
case1 = genfromtxt('case1.dat')

loglog(data['k'],0.5*np.sum(data['spectrum'],axis=1)/0.2**2,label='PyDG')
loglog(datatmp['k'],0.5*np.sum(datatmp['spectrum'],axis=1)/0.2**2,ls='--',label='PyDG')
loglog(case1[:,0],case1[:,1],'o',label='Rosales')
xlabel(r'$k$',**axis_font)
ylabel(r'$E$',**axis_font)
legend(loc=1)
xlim([1,100])
ylim([1e-9,1e-1])
savefig('combution_hitic.pdf')
show()
