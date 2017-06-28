from pylab import *
axis_font = {'size':'24'}
data = load('stats.npz')

integ = trapz(data['PLFsol'],data['t'])
PLFsol0 = data['PLFsol'][:,:,:,:,:,:,0]

corr = np.corrcoef(integ.flatten(),PLFsol0.flatten())

scale = norm(PLFsol0)/norm(integ)

Nel = 4.
order = 16.
ainf = 5.
taus = 2.*np.pi/(Nel*order**2*ainf)
print('Correlation = ' + str(corr[0,1]))
print('scale = ' + str(1./scale))
print('tau* = ' + str(taus)) 

plot(data['t'],data['PLF']/data['PLF'][0],lw=1.5,color='black')
xlabel(r'$t$',**axis_font)
ylabel(r'$||\mathcal{PL}F||_2$',**axis_font)
xlim([1.e-5,5.])
xscale('log')

figure(2)
plot(data['t'],data['PLFsol'][2,2,2,0,0,0]/data['PLFsol'][2,2,2,0,0,0,0],lw=1.5,color='black')
xlabel(r'$t$',**axis_font)
ylabel(r'$\mathcal{PL}F$',**axis_font)
xlim([1.e-5,5.])
xscale('log')

figure(3)
plot(integ.flatten())
plot(PLFsol0.flatten()/scale)
ylabel(r'$\int \mathcal{PL}F$',**axis_font)

figure(4)
plot(abs(integ.flatten()))
plot(abs(PLFsol0.flatten()/scale - integ.flatten()),color='black')
#plot(abs(PLFsol0.flatten()/scale),color='blue')
ylabel(r'$\int \mathcal{PL}F$',**axis_font)
#xlim([1.e-3,5.])
#xscale('log')

show()
