from pylab import *
axis_font = {'size':'24'}
p7 = load('p7.npz')
p7_MZ = load('p7_MZ.npz')
p15F = load('p15F_N2.npz')

DNS = load('PySpec_400_128.npz')

figure(1)
plot(p7['t'][:],p7['E'],lw=1.5,color='blue',label=r'$p=7$')
plot(p7_MZ['t'][:],p7_MZ['E'],lw=1.5,color='blue',ls='--',label=r'$p=7 MZ$')
plot(p15F['t'][:],p15F['E'],lw=1.5,color='green',ls='--',label=r'$p=15$ Filtered')

plot(DNS['t'][:],DNS['Energy'],lw=.75,color='black',label='DNS')
ylim([0.04,0.14])
xlim([0,10])
xlabel(r'$t$',**axis_font)
ylabel(r'$E$',**axis_font)
legend(loc=1)
savefig('TGV32_Energy.png')

figure(2)
plot(p7['t'][1::],-diff(p7['E'])*1./p7['t'][1],lw=1.5,color='blue',label=r'$p=7$')
plot(p7_MZ['t'][1::],-diff(p7_MZ['E'])*1./p7_MZ['t'][1],lw=1.5,color='blue',ls='--',label=r'$p=7MZ$')
plot(p15F['t'][1::],-diff(p15F['E'])*1./p15F['t'][1],lw=1.5,color='green',ls='--',label=r'$p=15 Filtered$')

plot(DNS['t'][1::],-DNS['Dissipation_resolved'],lw=.75,color='black',label='DNS')
ylim([0,0.02])
xlim([0,10])
xlabel(r'$t$',**axis_font)
ylabel(r'$-\frac{dE}{dt}$',**axis_font)
legend(loc=2)
savefig('TGV32_Dissipation.png')


data = np.load('npspec500.npz')
figure(3)
loglog(p7['k'],0.5*sum(p7['spectrum'],axis=1),color='blue',lw=1.5,label=r'$p=7$')
loglog(p7_MZ['k'],0.5*sum(p7_MZ['spectrum'],axis=1),color='blue',ls='--',lw=1.5,label=r'$p=7 MZ$')

loglog(p15F['k'],0.5*sum(p15F['spectrum'],axis=1),color='green',ls='--',lw=1.5,label=r'$p=15$ Filtered')

loglog(data['k'],0.5*(data['E'][:,0] + data['E'][:,1] + data['E'][:,2]),'-',color='black',lw=0.75,label=r'Spectral')
xlim([2,amax(data['k']) + 10])
ylim([1e-8,0.1])
xlabel(r'$k$',**axis_font)
ylabel(r'$E(k)$',**axis_font)
legend(loc=1)
savefig('TGV32_Spectrum.png')

show()
