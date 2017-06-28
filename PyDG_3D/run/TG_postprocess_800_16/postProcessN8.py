from pylab import *
axis_font = {'size':'24'}
p1 = load('p1.npz')
p1_MZ = load('p1_MZ.npz')
p3F = load('p3F_N8.npz')

DNS = load('PySpec_400_128.npz')

figure(1)
plot(p1['t'][:],p1['E'],lw=1.5,color='blue',label=r'$p=1$')
plot(p1_MZ['t'][:],p1_MZ['E'],lw=1.5,color='blue',ls='--',label=r'$p=1 MZ$')
plot(p3F['t'][:],p3F['E'],lw=1.5,color='green',ls='--',label=r'$p=3$ Filtered')

plot(DNS['t'][:],DNS['Energy'],lw=.75,color='black',label='DNS')
ylim([0.04,0.14])
xlim([0,10])
xlabel(r'$t$',**axis_font)
ylabel(r'$E$',**axis_font)
legend(loc=1)
savefig('TGV32_Energy.png')

figure(2)
plot(p1['t'][1::],-diff(p1['E'])*1./p1['t'][1],lw=1.5,color='blue',label=r'$p=3$')
plot(p1_MZ['t'][1::],-diff(p1_MZ['E'])*1./p1_MZ['t'][1],lw=1.5,color='blue',ls='--',label=r'$p=3MZ$')
plot(p3F['t'][1::],-diff(p3F['E'])*1./p3F['t'][1],lw=1.5,color='green',ls='--',label=r'$p=3 Filtered$')

plot(DNS['t'][1::],-DNS['Dissipation_resolved'],lw=.75,color='black',label='DNS')
ylim([0,0.02])
xlim([0,10])
xlabel(r'$t$',**axis_font)
ylabel(r'$-\frac{dE}{dt}$',**axis_font)
legend(loc=1)
savefig('TGV32_Dissipation.png')


data = np.load('npspec500.npz')
figure(3)
loglog(p1['k'],0.5*sum(p1['spectrum'],axis=1),color='blue',lw=1.5,label=r'$p=1$')
loglog(p1_MZ['k'],0.5*sum(p1_MZ['spectrum'],axis=1),color='blue',ls='--',lw=1.5,label=r'$p=1 MZ$')

loglog(p3F['k'],0.5*sum(p3F['spectrum'],axis=1),color='green',ls='--',lw=1.5,label=r'$p=3$ Filtered')

loglog(data['k'],0.5*(data['E'][:,0] + data['E'][:,1] + data['E'][:,2]),'-',color='black',lw=0.75,label=r'Spectral')
xlim([2,amax(data['k']) + 10])
ylim([1e-8,0.1])
xlabel(r'$k$',**axis_font)
ylabel(r'$E(k)$',**axis_font)
legend(loc=1)
savefig('TGV32_Spectrum.png')

show()
