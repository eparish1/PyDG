from pylab import *
axis_font = {'size':'24'}
p1 = load('p132.npz')
p3 = load('p316.npz')
p7 = load('p78.npz')
p15 = load('p154.npz')
p3b = load('p4_n16.npz')
p32 = load('p321.npz')
#p32 = load('p132.npz')

p1murm = genfromtxt('p132_Murman.dat')
p1mavrip = genfromtxt('p132_mavrip.dat')
p3murm = genfromtxt('p316_Murman.dat')
p7murm = genfromtxt('p78_Murman.dat')
p15murm = genfromtxt('p154_Murman.dat')

DNS = load('PySpec_1600_256.npz')

figure(1)
plot(p1['t'][:],p1['E'],lw=1.5,color='blue',label=r'$p=1$')
plot(p3['t'][:],p3['E'],lw=1.5,color='red',label=r'$p=3$')
plot(p3b['t'][:]/5.,p3b['E']/0.2**2,lw=1.5,color='red',ls='--',label=r'$p=3$')

plot(p7['t'][:],p7['E'],lw=1.5,color='green',label=r'$p=7$')
plot(p15['t'][:],p15['E'],lw=1.5,color='orange',label=r'$p=15$')
plot(p32['t'][:],p32['E'],lw=1.5,color='pink',label=r'$p=31$')

plot(DNS['t'][:],DNS['Energy'],lw=.75,color='black',label='DNS')
ylim([0.04,0.14])
xlim([0,10])
xlabel(r'$t$',**axis_font)
ylabel(r'$E$',**axis_font)
legend(loc=1)
savefig('TGV64_Energy.png')

figure(2)
plot(p1['t'][1::],-diff(p1['E'])*1./p1['t'][1],lw=1.5,color='blue',label=r'$p=1$')
plot(p3['t'][1::],-diff(p3['E'])*1./p3['t'][1],lw=1.5,color='red',label=r'$p=3$')
plot(p3b['t'][1::]/5.,-diff(p3b['E'])*1./p3b['t'][1]*5./0.2**2,lw=1.5,color='red',ls='--',label=r'$p=3$')

plot(p7['t'][1::],-diff(p7['E'])*1./p7['t'][1],lw=1.5,color='green',label=r'$p=7$')
plot(p15['t'][1::],-diff(p15['E'])*1./p15['t'][1],lw=1.5,color='orange',label=r'$p=15$')
plot(p32['t'][1::],-diff(p32['E'])*1./p32['t'][1],lw=1.5,color='pink',label=r'$p=31$')

#plot(p1murm[:,0],p1murm[:,1],ls='--',lw=0.75,color='blue',label='Murman')
#plot(p1mavrip[:,0],p1mavrip[:,1],ls='-.',lw=0.75,color='blue',label='Mavriplis')
plot(p3murm[:,0],p3murm[:,1],color='red',ls='--',lw=0.75)
plot(p7murm[:,0],p7murm[:,1],ls='--',lw=0.75,color='green')
plot(p15murm[:,0],p15murm[:,1],ls='--',lw=0.75,color='orange')

plot(DNS['t'][1::],-DNS['Dissipation'],lw=.75,color='black',label='DNS')
ylim([0,0.014])
xlim([0,10])
xlabel(r'$t$',**axis_font)
ylabel(r'$-\frac{dE}{dt}$',**axis_font)
legend(loc=2)
savefig('TGV64_Dissipation.png')


data = np.load('npspec500.npz')
figure(3)
loglog(p1['k'],0.5*sum(p1['spectrum'],axis=1),color='blue',lw=1.5,label=r'$p=1$')
loglog(p3['k'],0.5*sum(p3['spectrum'],axis=1),color='red',lw=1.5,label=r'$p=3$')
loglog(p3b['k'],0.5*sum(p3b['spectrum'],axis=1)/0.2**2,color='red',ls='--',lw=1.5,label=r'$p=3$')

loglog(p7['k'],0.5*sum(p7['spectrum'],axis=1),color='green',lw=1.5,label=r'$p=7$')
loglog(p15['k'],0.5*sum(p15['spectrum'],axis=1),color='orange',lw=1.5,label=r'$p=15$')
loglog(p32['k'],0.5*sum(p32['spectrum'],axis=1),color='pink',lw=1.5,label=r'$p=31$')

loglog(data['k'],0.5*(data['E'][:,0] + data['E'][:,1] + data['E'][:,2]),'-',color='black',lw=0.75,label=r'Spectral')
xlim([2,amax(data['k']) + 10])
ylim([1e-8,0.1])
xlabel(r'$k$',**axis_font)
ylabel(r'$E(k)$',**axis_font)
legend(loc=1)
savefig('TGV64_Spectrum.png')

show()
