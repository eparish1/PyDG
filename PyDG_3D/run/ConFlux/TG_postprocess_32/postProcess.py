from pylab import *
axis_font = {'size':'24'}
p1 = load('p116.npz')
p3 = load('p38.npz')
p7 = load('p4_n8_check.npz')
#p7 = load('p74.npz')
p15 = load('p152.npz')
p31 = load('p321.npz')

DNS = load('PySpec_1600_256.npz')

figure(1)
plot(p1['t'][:],p1['E'],lw=1.5,color='blue',label=r'$p=1$')
plot(p3['t'][:],p3['E'],lw=1.5,color='red',label=r'$p=3$')
plot(p7['t'][:]/5.,p7['E']/0.2**2,lw=1.5,color='orange',label=r'$p=7$')
plot(p15['t'][:],p15['E'],lw=1.5,color='green',label=r'$p=15$')
plot(p31['t'][:],p31['E'],lw=1.5,color='purple',label=r'$p=31$')

plot(DNS['t'][:],DNS['Energy'],lw=.75,color='black',label='DNS')
ylim([0.04,0.14])
xlim([0,10])
xlabel(r'$t$',**axis_font)
ylabel(r'$E$',**axis_font)
legend(loc=1)
savefig('TGV32_Energy.png')

figure(2)
plot(p1['t'][1::],-diff(p1['E'])*1./p1['t'][1],lw=1.5,color='blue',label=r'$p=1$')
plot(p3['t'][1::],-diff(p3['E'])*1./p3['t'][1],lw=1.5,color='red',label=r'$p=3$')
plot(p7['t'][1::]/5.,-diff(p7['E'])*1./(p7['t'][1]/5.)/0.2**2,lw=1.5,color='orange',label=r'$p=7$')
plot(p15['t'][1::],-diff(p15['E'])*1./p15['t'][1],lw=1.5,color='green',label=r'$p=15$')
plot(p31['t'][1::],-diff(p31['E'])*1./p31['t'][1],lw=1.5,color='purple',label=r'$p=31$')

plot(DNS['t'][1::],-DNS['Dissipation'],lw=.75,color='black',label='DNS')
ylim([0,0.014])
xlim([0,10])
xlabel(r'$t$',**axis_font)
ylabel(r'$-\frac{dE}{dt}$',**axis_font)
legend(loc=2)
savefig('TGV32_Dissipation.png')


data = np.load('npspec500.npz')
figure(3)
loglog(p1['k'],0.5*sum(p1['spectrum'],axis=1),color='blue',lw=1.5,label=r'$p=1$')
loglog(p3['k'],0.5*sum(p3['spectrum'],axis=1),color='red',lw=1.5,label=r'$p=3$')
loglog(p7['k'],0.5*sum(p7['spectrum'],axis=1)/0.2**2,color='orange',lw=1.5,label=r'$p=7$')
loglog(p15['k'],0.5*sum(p15['spectrum'],axis=1),color='green',lw=1.5,label=r'$p=15$')
loglog(p31['k'],0.5*sum(p31['spectrum'],axis=1),color='purple',lw=1.5,label=r'$p=31$')

loglog(data['k'],0.5*(data['E'][:,0] + data['E'][:,1] + data['E'][:,2]),'-',color='black',lw=0.75,label=r'Spectral')
xlim([2,amax(data['k']) + 10])
ylim([1e-8,0.1])
xlabel(r'$k$',**axis_font)
ylabel(r'$E(k)$',**axis_font)
legend(loc=1)
savefig('TGV32_Spectrum.png')

show()
