from pylab import *
axis_font = {'size':'24'}
nu100 = load('nu100.npz')
nu1600 = load('nu1600.npz')

plot(nu100['t'],nu100['E']/nu100['E'][0],label=r'$\nu = 100$')
plot(nu1600['t'],nu1600['E']/nu1600['E'][0],label=r'$\nu = 1600$')
xlabel(r'$t$',**axis_font)
ylabel(r'$||\mathcal{PL}F||_2$',**axis_font)
xscale('log')
legend(loc=1)
gcf().subplots_adjust(left=0.175)
show()

