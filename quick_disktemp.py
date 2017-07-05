import scipy as sp
import matplotlib.pyplot as plt

h = 6.6e-27
c = 2.997925e10
k = 1.4e-16

def BB(lamb,T):
    f1 = 2*h*c**2/lamb**5
    f2 = 1./(sp.exp(h*c/T/k/lamb) - 1)
    return f1*f2
    

#lamb = sp.r_[1.e2:1.e5:1000j]
lamb = 10**sp.r_[1.75:5:100j]
T = 1.e5
f = BB(lamb/1.e8,T)

plt.plot(lamb,f)

plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
l,u = plt.gca().get_xlim()
plt.gca().set_xlim([u,l])

l,u = plt.gca().get_ylim()

w = 2.9e7/T#*1.e8
char = h*c/k/T*1.e8

print 'characteristic/wien:', char/w
print '%e'%(char*T)
print 'characterist:   %e'%(char*T)
print 'flux weighted: %e'%(char*T/2.49)
print 'temp perturb:  %e'%(char*T/3.37)

plt.plot([w,w],[l,u],'k--')
plt.plot([char,char],[l,u],'r--')

plt.show()
