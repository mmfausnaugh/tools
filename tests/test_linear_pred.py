import scipy as sp
from scipy import linalg
from tools.tools import gen_lc
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AR



sp.random.seed(1111)

def get_response(coeffs):
    extrap = sp.zeros(len(coeffs))
    extrap[-1] = 1.0
    for ii in range(len(coeffs)):
        extrap[len(coeffs) -1 - ii] = sp.sum(extrap*coeffs[::-1])
        if ii  <5:
            print(extrap[-5:])

    return extrap

#this uses the y values to predict/extrapolate, and so is an autoregressive model

t = sp.r_[0:1000:5000j]
f = gen_lc(t,30)*3
s = 0.5*sp.sin(2*sp.pi*t/15)
#don't use for now
n = sp.randn(len(f))*0.
f = f + s + n

acf = sp.correlate(f,f,'same')

#params
#how much memory..
M = 630
#how far to extrapolate
N_extrap = 10000

acf_roll = []
for m in range(M):
    acf_tmp = sp.roll(acf,-(m+1))
    acf_tmp[-(m+1):] = 0
    acf_roll.append(acf_tmp)

acf_roll= sp.array(acf_roll)

#fill in the design matrices
C = []
for ii in range(len(acf_roll)):
    C.append(sp.sum(acf*acf_roll[ii]))
C = sp.array(C)

A = []
for ii in range(len(acf_roll)):
    row = []
    for jj in range(len(acf_roll)):
        row.append(sp.sum(acf_roll[ii]*acf_roll[jj]))
    A.append(row)
A = sp.array(A)

print('A shape, C shape',A.shape, C.shape)
                    
B = linalg.solve(A,C)

pred = sp.convolve(f,B,'same')
extrap = f[-M:]
dt = t[1] - t[0]
t_extrap = sp.r_[t[-M] : t[-1] + dt*N_extrap + 0.1*dt:dt]
for i in range(N_extrap):
    extrap = sp.r_[extrap, sp.sum(extrap[-M:]*B[::-1])]

#try using the statsmodels thing
mod = AR(f)
result = mod.fit(maxlag = M,trend='nc')
statsmod_pred = result.predict(len(f) - M, len(f) + N_extrap - 1)

print(abs(result.roots))
resp1 = get_response(B)
print(B[0:5])
print(result.params[0:5])
resp2 = get_response(result.params)
plt.plot(resp1[::-1],'.-',label='MMF')
plt.plot(resp2[::-1],'.-',label='statsmodel')
plt.figure()
#diff = result.params - B
#print(max(diff))
#print(min(diff))
#print(min(abs(diff)))
#plt.plot(result.params,B,'k.')
#l,h = plt.gca().get_xlim()
#plt.plot([l,h],[l,h],'k')
#plt.gca().set_xlabel('statmodel')
#plt.gca().set_ylabel('MMF')
#
#plt.figure()
plt.plot(t,f,'r.-')
plt.plot(t[0:-M] + (M/2 -1)*(t[1] - t[0]), pred[0:-M],'b.-')
#plt.plot(t[0:-M] , pred[0:-M],'b.-')
plt.plot(t_extrap, extrap+0.1,'m.-')
plt.plot(t_extrap, statsmod_pred,'.-',color='purple')

plt.plot(t[M:] - t[1] + t[0],result.fittedvalues, 'c.-')
plt.show()
