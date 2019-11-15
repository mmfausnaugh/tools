import scipy as sp
from scipy import linalg
from tools.tools import gen_lc
import matplotlib.pyplot as plt

#now, do an ARMA.  Star with autoregressive, to estimate the noise part, then iterate a few times.  Later, I'll try inverting the matrix all at once
sp.random.seed(11111)


def get_noise_coeffs(noise,q):
    #q is an integer to define how long the integer is
    noise_roll = []
    for ii in range(q):
        noise_tmp = sp.roll(noise,-(ii+1))
        noise_tmp[-(ii+1):] = 0
        noise_roll.append(noise_tmp)

    noise_roll= sp.array(noise_roll)

    #fill in the design matrices
    C = []
    for ii in range(len(noise_roll)):
        C.append(sp.sum(noise*noise_roll[ii]))
    C = sp.array(C)

    A = []
    for ii in range(len(noise_roll)):
        row = []
        for jj in range(len(noise_roll)):
            row.append(sp.sum(noise_roll[ii]*noise_roll[jj]))
        A.append(row)
    A = sp.array(A)

    #print('A shape, C shape',A.shape, C.shape)
                    
    B = linalg.solve(A,C)
    return B

def get_acf_coeffs(acf,p):
    #p is an iteger to define how long the memory is
    acf_roll = []
    for ii in range(p):
        acf_tmp = sp.roll(acf,-(ii+1))
        acf_tmp[-(ii+1):] = 0
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

    #print('A shape, C shape',A.shape, C.shape)
                    
    B = linalg.solve(A,C)
    return B



t = sp.r_[0:1000:5000j]
f = gen_lc(t,30)*3
s = 0.5*sp.sin(2*sp.pi*t/55)
#don't use for now
n = sp.randn(len(f))*0.0
f = f + s + n

#params
#how much memory for running mean..
P = 200
Q = 10
#how far to extrapolate
N_extrap = 2000
#how many iterations?
N_iter = 10

acf = sp.correlate(f,f,'same')
acf_p = get_acf_coeffs(acf,P)
AR_pred = sp.convolve(f,acf_p,'save')[P:]
noise = f[P:] - AR_pred
noise_q = get_noise_coeffs(noise,Q)

for ii in range(N_iter):    
    pred = sp.convolve(noise, noise_q,'same') + AR_pred
    noise = f[P:] - pred
    noise_q = get_noise_coeffs(noise, Q) 

#finished with loop, get final noise estimates
pred = sp.convolve(noise,noise_q,'same') + AR_pred

M = max([P,Q])
extrap = f[-M:]
extrap_noise = extrap - sp.convolve(extrap,acf_p,'same')

dt = t[1] - t[0]
t_extrap = sp.r_[t[-M] : t[-1] + dt*N_extrap + 0.1*dt:dt]

for i in range(N_extrap):
    AR_extrap = sp.sum(extrap[-P:]*acf_p[::-1])
    MA_extrap = sp.sum(extrap_noise[-Q:]*noise_q[::-1])
    extrap = sp.r_[extrap,  AR_extrap ]
    extrap_noise = sp.r_[extrap_noise, MA_extrap]

extrap += extrap_noise
plt.plot(t,f,'r.-')
plt.plot(t[P:] , pred,'b.-')
#plt.plot(t[0:-M] , pred[0:-M],'b.-')
plt.plot(t_extrap, extrap+0.1,'m.-')
#plt.gca().set_xlim([t[0] - 100,t[-1]+100])
plt.show()
