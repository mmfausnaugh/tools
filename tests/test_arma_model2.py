import scipy as sp
from scipy import linalg
from tools.tools import gen_lc
import matplotlib.pyplot as plt

#now, do an ARMA.  Star with autoregressive, to estimate the noise part, then iterate a few times.  Later, I'll try inverting the matrix all at once
sp.random.seed(11111)


def get_coeffs(acf, noise, p, q):
    acf_roll = []
    for ii in range(p):
        acf_tmp = sp.roll(acf,-(ii+1))
        acf_tmp[-(ii+1):] = 0
        acf_roll.append(acf_tmp)

    acf_roll= sp.array(acf_roll)

    noise_roll = []
    for ii in range(q):
        noise_tmp = sp.roll(noise,-(ii+1))
        noise_tmp[-(ii+1):] = 0
        noise_roll.append(noise_tmp)

    noise_roll= sp.array(noise_roll)

    #fill in the design matrices
    C = []
    for ii in range(len(acf_roll)):
        C.append(sp.sum(acf*acf_roll[ii]))
    for ii in range(len(noise_roll)):
        C.append(sp.sum(acf*noise_roll[ii]))
    C = sp.array(C)

    A = []
    roll = sp.r_[acf_roll,noise_roll]
    for ii in range(len(roll)):
        row = []
        for jj in range(len(roll)):
            row.append(sp.sum(roll[ii]*roll[jj]))
        A.append(row)
    A = sp.array(A)

    #print('A shape, C shape',A.shape, C.shape)
                    
    B = linalg.solve(A,C)
    return B


t = sp.r_[0:1000:5000j]
f = gen_lc(t,30)*3
s = 0.0*sp.sin(2*sp.pi*t/55)
#don't use for now
n = sp.randn(len(f))*0.2
f = f + s + n

#params
#how much memory for running mean..
P = 200
Q = 10
#how far to extrapolate
N_extrap = 2000
#how many iterations?
N_iter = 100

#to initalize, do a running mean
noise = f - sp.convolve(f, sp.ones(Q)/Q,'same')
model_coeffs = get_coeffs(f, noise, P, Q)

for ii in range(N_iter):
    AR_pred = sp.convolve(f,model_coeffs[0:P], 'same')
    MA_pred = sp.convolve(noise,model_coeffs[P::], 'same')

    #P index should equal -Q index to combine, as long as P is > Q,
    #the lag in P makes sure the memory of noise Q lines up with
    #predictions from P.  condition of P>Q vs Q>P?  make this general
    #(i.e., max([P,Q])
    pred = AR_pred[P::] + MA_pred[P::]
    noise = f[P:] - pred
    noise = sp.r_[sp.zeros(P), noise]

    model_coeffs = get_coeffs(f, noise, P,Q) 

#finished with loop, get final prediction
AR_pred = sp.convolve(f,model_coeffs[0:P], 'same')
#is this actually the noise at this point, or do I need one more
#iteration with these params?  Hopefully it is converged, i.e., noise
#and model params did not change on the last 2 iterations (or so)
MA_pred = sp.convolve(noise, model_coeffs[P::], 'same')
pred = AR_pred[P::] + MA_pred[P::]

##
##M = max([P,Q])
##extrap = f[-M:]
##extrap_noise = extrap - sp.convolve(extrap,acf_p,'same')
##
##dt = t[1] - t[0]
##t_extrap = sp.r_[t[-M] : t[-1] + dt*N_extrap + 0.1*dt:dt]
##
##for i in range(N_extrap):
##    AR_extrap = sp.sum(extrap[-P:]*acf_p[::-1])
##    MA_extrap = sp.sum(extrap_noise[-Q:]*noise_q[::-1])
##    extrap = sp.r_[extrap,  AR_extrap ]
##    extrap_noise = sp.r_[extrap_noise, MA_extrap]
##
##extrap += extrap_noise

plt.plot(t,f,'r.-')
plt.plot(t[P:] , pred,'b.-')
#plt.plot(t[0:-M] , pred[0:-M],'b.-')
#plt.plot(t_extrap, extrap+0.1,'m.-')
#plt.gca().set_xlim([t[0] - 100,t[-1]+100])
plt.show()
