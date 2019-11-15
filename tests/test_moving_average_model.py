import scipy as sp
from scipy import linalg
from tools.tools import gen_lc
import matplotlib.pyplot as plt

#going to try a moving average model.  I'll iteratively estimate the
#noise at each point by starting with a rolling mean, and then feeding
#back in the prediction.

sp.random.seed(111)

def get_response(coeffs):
    #unit white noise
    n = sp.randn(len(coeffs))
    extrap = sp.r_[n, sp.zeros(len(coeffs)) ]
    for ii in range(len(coeffs) ):
        #numpy indexing makes sure that ii + len(coeffs) is correct in these cases
        extrap[ii + len(coeffs) ] = sp.sum(extrap[ii: ii + len(coeffs)]*coeffs[-1])
    return extrap

def get_coeffs(noise):
    noise_roll = []
    for m in range(M):
        noise_tmp = sp.roll(noise,-(m+1))
        noise_tmp[-(m+1):] = 0
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



t = sp.r_[0:1000:5000j]
f = gen_lc(t,30)*3
s = 0.5*sp.sin(2*sp.pi*t/55)
#don't use for now
n = sp.randn(len(f))*0.2
f = f + s + n

#params
#how much memory for running mean..
M = 20
#how far to extrapolate
N_extrap = 2000
#how many iterations?
N_iter = 10


w = sp.ones(M)
w /= w.sum()

#intial guess alarms, alarms!  because the running mean lags the
#signal, the residuals (i.e. noise estimate) are correlated.  The
#model absorbs that information when optimizing the parameters---this
#results in (1) underestimated noise-response, and the lag peresists in
#the prediction, and (2) looking at the response to white noise drives
#towards zero.  It's an iteresting problem, should try initializing with an AR, etc.

running_mean = sp.convolve(f,w,'same')
#noise is now predictors against which to optimize coefficients.
noise = f - running_mean
print('min, max, min(abs), mean, std')
print(min(noise),max(noise),min(abs(noise)),sp.mean(noise),sp.std(noise))
B = get_coeffs(noise[M::])

for ii in range(N_iter):
    pred = sp.convolve(noise, B,'same') + running_mean
    running_mean = sp.convolve(pred,w,'same')
    noise = f - running_mean 
    B = get_coeffs(noise[M::]) 

#for the final one, redo the running mean from the original time series
##***update---weirdly[??], doesn't seem to matter which running mean I use.  Most be similar or have conerged
#running_mean = sp.convolve(pred,w,'same')
#noise = f - running_mean
pred = sp.convolve(noise,B,'same') + running_mean
noise = f - sp.convolve(pred,w,'same')
pred_noise = sp.convolve(noise, B, 'same')
plt.plot(noise,'r.-')
plt.plot(sp.r_[0:len(noise)] + M/2 - 1 ,pred_noise,'b.-')

plt.figure()
plt.plot(B,'k.-')

plt.figure()
resp1 = get_response(B)
plt.plot(resp1,'.-')
plt.figure()

extrap = f[-M:]
extrap = extrap - sp.mean(extrap)
dt = t[1] - t[0]
t_extrap = sp.r_[t[-M] : t[-1] + dt*N_extrap + 0.1*dt:dt]
for i in range(N_extrap):
    extrap = sp.r_[extrap, sp.sum(extrap[-M:]*B[::-1])]

plt.plot(t,f,'r.-')
plt.plot(t[0:-M] + (M/2 -1)*(t[1] - t[0]), pred[0:-M],'b.-')
plt.plot(t,running_mean,'c.-')
#plt.plot(t[0:-M] , pred[0:-M],'b.-')
plt.plot(t_extrap, extrap+0.1,'m.-')
#plt.gca().set_xlim([t[0] - 100,t[-1]+100])
plt.show()
