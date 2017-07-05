### to make a dynammic loading library (dll), i took keith's code and
### compliled with: 
##  ifort -shared -fpic -o ccdvar.so ccdvar.for

import scipy as sp
import matplotlib.pyplot as plt
from ctypes import *
from astropy.io import ascii
from astropy.table import Table
from copy import deepcopy
#tests
#fort = CDLL('./add.so')
#add = fort.add_ii_
#x = c_int(3)
#y = c_int(4)
#
#add(byref(x),byref(y))
#print 'x',x
#print 'y',y
#add(byref(x),byref(y))
#add(byref(x),byref(y))
#print 'x',x
#print 'y',y

def sigmaclip(y):

    res = y - y.mean()

    m = abs(res) > 10*y.std()
    yuse = deepcopy(y)
    while (m == True).any():
        yuse = yuse[-m]

        res = yuse - yuse.mean()
        m = abs(res) > 10*yuse.std()

    print len(y[sp.in1d(y,yuse)])
    return sp.in1d(y,yuse)

        

fort = CDLL('./ccdvar.so')
ccdvar = fort.ccdvar_

starlist = sp.genfromtxt('starlist',dtype=str)
print starlist.size
apphot = Table.read('ref.phot',format='daophot')
print apphot['FLUX'].size
mask = apphot['FLUX'] > 1.e-20

fuse = sp.array(apphot['FLUX'])[mask]
starlist=starlist[mask]

i = sp.argsort(fuse)
fuse = fuse[i][::-1]
starlist = starlist[i][::-1]

#flx = (c_float(x) for x in fuse)
#flx = [x for x in fuse]
#print flx
#flx = CF(flx)

x1 = sp.genfromtxt(starlist[0],usecols=(0))

for i in range(x1.size):

    dat2,sig2 = [],[]
    fuse2 = deepcopy(fuse)
    for j,star in enumerate(starlist):
        x,y,z = sp.genfromtxt(star,unpack=1,usecols=(0,1,2))
        dat2 = sp.r_[dat2,y[i]]
        sig2 = sp.r_[sig2,z[i]]

    m = sigmaclip(dat2)
    fuse2 = fuse2[m]
    dat2 = dat2[m]
    sig2 = sig2[m]

    sp.savetxt('mcg0811.g_image%i.dat'%i,sp.c_[fuse2,dat2,sig2],fmt='% 10.4f % 5.4f % 5.4f')
    plt.plot(-2.5*sp.log10(fuse2),dat2/sig2,'ko')
    plt.show()
#initialize for fortran call

    dat = (c_float*fuse2.size)()
    sig = (c_float*fuse2.size)()
    flx = (c_float*fuse2.size)()
    for i in range(fuse2.size):
        flx[i] = fuse2[i]
        dat[i] = dat2[i]
        sig[i] = sig2[i]

    sig0 = c_double()
    sig1 = c_double()
    sig2 = c_double()


    print 'running ccdvar'

    ccdvar(byref(c_int(fuse2.size)),
           flx,
           dat, 
           sig,
           byref(sig0),
           byref(sig1),
           byref(sig2),
           byref(c_int(10000)))

    print sig0,sig1,sig2
#    raw_input()
