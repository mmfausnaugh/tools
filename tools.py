from __future__ import print_function

import scipy as sp
from scipy import optimize,linalg
from scipy.interpolate import interp1d,interp2d
import matplotlib.pyplot as plt
import scipy.fftpack as fft

#import h5py

#######Python stuff##############

class ProtectAttr(object):
    """
    This is a descriptor that imitates the @property decorator.  The
    advantage is that you do not need to individually define the
    properties.  At the class level, simply define:

    x = ProtectAttr('_x')
    y = ProtectAttr('_y')

    and at the instance level define:
    def __init__(self,input_x,input_y):
        self._x = input_x
        self._y = input_y

    etc.  For more specialized setters, use @property.  If you want to
    resuse the setter, define a new, module-specific descriptor.

    Somewhere on the internet, a person suggested using WeakKeyDict
    instead of getattr/setattr, in order to prevent memory leaks.
    Probably won't be an issue, but maybe something to test if there
    are problems.

    """
    #print cls
    def __init__(self,name):
        self.name = name
        
    def __get__(self,instance,cls_owner):
        #instance and cls_owner are automatically supplied when this
        #is called.  cls_owner is the owning class, and is a dummy
        #variable for these purposes.
        return getattr(instance,self.name)
    
    def __set__(self,instance,value):
#        print 'invoking __set__'
        setattr(instance,self.name,value)


#######Fitting curves##############
def linfit(x,y,ey):
    """
    This function minimizes chi^2 for a least squares fit to a simple
    linear model.  The errors in x are assumed to be much smaller than
    the errors in y.  It automatically fills in the Fisher matrix, and
    solves the simultaneous equations.
    
    The function returns p, and covar.  p[0] is the zeropoint
    p[1] is the slope.
    
    The covariance matrix is the inverted Fisher matrix.  Note that
    you must take the square root to get the errors, in all cases.
    """
    C = sp.array([sp.sum(y/ey/ey),sp.sum(y*x/ey/ey)])
    A = sp.array([
            [sp.sum(1./ey/ey), sp.sum(x/ey/ey)],
            [sp.sum(x/ey/ey),  sp.sum(x*x/ey/ey)]
            ] )
    p     = linalg.solve(A,C)
    covar = linalg.inv(A)
    return p,covar

def linfit_xerr(x,y,ex,ey):
    """
    This function minimizes chi^2 for a least squares fit to a simple
    linear model.  The errors in x may be arbitarily large.  The
    module is scipy.optimize.leastsq(), which uses a
    Levenberg-Marquardt algorithm (gradient chasing and Newton's
    method).

    This method does something to estimate the covariance matrix, in
    the sense of the second derivative of chi^2 with respect to all
    parameters.  linfit_xerr was checked with small errors on x, and
    matches the covariance matrix returned for linfit()

    This function returns the parameters and the covariance matrix.
    Note that leastsq returns the jacobian, a message, and a flag as
    well.
    """
    merit = lambda p1,x1,xe1,y1,ye1: (y1 - p1[0] - p1[1]*x1)/sp.sqrt((ye1**2 + (p1[1]*xe1)**2))
    p,covar,dum1,dum2,dum3 = optimize.leastsq(merit,x0=[0,0],args=(x,ex,y,ey),xtol = 1e-8,full_output=1)
    return p,covar

def fitfunc(func,pin,x,y,ey):
    """
    Non-linear least square fitter.  Assumes no errors in x.  Utilizes
    scipy.optimize.leastsq.  Uses a Levenberg-Marquardt algorithm
    (gradient chasing and Newton's method).  The user must provide an
    object that is the functional form to be fit, as well as initial
    guesses for the parameters.

    This function returns the parameters and the covariance matrix.
    Note that leastsq returns the jacobian, a message, and a flag as
    well.
    
    """
    merit = lambda params,func,x,y,ey:  (y - func(x,params))/ey
    p,covar,dum1,dum2,dum3 = optimize.leastsq(merit,pin,args=(func,x,y,ey),xtol = 1e-8,full_output=1)
    return p,covar

def fitfunc_bound(func,pin,x,y,ey,pbound):
    """
    Non-linear least square fitter, with an option of contraints

    ***********
    
    """
    merit = lambda params,func,x,y,ey:  sp.sum( (y - func(x,params))**2/ey**2 )
    #return with weird object....
    out = optimize.minimize(merit,pin,args=(func,x,y,ey),tol = 1.e-8,method='L-BFGS-B',bounds = pbound,options={'disp':True, 'maxiter':1000})
#    out = optimize.fmin_bfgs(merit,pin,args=(func,x,y,ey),gtol = 1.e-8)
#    out = optimize.minimize(merit,pin,args=(func,x,y,ey),tol = 1.e-8,method='SLSQP',bounds = pbound)
#    print out
#    print 'success:',out.success
#    errors = sp.sqrt(ey**2 * out.jac**2)
#    print out.keys()
#    print 'iterations:',out.nit,out.nfev#, out.njev, out.nhev
#    help(out)
#    print out.fun,out.jac#,out.hess
    return out.x,out.jac

#    p,covar = optimize.curve_fit( func, x, y, p0=pin, sigma=ey, bounds=pbound, method='trf')
#    return p,covar

######operations on lightcurves####################

def rebin(x,y,z,bins):
    """
    For rebinning data, maybe changing the sampling of a lightcurve
    (to build signal).
    """
    index = sp.digitize(x,bins)
    xout = []
    yout = []
    zout = []
    for i in range(bins.size):
        index2 = sp.where(index==i)[0]
        if index2.size > 0:
            xout.append(sp.mean(x[index2]))
            yout.append(sp.mean(y[index2]))

            z1 = sp.std(y[index2])/sp.sqrt(y[index2].size)
            z2 = sp.sqrt(1./sp.sum(1./z[index2]**2))
            zout.append(max(z1,z2))

        else:
            xout.append(bins[i])
            yout.append(0)
            zout.append(0)
                        
            
    return sp.array(xout),sp.array(yout),sp.array(zout)

def gen_lc(t, tau_damp):
    """ alt method for doing a drw"""

    x = sp.randn(t.size)

    dif =  t - t.reshape(t.size,1)
    covar = sp.exp(-abs(dif)/tau_damp)
    l = linalg.cholesky(covar)

    return sp.asarray(x*sp.matrix(l))[0]
    
    

class FakeLC(object):
    """
    a class for generating fake data.  You specify the power spectrum
    of the process, and this returns samples for an input time series
    
    alpha is the slope of the power law for the power spec 

    t_break is the minimum time scale at which you expect to see
    correlations--below this, variations on shorter scales have 0 power
    (can add in measurement noise for these).

    t_turn is the maximum time scale at which you expect to see
    correlations, i.e., after this, the lightcurve becomes decoherent
    (zero power for longterm trends)
    """

    def __init__(self,alpha,t_break,t_turn):
        self.alpha = sp.sqrt(alpha)

        if t_break == 0:
            self.fbreak = sp.inf
        else:
            self.fbreak  = 1./t_break

        self.fturn  = 1./t_turn

    def __call__(self,t):
        npoints = int(len(t)*200.0)
        #adding 1 to the range to make sure that interpolation
        #array has a larger domain than t, else scipy raises an error
        tstep = (t[-1] + 1 - t[0])/npoints
        fmax = 1./tstep


        f = sp.randn(npoints//2 + 1) + sp.randn(npoints//2 + 1)*1j
        feval = sp.r_[0:fmax:1j*(npoints//2.0 + 1)]
        p = sp.r_[0.,
                  (1./feval[1::])**(self.alpha)
                  ]
        
        #this could be adjusted  for amplitude of gaussian white noise
        p[feval > self.fbreak] = 0.0 
        #this could be adjusted for long term trends (mean, linear, etc.)
        p[feval < self.fturn]  = 0.0
        f *= p

        f = sp.r_[f,sp.conj(f[-1:0:-1])]
        #set the mean to 0
        f[0] = 0.0

        y = fft.ifft(f*(2*npoints - 1))
        #take out the scaling
        y = 2*(y.real - y.real.min())/(y.real.max() - y.real.min()) - 1 
        
        tout = sp.r_[t[0] : t[-1]  + 1 + tstep : tstep]
        interp = interp1d(tout,y)

        return interp(t)

class RandomField(object):
    """
    This will generate a 2D random field. You can specify the power
    spectrum and dimensions of the field.  Spatial units are pixels.
    Uses the same procedure as FakeLC.

    Field properties are in 1D (radius), and isotropic (independent of theta)

    Maybe has some applications to image processing?

    """
    def __init__(self,alpha,r_break,r_turn):
        self.alpha = sp.sqrt(alpha)

        if r_break == 0:
            self.fbreak = sp.inf
        else:
            self.fbreak  = 1./r_break

        self.fturn  = 1./r_turn

    def __call__(self,nx,ny):


        f = sp.randn( (nx + 1 )*(ny//2 + 1) ) + sp.randn( (nx + 1)*(ny//2 + 1) )*1j            
        f = f.reshape( (nx + 1 ),ny//2 + 1)
        #hermitian condition is tricky
        f2  = sp.c_[sp.zeros(f.shape)[:,1:],f]

        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                if i >= nx//2:
                    if j == 0: 
                        continue
                f2[i,ny//2 -j ] = sp.conj( f[f.shape[0]-1 - i,j] )

        #apply the filter
        r = sp.sqrt( sp.r_[-nx//2:nx//2 + 1]**2 + sp.r_[-ny//2:ny//2 + 1].reshape(ny + 1 ,1)**2)
        feval = 1/r

        p = feval**(-self.alpha)
        p[feval > self.fbreak] = 0
        p[feval < self.fturn]  = 0

        f2 *= p

                    
        #now do the wrap arround
        f = sp.r_[f2[ny//2::],f2[0:ny//2]]
        f = sp.c_[f[:,nx//2::],f[:,0:nx//2]]
        f[0,0] = 0
       
        z  = fft.ifft2(f*f.size).real
        return (z - z.min())/(z.max() - z.min())


def decimal_to_sexigesimal(ra,dec, return_string=True):
    ra1 = (ra*12./180).astype(int)
    ra2 = ((ra*12./180 - ra1)*60).astype(int)
    ra3 = ((ra*12./180 - ra1)*60 - ra2)*60

    dec1 = dec.astype(int)
    dec2 = ((dec - dec1)*60).astype(int)
    dec3 = ((dec - dec1)*60 - dec2)*60

    RA  = sp.array(['%i:%02i:%02.4f'%(ra1[i],ra2[i],ra3[i]) for i in range(ra.size)])
    DEC = sp.array(['%i:%02i:%02.4f'%(dec1[i],dec2[i],dec3[i]) for i in range(dec.size)])

    return RA,DEC

def sexigesimal_to_decimal(ra,dec, return_string=True):
    RA = ((ra[2]/60. + ra[1])/60. + ra[0])*180/12.
    if dec[0] < 0:
        DEC = -(dec[2]/60. + dec[1])/60. + dec[0]
    else:
        DEC = (dec[2]/60. + dec[1])/60. + dec[0]


    return RA,DEC

def boxgraph(ax):
    xl,xh = ax.get_xlim()
    yl,yh = ax.get_ylim()
    r = (yh - yl)/(xh - xl)
    ax.set_aspect(1./r)

def equ_to_ecliptic(ra,dec):
    """Downloaded transforms from wikipedia, takes degrees, outputs
    degrees, but internal is radians.

    """
#    ra1 = ra.values*3600/206265.
 #   dec1 = dec.values*3600/206265.
    ra1 = ra*3600/206265.
    dec1 = dec*3600/206265.
    obliquity = 23*3600/206265.
    
#    tan_lam  = (sp.sin(ra1)*sp.cos(obliquity) + sp.tan(dec1)*sp.sin(obliquity))/sp.cos(ra1)
#    sin_beta = sp.sin(dec1)*sp.cos(obliquity) - sp.cos(dec1)*sp.sin(obliquity)*sp.sin(ra1)
#
#    return sp.arctan(tan_lam)*206265/3600., sp.arcsin(sin_beta)*206265/3600.

    sin_beta = sp.sin(dec1)*sp.cos(obliquity) - sp.cos(dec1)*sp.sin(obliquity)*sp.sin(ra1)
    cos_beta = sp.cos(sp.arcsin(sin_beta))
    
    cos_lam = sp.cos(dec1)*sp.cos(ra1)/cos_beta
    sin_lam = sp.cos(dec1)*sp.sin(ra1)*sp.cos(obliquity) + sp.sin(dec1)*sp.sin(obliquity)

    lam = sp.arctan(sin_lam/cos_lam)
    #quad 2, but thinks it is in quad 4
    m = (cos_lam < 0)*(sin_lam > 0)
    lam[m] = lam[m] + sp.pi
    #quad 3, but thinks it is in quad 1
    m = (cos_lam < 0)*(sin_lam < 0)
    lam[m] += sp.pi
    #quad 4, turn neg to pos
    m = (cos_lam > 0)*(sin_lam < 0)
    lam[m] += 2*sp.pi
    #    if cos_lam > 0 and sin_lam > 0:
 #   #quad 1
 #       print('quad 1')
 #   elif cos_lam < 0 and sin_lam > 0:
 #       #quad 2, but it thinks it is in quad 4
 #       lam += sp.pi
 #   elif cos_lam < 0 and sin_lam < 0:
 #       #quad 3, but it thinks it is in quad 1
 #       lam += sp.pi
 #   elif cos_lam > 0 and sin_lam < 0:
 #      #quad 4
 #       print('quad 4')
        
    return lam*206265/3600., sp.arcsin(sin_beta)*206265/3600.
    
###class MatlabStructure(object):
###    """A class to make working with data structures from matlab tractable.
###    This is based on matlab files produced by running ray-tracing
###    calculations on models of the TESS lenses, went to me by Deb Woods (of
###    Lincoln Labs).  
###    
###    The main point are: 
###    1) read the file, using h5py.
###    2) get an initial list of keys for the groups, and likely subgroups
###    3) check if something is a group or dataset---
###       if group get keys, if dataset get dimensions
###    4) dereference the data, given the group and dimensions.
###    
###    Likely won't work on all .mat files, but a good starting point.
###    """    
###    def __init__(self, infile):
###        self._file_buffer = h5py.File(infile)
###        self.top_keys = [ key for key in self._file_buffer.keys() if '#refs' not in key ]
###        self.sub_keys = {}
###        
###        for k in self.top_keys:
###            if isinstance(self._file_buffer[k], h5py._hl.group.Group):
###                if k == '#refs#': continue
###                self.sub_keys[k] = [ key for key in self._file_buffer[k].keys() ]
###
###    def print_data_keys(self):
###        for key in self.top_keys:
###            print(key)
###            for k in self.sub_keys[key]:
###                print('     ',k)
###            
###    def check_group(self,topkey,subkey):
###        if isinstance(self._file_buffer[topkey][subkey], h5py._hl.group.Group):
###            print('{sub:s} in {top:s} is a group'.format(sub=subkey,top=topkey))
###            newkeys = [ k for k in self._file_buffer[topkey][subkey].keys() ]
###            print('new group with keys:')
###            print(newkeys)
###            return newkeys
###        else:
###            print('{sub:s} in {top:s} not a group'.format(sub=subkey,top=topkey))
###
###    def check_data(self,topkey,subkey):
###        if isinstance(self._file_buffer[topkey][subkey], h5py._hl.dataset.Dataset):
###            print('{sub:s} in {top:s} is a data set of shape:'.format(sub=subkey,top=topkey))
###            print(self._file_buffer[topkey][subkey].shape)
###            return self._file_buffer[topkey][subkey].shape
###        else:
###            print('{sub:s} in {top:s} not a dataset'.format(sub=subkey,top=topkey))
###            
###    def dereference(self, topkey, subkey,indices):
###        return self._file_buffer[
###                                             self._file_buffer[topkey][subkey][indices]
###                                            ][:]


    
#These were written before I appreciated scipy packages.  Maybe of conceptual use....

####def lininterp(x,y,shift,deltax):
####    """
####    linearly interpolate on y for a set of evenly spaced x values.  Shift
####    is the chang in x, and deltax is the spacing between elements in x.
####    It is assumed that shift is less than deltax, otherwise exits.
####    """
####    if(shift>deltax):
####        print( "Interpolation failed:  didn't shift data correctly.")
####    #        exit
####    s = shift/deltax 
####    yint = np.zeros(len(y)-1)
####    for i in range(len(y)-1):
####        yint[i] = s*(y[i+1] - y[i]) + y[i]
####    return np.array(yint)
####
####def convolve(signal,response,p,dispersion,flag):
####    """
####    Convolves a signal with a response function, following Numerical
####    Recipes 13.1.1-2.  signal is the data, response is a function, p
####    are parameters that go into that function, and dispersion is delta
####    x of the signal, to make sure the units are correct.  If flag is
####    set to -1, deconvolution is done instead.
####
####    The response is assumed to be symetric, and is constructed in such
####    a way that it samples the chosen function the same number of times
####    as the data.  For very broad response functions, there will be
####    edge effects.  For vary narrow response functions, they will not
####    be adequately sampled
####    """
#####    n1 = len(signal)
#####zeropad
#####    signal = np.append(signal,np.zeros(int(n1/2) ))
####    n = len(signal)
####    print(n)
#####in wavelength space, make sure units are correct
####    x  = np.r_[-n/2:n/2]*dispersion
####    r1 = response(x,p)
#####wrap around
####    r1 = np.r_[r1[n/2+1::],r1[0:n/2+1.]]
#####do the convolution
####    z1 = np.fft.fft(signal)
####    z2 = np.fft.fft(r1)
####
####    if(flag==-1):
#####deconvolve
####        conv = np.fft.ifft(z1/z2)
####    else:
####        conv = np.fft.ifft(z1*z2)
####
#####return the unpolluted part
####    return np.real(conv)
####
####def cross_correlate(in1,in2,maxlag,dispersion):
####    """" Cross correlate inputs, so as to calculate initial guess for
####    offset.
####
####    in1 and in2 are the working data 
####    maxlag is the maximum shift to try 
####    dispersion is the spacing of the data.
####
####    This function cross correlates both spectra, and returns the smaller lag.
####    """
####    pad = int(maxlag/dispersion)
####    w1 = np.append(in1,np.zeros(pad))
####    w2 = np.append(in2,np.zeros(pad))
####
####    z1 = np.fft.fft(w1)
####    z2 = np.fft.fft(w2)
####
####    corr1  = np.fft.ifft(z1*z2.conj())
####    shift1 = np.where(corr1==np.max(corr1))
####    corr2  = np.fft.ifft(z2*z1.conj())
####    shift2 = np.where(corr2==np.max(corr2))
####
####    if shift1<shift2:
####        return -shift1[0][0]*dispersion
####    else:
####        return shift2[0][0]*dispersion

