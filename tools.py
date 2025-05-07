from __future__ import print_function

import numpy as np
import scipy as sp
from scipy import optimize,linalg
from scipy.interpolate import interp1d,interp2d
from scipy.interpolate import splrep, splev
from scipy.integrate import quad
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
    C = np.array([np.sum(y/ey/ey),np.sum(y*x/ey/ey)])
    A = np.array([
            [np.sum(1./ey/ey), np.sum(x/ey/ey)],
            [np.sum(x/ey/ey),  np.sum(x*x/ey/ey)]
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
    merit = lambda p1,x1,xe1,y1,ye1: (y1 - p1[0] - p1[1]*x1)/np.sqrt((ye1**2 + (p1[1]*xe1)**2))
    p,covar,dum1,dum2,dum3 = optimize.leastsq(merit,x0=[0,0],args=(x,ex,y,ey),xtol = 1e-8,full_output=1)
    return p,covar

def fitfunc(func,pin,x,y,ey):
    """Non-linear least square fitter.  Assumes no errors in x.  Utilizes
    scipy.optimize.leastsq.  Uses a Levenberg-Marquardt algorithm
    (gradient chasing and Newton's method).  The user must provide an
    object that is the functional form to be fit, as well as initial
    guesses for the parameters.

    The function definition should call f(x,params), where x is the
    independent variable and params is an array

    This function returns the parameters and the covariance matrix.
    Note that leastsq returns the jacobian, a message, and a flag as
    well.

    """
    merit = lambda params,func,x,y,ey:  (y - func(x,params))/ey
    p,covar,dum1,dum2,dum3 = optimize.leastsq(merit,pin,args=(func,x,y,ey),xtol = 1e-8,full_output=1)
    #print(dum1)
    #print(dum2)
    #print(dum3)
    return p,covar

def fitfunc_bound(func,pin,x,y,ey,pbound):
    """
    Non-linear least square fitter, with an option of contraints

    ***********
    
    """
    merit = lambda params,func,x,y,ey:  np.sum( (y - func(x,params))**2/ey**2 )
    #return with weird object....
    out = optimize.minimize(merit,pin,args=(func,x,y,ey),tol = 1.e-8,method='L-BFGS-B',bounds = pbound,options={'disp':True, 'maxiter':1000})
#    out = optimize.fmin_bfgs(merit,pin,args=(func,x,y,ey),gtol = 1.e-8)
#    out = optimize.minimize(merit,pin,args=(func,x,y,ey),tol = 1.e-8,method='SLSQP',bounds = pbound)
#    print out
#    print 'success:',out.success
#    errors = np.sqrt(ey**2 * out.jac**2)
#    print out.keys()
#    print 'iterations:',out.nit,out.nfev#, out.njev, out.nhev
#    help(out)
#    print out.fun,out.jac#,out.hess
    return out.x,out.jac

#    p,covar = optimize.curve_fit( func, x, y, p0=pin, sigma=ey, bounds=pbound, method='trf')
#    return p,covar


#sometimes, I just want to see how good a fit to my residuals
#reproduces white noise.  No weights
def fit_gaussian(x,y):
    gauss = lambda x,p: p[2]*np.exp(-0.5*(x - p[0])**2/p[1]**2)
    p,covar = fitfunc(gauss, [0.0, 1.0, max(y)], x,y, np.ones(len(y))  )
    return p,covar
######operations on lightcurves####################

def rms_slide(t,y,win_len):
    w = np.ones(win_len)
    w = w/np.sum(w)
    #root of the sliding-mean square
    rms1 = np.sqrt( np.convolve(( y - y.mean() )**2,w,mode='same'))
    
    if len(rms1) > len(y):
        rms1 = np.ones(len(y))*np.std(y - y.mean())
        
    return rms1

def rebin(x,y,z,bins, use_std=False, rescale_error=False, min_points=None):
    """
    For rebinning data, maybe changing the sampling of a lightcurve
    (to build signal).
    """
    index = np.digitize(x,bins)
    xout = np.zeros(len(bins) )
    yout = np.zeros(len(bins) )
    zout = np.zeros(len(bins) )
    for i in np.unique(index):
        if i == 0 or i == len(bins):
            continue
        m = np.where(index == i)[0]
        if min_points is not None:
            if len(x[m]) < min_points:
                continue
            
        xout[i - 1] = np.mean(x[m])
        yout[i - 1] = np.mean(y[m])
        if use_std:
            zout[i - 1] = np.std(y[m])
        else:
            zout[i - 1] = np.mean(z[m])

        if rescale_error:
            zout[i -1] = zout[i - 1]/np.sqrt(len(y[m]))

        
            
    return xout, yout,zout

def gen_lc(t, tau_damp):
    """ alt method for doing a drw"""

    x = np.random.normal(0,1,t.size)

    dif =  t - t.reshape(t.size,1)
    covar = np.exp(-abs(dif)/tau_damp)
    l = linalg.cholesky(covar)

    return np.asarray(x*np.matrix(l))[0]
    
    

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

    def __init__(self,alpha,t_break,t_turn,
                 low_freq_power=False):
        self.alpha = np.sqrt(alpha)

        if t_break == 0:
            self.fbreak = np.inf
        else:
            self.fbreak  = 1./t_break

        self.fturn  = 1./t_turn

        self.low_freq_power = low_freq_power

    def __call__(self,t):
        npoints = int(len(t)*4.0)
        #adding 1 to the range to make sure that interpolation
        #array has a larger domain than t, else scipy raises an error
        tstep = (t[-1] + 1 - t[0])/npoints
        fmax = 1./tstep


        f = np.random.normal(0,1, npoints//2 + 1
                             ) + np.random.normal(0,1,npoints//2 + 1)*1j
        feval = np.r_[0:fmax:1j*(npoints//2.0 + 1)]
        p = np.r_[0.,
                  (1./feval[1::])**(self.alpha)
                  ]
        
        #this could be adjusted  for amplitude of gaussian white noise
        p[feval > self.fbreak] = 0.0
        if self.low_freq_power:
            abs_diff = abs( feval - self.fturn )
            p[feval < self.fturn]  = p[ abs_diff == min(abs_diff) ]
        else:
            #this could be adjusted for long term trends (mean, linear, etc.)
            p[feval < self.fturn]  = 0.0
        
        f *= p

        f = np.r_[f,np.conj(f[-1:0:-1])]
        #set the mean to 0
        f[0] = 0.0

        y = fft.ifft(f*(2*npoints - 1))
        #take out the scaling
        y = 2*(y.real - y.real.min())/(y.real.max() - y.real.min()) - 1 

        
        tout = np.r_[t[0] : t[-1]  + 1 + tstep : tstep]
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
        self.alpha = np.sqrt(alpha)

        if r_break == 0:
            self.fbreak = np.inf
        else:
            self.fbreak  = 1./r_break

        self.fturn  = 1./r_turn

    def __call__(self,nx,ny):


        f = np.random.normal(0,1, (nx + 1 )*(ny//2 + 1)
                             ) + np.random.normal(0,1, (nx + 1)*(ny//2 + 1) )*1j            
        f = f.reshape( (nx + 1 ),ny//2 + 1)
        #hermitian condition is tricky
        f2  = np.c_[np.zeros(f.shape)[:,1:],f]

        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                if i >= nx//2:
                    if j == 0: 
                        continue
                f2[i,ny//2 -j ] = np.conj( f[f.shape[0]-1 - i,j] )

        #apply the filter
        r = np.sqrt( np.r_[-nx//2:nx//2 + 1]**2 + np.r_[-ny//2:ny//2 + 1].reshape(ny + 1 ,1)**2)
        feval = 1/r

        p = feval**(-self.alpha)
        p[feval > self.fbreak] = 0
        p[feval < self.fturn]  = 0

        f2 *= p

                    
        #now do the wrap arround
        f = np.r_[f2[ny//2::],f2[0:ny//2]]
        f = np.c_[f[:,nx//2::],f[:,0:nx//2]]
        f[0,0] = 0
       
        z  = fft.ifft2(f*f.size).real
        return (z - z.min())/(z.max() - z.min())


class ARModel(object):
    def __init__(self,y, N_memory, mode='timeseries'):
        self.y = y
        acf = np.correlate(f,f,'same')
        self.acf = acf
        self.N_memory = N_memory        
        self.mode = mode

        if self.mode == 'timeseries':
            self.coeffs = self.fit_coeffs(self.y, self.N_memory)
        elif self.mode == 'acf':
            self.coeffs = self.fit_coeffs(self.acf, self.N_memory)
        else:
            raise ValueError("must instantiate ARModel with mode equal to 'timeseries' or 'acf'")

        self.model_values = np.convolve(self.y,self.coeffs,'valid')
        self.undefined_times =  np.zeros(len(self.y),dtype=bool)
        self.undefined_times[0:self.N_memory] = True

    def fit_coeffs(self, acf, N_memory):        
        acf_roll = []
        for n in range(N_memory):
            acf_tmp = np.roll(acf, -(n+1))
            #to avoid wrap around, set the end of the lagged acf to zero
            acf_tmp[-(n+1):] = 0
            acf_roll.append(acf_tmp)
        
        acf_roll = np.array(acf_roll)
    
        #fill in the design matrices
        #len(acf)x1 matrix
        C = []
        for ii in range(len(acf_roll)):
            C.append(np.sum(acf*acf_roll[ii]))
        C = np.array(C)

        #len(acf)xN_memory matrix
        A = []
        for ii in range(len(acf_roll)):
            row = []
            for jj in range(len(acf_roll)):
                row.append(np.sum(acf_roll[ii]*acf_roll[jj]))
            A.append(row)
        A = np.array(A)

        B = linalg.solve(A,C)
        return B

    def get_response(self):
        extrap = np.zeros(len(self.coeffs))
        extrap[-1] = 1.0
        for ii in range(len(self.coeffs)):
            extrap[len(coeffs) -1 - ii] = np.sum(extrap*self.coeffs[::-1])

        return extrap[::-1]

    def extrapolate(self,npredict):
        extrap = self.y[-len(coeffs):]
        for ii in range(npredict):
            extrap = np.r_[extrap, np.sum(extrap[-len(coeffs):]*coeffs[::-1]) ]
        return extrap        

def decimal_to_sexigesimal(ra,dec, return_string=True):
    ra1 = (ra*12./180).astype(int)
    ra2 = ((ra*12./180 - ra1)*60).astype(int)
    ra3 = ((ra*12./180 - ra1)*60 - ra2)*60

    dec1 = dec.astype(int)
    dec2 = ((dec - dec1)*60).astype(int)
    dec3 = ((dec - dec1)*60 - dec2)*60

    RA  = np.array(['%i:%02i:%02.4f'%(ra1[i],ra2[i],ra3[i]) for i in range(ra.size)])
    DEC = np.array(['%i:%02i:%02.4f'%(dec1[i],dec2[i],dec3[i]) for i in range(dec.size)])

    return RA,DEC

def sexigesimal_to_decimal(ra,dec, return_string=True):
    RA = ((ra[2]/60. + ra[1])/60. + ra[0])*180/12.
    #bug exists, for dec between -01 and 00, will choose the wrong
    #conditional
    if dec[0] < 0:
        DEC = -(dec[2]/60. + dec[1])/60. + dec[0]
    else:
        DEC = (dec[2]/60. + dec[1])/60. + dec[0]


    return RA,DEC

def parse_sexigesimal(instring):
    ra,dec = instring.replace(',',' ').split()
    ra_list = [ float(e)  for e in ra.split(':') ]
    dec_list = [ float(e) for e in dec.split(':') ]
    return ra_list,dec_list


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
    
#    tan_lam  = (np.sin(ra1)*np.cos(obliquity) + np.tan(dec1)*np.sin(obliquity))/np.cos(ra1)
#    sin_beta = np.sin(dec1)*np.cos(obliquity) - np.cos(dec1)*np.sin(obliquity)*np.sin(ra1)
#
#    return np.arctan(tan_lam)*206265/3600., np.arcsin(sin_beta)*206265/3600.

    sin_beta = np.sin(dec1)*np.cos(obliquity) - np.cos(dec1)*np.sin(obliquity)*np.sin(ra1)
    cos_beta = np.cos(np.arcsin(sin_beta))
    
    cos_lam = np.cos(dec1)*np.cos(ra1)/cos_beta
    sin_lam = np.cos(dec1)*np.sin(ra1)*np.cos(obliquity) + np.sin(dec1)*np.sin(obliquity)

    lam = np.arctan(sin_lam/cos_lam)
    #quad 2, but thinks it is in quad 4
    m = (cos_lam < 0)*(sin_lam > 0)
    lam[m] = lam[m] + np.pi
    #quad 3, but thinks it is in quad 1
    m = (cos_lam < 0)*(sin_lam < 0)
    lam[m] += np.pi
    #quad 4, turn neg to pos
    m = (cos_lam > 0)*(sin_lam < 0)
    lam[m] += 2*np.pi
    #    if cos_lam > 0 and sin_lam > 0:
 #   #quad 1
 #       print('quad 1')
 #   elif cos_lam < 0 and sin_lam > 0:
 #       #quad 2, but it thinks it is in quad 4
 #       lam += np.pi
 #   elif cos_lam < 0 and sin_lam < 0:
 #       #quad 3, but it thinks it is in quad 1
 #       lam += np.pi
 #   elif cos_lam > 0 and sin_lam < 0:
 #      #quad 4
 #       print('quad 4')
        
    return lam*206265/3600., np.arcsin(sin_beta)*206265/3600.
    
##class MatlabStructure(object):
##    """A class to make working with data structures from matlab tractable.
##    This is based on matlab files produced by running ray-tracing
##    calculations on models of the TESS lenses, went to me by Deb Woods (of
##    Lincoln Labs).  
##    
##    The main point are: 
##    1) read the file, using h5py.
##    2) get an initial list of keys for the groups, and likely subgroups
##    3) check if something is a group or dataset---
##       if group get keys, if dataset get dimensions
##    4) dereference the data, given the group and dimensions.
##    
##    Likely won't work on all .mat files, but a good starting point.
##    """    
##    def __init__(self, infile):
##        self._file_buffer = h5py.File(infile)
##        self.top_keys = [ key for key in self._file_buffer.keys() if '#refs' not in key ]
##        self.sub_keys = {}
##        
##        for k in self.top_keys:
##            if isinstance(self._file_buffer[k], h5py._hl.group.Group):
##                if k == '#refs#': continue
##                self.sub_keys[k] = [ key for key in self._file_buffer[k].keys() ]
##
##    def print_data_keys(self):
##        for key in self.top_keys:
##            print(key)
##            for k in self.sub_keys[key]:
##                print('     ',k)
##            
##    def check_group(self,topkey,subkey):
##        if isinstance(self._file_buffer[topkey][subkey], h5py._hl.group.Group):
##            print('{sub:s} in {top:s} is a group'.format(sub=subkey,top=topkey))
##            newkeys = [ k for k in self._file_buffer[topkey][subkey].keys() ]
##            print('new group with keys:')
##            print(newkeys)
##            return newkeys
##        else:
##            print('{sub:s} in {top:s} not a group'.format(sub=subkey,top=topkey))
##
##    def check_data(self,topkey,subkey):
##        if isinstance(self._file_buffer[topkey][subkey], h5py._hl.dataset.Dataset):
##            print('{sub:s} in {top:s} is a data set of shape:'.format(sub=subkey,top=topkey))
##            print(self._file_buffer[topkey][subkey].shape)
##            return self._file_buffer[topkey][subkey].shape
##        else:
##            print('{sub:s} in {top:s} not a dataset'.format(sub=subkey,top=topkey))
##            
##    def dereference(self, topkey, subkey,indices):
##        return self._file_buffer[
##                                             self._file_buffer[topkey][subkey][indices]
##                                            ][:]


def lumdist(z):
    def E(z,Omega_m):
        #assumes flat universe
        return 1./np.sqrt(
            Omega_m*(1.+z)**3 + (1. - Omega_m)
        )
    H0 = 70
    Omega_m = 0.3
    c  = 2.99792458e10

    #calculate the luminosity distnace, returns in Mpc
    integral = quad(E, 0, z, args=(Omega_m))
    #h0 in km/s, must convert to 1./s
    dp = c/(H0 /3.08568e13/1.e6)*integral[0]
    
    return (1+z)*dp/3.08568e18/1.e6

#import psycopg2
import pandas as pd

##class RemoteDatabase(object):
##    def __init__(self,host,user,password,dbname):
##        self.connection = psycopg2.connect( host=host, user=user, password=password, dbname=dbname)
##        
##    def do_query(self, query ) :
##        cur = self.connection.cursor()
##        cur.execute( query )
##        result = pd.DataFrame(cur.fetchall(),columns=('tic','tmag','rad','mass','lumclass','objtype'))
##        return result
    

def split_list(inlist, nlist):
    """
    A function to split an input list into multiple lists, to run on several cores
    """
    file_list = np.genfromtxt(inlist,dtype=str)

    count = 1
    i0 = 0
    for i in range(file_list.size):
        if i == 0: continue
        if i % nlist == 0:
            np.savetxt(inlist+ str(count), np.c_[file_list[i0:i0 + 1000]], fmt='%s')
            io = deepcopy(i)
            count += 1
        elif i == stop -1:
            np.savetxt(inlist + str(count),np.c_[ file_list[i0::]], fmt='%s')

def detrend_lc_with_splines(t,f,smooth=1.e7):
    #originally developed for akshata, this does a reasonably job of
    #fitting a smooth spline to a TESS 2min LC
    spline_params = splrep(t,f,w = np.ones(len(f)), k=3, s= smooth)
    return splev(t, spline_params)



def calc_t90(x,y, plot=False):

    #assume y is already sorted
    cumsum = np.cumsum(y)
    cumsum /= np.sum(y)


    abs_r05 = abs(cumsum - 0.05)
    abs_r95 = abs(cumsum - 0.95)
    m05 = np.where( abs_r05 == min(abs_r05))[0][0]
    m95 = np.where( abs_r95 == min(abs_r95))[0][0]
    if plot == True:
        plt.figure()
        plt.plot(x,cumsum,'k')
        yl,yh = plt.gca().get_ylim()
        xl,xh = plt.gca().get_xlim()

        plt.plot([xl,xh],[0.05,0.05],'r')
        plt.plot([xl,xh],[0.95,0.95],'r')
        plt.plot([x[m05],x[m05]],[yl,yh],'r')
        plt.plot([x[m95],x[m95]],[yl,yh],'r')

        plt.gca().set_ylim([yl,yh])
        plt.gca().set_xlim([xl,xh])

    if plot == False:
        return x[m95] - x[m05]
    else:
        return x[m95] - x[m05], plt.gcf(), plt.gca()


def Planck(lam,T):

    h = 6.63e-27
    k = 1.4e-16
    c = 2.9979e10
    print(h*c/k/T, 0.290/T)
    num = 2*h*c**2/lam**5
    denom = np.exp(h*c/lam/k/T) - 1
    return num/denom
    

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

