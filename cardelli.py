import scipy as sp

def extinction(lambda1in,R,unit = 'microns'):
###This is CCM89, and assumes microns
    if 'ang' in unit:
        lambda1 = lambda1in/1.e4
    else:
        lambda1 = lambda1in

    if (lambda1 > 100).all():
        print "Check units!  This program assumes microns"

    if (lambda1 > 3.0).any():
        print  "Warning: extrapolating into the far IR (lambda > 3 microns)"
    if (lambda1 < 0.125).any():
        print 'Warning:  extreme UV is an extrapolation'
    if (lambda1 < 0.1).any():
        print 'warning: extrapolating into the extreme UV (lambda < 1000 A)'


    a = sp.zeros(lambda1.size)
    b = sp.zeros(lambda1.size)

    m = (lambda1 > 0.909)
    a[m] =  0.574*(1/lambda1[m])**(1.61)
    b[m] = -0.527*(1/lambda1[m])**(1.61)
        
    m = (lambda1 > 0.30303)*(lambda1 <= 0.909)
    x = 1/lambda1[m] - 1.82
    a[m] = 1 + 0.17699*x - 0.50447*x**2 - 0.02427*x**3 + 0.72085*x**4 + 0.01979*x**5 - 0.7753*x**6 + 0.32999*x**7
    b[m] =     1.41338*x + 2.28305*x**2 + 1.07233*x**3 - 5.38434*x**4 - 0.62251*x**5 + 5.3026*x**6 - 2.09002*x**7

    m = (lambda1 > 0.125)*(lambda1 <= 0.30303)
    x = 1/lambda1[m]
    a[m] =  1.752 - 0.316*x - 0.104/( (x - 4.67)**2 + 0.341) 
    b[m] = -3.090 + 1.825*x + 1.206/( (x - 4.62)**2 + 0.263) 

    m = (lambda1 > 0.125)*(lambda1 <= 0.1695)
    x = 1/lambda1[m]
    a[m] += -0.04473*(x - 5.9)**2 - 0.009779*(x-5.9)**3
    b[m] +=  0.21300*(x - 5.9)**2 + 0.120700*(x - 5.9)**3

    m = (lambda1 < 0.125)
    x = 1/lambda1[m]
    a[m] = -1.073 - 0.628*(x - 8.) + 0.137*(x - 8.)**2 - 0.070*(x - 8.)**3
    b[m] = 13.670 + 4.257*(x - 8.) - 0.420*(x - 8.)**2 + 0.374*(x - 8.)**3

    return a + b/R
