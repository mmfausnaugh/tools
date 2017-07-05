import scipy as sp
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord#, skyoffset_frame
import astropy.units as u

from scipy.integrate import simps

from tools import equ_to_ecliptic as eq2ec

""""
Models for seeing where and what is in a TESS field of view.  For design:

1---need to know where TESS is pointing, will generally be equatorial

2---So will generally need to map this to ecliptic coordinates if we
want to map the camera pointings 

3---need to define a fov frame based on camera centers in
order to get the edges correct (I think?), astropy makes this easy!

"""


class TESS:
    """
stores the pointing, and knows how to map that to camera centers
"""
    def __init__(self, ra, hemisphere, ecliptic_coords = False):
        self.ra = ra
        self.hemisphere = hemisphere
        if 'N' in self.hemisphere or 'n' in self.hemisphere:
            self.sign = 1
        elif 'S' in self.hemisphere or 's' in self.hemisphere:
            self.sign = -1
        else:
            raise ValueError("Only accepts north or south hemisphere (N or S)")

        #TESS lattidue pointing and camera centers are always
        #predefined, cams 1 through 4
        self.camcent_latt = sp.r_[ 18., 42., 66., 90. ] * self.sign
        

        #user has the option to specify ecliptic coordinate
        if ecliptic_coords:
            self.camcent_long = sp.ones(4)*ra*180./12.
        else:
            #pick out ecliptic lattitude
            self.camcent_long = self._get_ecliptic_pointing()

        #may as well store SkyCoord objects with the camera pointing centers
        self.camcenters = []
        
        for z in zip(self.camcent_long, self.camcent_latt):
            self.camcenters.append(
                SkyCoord(z[0], z[1], unit=(u.degree,u.degree),frame = 'geocentrictrueecliptic')
            )
            
            
        #define SkyCoord objects with size of tess cameras and centers
        #at the camera pointings

        #specifiy the edges of the FOV in camera cordinates (units of degrees) where the
        #There are five corners so as to close
        #the loop,
        xcorner = sp.array([-12., 12., 12., -12., -12.,])
        ycorner = sp.array([-12., -12., 12., 12., -12.,])
        self.FOVs = []
        for center in self.camcenters:
            self.FOVs.append( FOV(center, xcorner, ycorner) )
            

    def _get_ecliptic_pointing(self):
        #calculate a grid of ecliptic cordinates for the RA slice,
        #then pick the Dec that matches where the telescope points
        #(middle will always be 54 degrees, just either north or south)
        ra_use = sp.ones(100)*self.ra
        dec_use = sp.r_[-90:90:100j]
        temp_pointing = SkyCoord(ra_use, dec_use, unit=(u.hour, u.degree), frame='icrs')
        ecliptic1 = temp_pointing.transform_to('geocentrictrueecliptic')


        #pick ecliptic lattiude nearest +/- 54 degrees, to cross match
        #ecliptic longitude
        iuse = sp.where( abs(ecliptic1.lat.degree - (self.sign*54.)) ==
                                   abs(ecliptic1.lat.degree - (self.sign*54.)).min() )[0]

        return  sp.ones(4)*ecliptic1[iuse].lon.degree


    def plot_FOVs(self,axuse, cuse='green'):
        """Should note that we are kind of expecting aitoff or mollweide
        projection, so auto converts to radians

        """
        for fov in self.FOVs:
            fov.plot(axuse,cuse=cuse)
            

class FOV:
    def __init__(self, center, xcorners, ycorners):
        #five corners in order to all the way around the box
        self.center = center
        self.newframe = self.center.skyoffset_frame()
        self.xcorner = xcorners
        self.ycorner = ycorners

        self.fov_cam_coords = SkyCoord(self.xcorner, self.ycorner, unit=(u.degree, u.degree), frame=self.newframe )
        self.fov_coords = self.fov_cam_coords.transform_to('geocentrictrueecliptic')

    def area(self):
        #integrate a grid on the FOV after translated into ecliptic
        x = sp.r_[ self.xcorner[0]: self.xcorner[1] : 500j ]
        y = sp.r_[ self.ycorner[0]: self.ycorner[3] : 500j ]
        X,Y = sp.meshgrid(x,y)

        temp = SkyCoord( X, Y, unit= u.degree, frame = self.newframe )
        temp2 = temp.transform_to('geocentrictrueecliptic')

        print(temp2[0:5,0:5])
        
        dx = sp.diff(sp.ravel(temp2.data.lon.degree))
        dy = sp.diff(sp.ravel(temp2.data.lat.degree)) 
        
        return sp.sum(abs(dx)*abs(dy))
        
    def filter_coord(self, lon , lat, coord_system='equatorial'):
        c = SkyCoord(lon,lat, unit = u.degree, frame=coord_system)
        c2 = c.transform_to(self.newframe)

#    lat  = sp.array([d.lat.rad  for d in co.data ])
        mx = ( c2.data.lon.degree > self.xcorner[0] )&( c2.data.lon.degree < self.xcorner[1] )
        my = ( c2.data.lat.degree > self.ycorner[0]  )&(  c2.data.lat.degree < self.ycorner[3] )

        return c[ mx & my ]

    
    def plot(self,axuse, cuse='green'):
        """Should note that we are kind of expecting aitoff or mollweide
        projection, so auto converts to radians

        """
        #define edges, because of the way matplotlib projections try to try geodesics...
        for i in range(self.xcorner.size-1):
            xedge = sp.r_[self.xcorner[i]: self.xcorner[i + 1]: 25j]
            yedge = sp.r_[self.ycorner[i]: self.ycorner[i + 1]: 25j]

            temp = SkyCoord(xedge, yedge, unit=u.degree, frame = self.newframe)
            temp = temp.transform_to('geocentrictrueecliptic')
            lon = temp.data.lon.rad
            lat = temp.data.lat.rad
            axuse.plot(
                lon - sp.pi, lat, '-',color=cuse
            )        
        
        #circles for the coerners
        lon = self.fov_coords.data.lon.rad
        lat = self.fov_coords.data.lat.rad
        axuse.plot(
            lon - sp.pi, lat, 'o',color=cuse
        )

    
        
class GreatCircle:
    """
    Just coordinates defined with longitude = 0 to 360 and lattitude = 0
    """
    def __init__(self, frame, spacing = 10):
        self.lo = sp.r_[0: 360.01:spacing]
        self.la = sp.zeros(self.lo.size)

        self.co = SkyCoord(self.lo, self.la, unit = u.degree,frame=frame)

        #    def plot(self,axuse,cuse='black'):
    def plot(self,axuse,label,cuse='black'):
        lon = self.co.data.lon.rad
        lat  =self.co.data.lat.rad
        axuse.plot(
            lon - sp.pi, lat,'o',color=cuse,label=label
        )                

    def transform_to(self,frame):
        self.co = self.co.transform_to(frame)
        
        
class MilkyWay(GreatCircle):
    def __init__(self):
        super().__init__('galactic')
        self.center = SkyCoord(0.0,0.0, unit = u.degree, frame='galactic')

        
class Equator(GreatCircle):
    def __init__(self):
        super().__init__('icrs')

class Ecliptic(GreatCircle):
    def __init__(self):
        super().__init__('geocentrictrueecliptic')

        
##testing
if __name__ == "__main__":
    t1 = TESS(16.,'N')
    A1,A2 = [],[]
    for fov in t1.FOVs:
        a1 = fov.area()
        A1 = sp.r_[A1,a1]
        #        a1, a2 = fov.area()
 #       A1 = sp.r_[A1,a1]
  #      A2 = sp.r_[A2,a2]
        
#        print(fov.area())
    #    F = plt.figure()
#    ax = F.add_subplot(111,projection='aitoff')
#
#    ra16_s = TESS(16.,'S')
#    ra16_s.plot_FOVs(ax)
#    ax.grid()
#    era = sp.r_[5:21:2.0]
#    for ra in era:
#        for hemi in ['S','N']:
#            ra16_n = TESS(ra, hemi, ecliptic_coords=True)
#            ra16_n.plot_FOVs(ax)
#    ax.grid()
#
#    mw = MilkyWay()
#    mw.transform_to('geocentrictrueecliptic')
#    mw.plot(ax,cuse='r')
###Atot = ( 4.*sp.pi*(180./sp.pi)**2)
###steradian_to_degree = Atot/4./sp.pi
###
###area_per_bin = 2*sp.pi*sp.array([
###    sp.cos(0) - sp.cos(12*sp.pi/180.),
###    sp.cos(12*sp.pi/180.) - sp.cos(36*sp.pi/180.),
###    sp.cos(36*sp.pi/180.) - sp.cos(60*sp.pi/180.),
###    sp.cos(60*sp.pi/180.) - sp.cos(84*sp.pi/180.),
###    sp.cos(84*sp.pi/180.) - sp.cos(96*sp.pi/180.),
###    sp.cos(96*sp.pi/180.) - sp.cos(120*sp.pi/180.),
###    sp.cos(120*sp.pi/180.) - sp.cos(144*sp.pi/180.),
###    sp.cos(144*sp.pi/180.) - sp.cos(168*sp.pi/180.),
###    sp.cos(168*sp.pi/180.) - sp.cos(180*sp.pi/180.)
###    ])
###print(area_per_bin * steradian_to_degree, sp.sum(area_per_bin), sp.sum(area_per_bin*steradian_to_degree))
###print(A1)
####print(sp.c_[A1,A2, A2*steradian_to_degree])
###print(A1[0]*12)
#    plt.show()
