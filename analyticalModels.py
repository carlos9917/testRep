'''
Created on 07.04.2016
contains the functions which allow to compute the power output of a wind farm , estimated thanks to 3 different models:
Jensen: (top hat)
Porte-Agel: (Gaussian)
2D_k Jensen: (cosine)
@author: 00063941
'''

import sys
#sys.path.append('/srv/scr/00061460/OpenFOAM/Puespoeck/analyticalModels/AnalyticalModels')
sys.path.append('/srv/scr/00061460/OpenFOAM/Puespoek/EnerconPlusVestas/analyticalModels/AnalyticalModels')
from farms.hmTools.build_farm import make_turbines
from farms.hmTools.divers import grad2rad
import numpy as np
import math as m


def get_area(R,r,d):
    R=float(R);r=float(r);d=float(d)
    '''
    computes the area of the intersection between two circles
    of radius R and r, separated by a distance d
    ''' 
    if (R+r)>d and d>R and d>r:
        F1=r**2*m.acos((d**2+r**2-R**2)/(2*d*r))
        F2=R**2*m.acos((d**2+R**2-r**2)/(2*d*R))
        F3=0.5*m.sqrt((-d+r+R)*(d+r-R)*(d-r+R)*(d+r+R))
        A=F1+F2-F3
    if (R+r)<d and d>R and d>r:
        A=0
    if d<R or d<r:
        A=np.amin([m.pi*r**2,m.pi*R**2])
    return A
def Iplus(wt, Ct, I0, st):
    '''
    computes the added turbulence intensity at the distance st induced by
    the turbine wt, which has a thrust coefficient Ct.
    The incoming turbulence intencity being I0
    The formulation is the one by Crespo and Hernandez (cf. Amin Niayifar and Fernando Porte-Agel).
    '''
    a=0.5*(1-m.sqrt(1-Ct))
    return 0.73*m.pow(a ,0.8325)*m.pow(I0,0.0325)*m.pow((st/wt.D),-0.32)
    
    
def beta(Ct):
    '''
    returns the factor beta as function of Ct (cf. Paper)
    '''
    return 0.5*(1+m.sqrt(1-Ct))/(m.sqrt(1-Ct))
def epsilon(beta):
    '''
    returns the factor epsilon as function of beta (cf. Amin Niayifar and Fernando Porte-Agel)
    '''
    return 0.2*m.sqrt(beta)
def interpolate(wt,Vref,value):
    '''
    interpolates from the Cp or Ct curve of the turbine wt,
    the power or thrust coefficient corresponding to the incoming wind velocity Vref.
    the variable value defines which Cp or Ct curve will be chosen.
    '''
    if Vref < 0:
        print 'error: unsupported velocity < 0'
        sys.exit(1)
    Vmax=np.amax(wt.vr)
    Vmin=np.amin(wt.vr)
    kar=int((Vref-Vmin)/(Vmax-Vmin)*(len(wt.vr)-1))
    a=float(Vref-wt.vr[kar])/float(wt.vr[kar+1]-wt.vr[kar])
    if value == 'Ct':
        return a*wt.Ct[kar+1]+(1-a)*wt.Ct[kar]
    if value == 'Cp':
        return a*wt.Cp[kar+1]+(1-a)*wt.Cp[kar]

def move_coord(origin, f, d):
    '''
    moves the vector f, expressed in the original
    set of coordinates to the axis system located
    at origin and rotated of an angle d around the vertical axis
    return the vector in the new coordinates (streamwise, spanwise, vertical)
    '''
    wdr=grad2rad(float(d))
    r=np.array([f[0]-origin[0],f[1]-origin[1],f[2]-origin[2]])
    windDirVec=np.array([np.cos(wdr), np.sin(wdr), 0])
    spanDirVec=np.array([np.sin(wdr), -np.cos(wdr), 0])
    vertDirVec=np.array([0,0,1])
    stream=np.dot(windDirVec,r)
    span=np.dot(spanDirVec,r)
    vert=np.dot(vertDirVec,r)
    return stream, span, vert

def gaussianWake(wt,Ct,f,d):
    '''
    calculates the velocity deficit induced by the turbine wt, which has a thrust coefficient Ct.
    The velocity deficit is returned at the point f, for a wind direction d.
    The velocity deficit is assumed to have a gaussian shape
    '''
    kstar=0.3837*wt.TI+0.003678
    st,sp,ve =move_coord(wt.loc3, f, d)
    rp=m.sqrt(sp**2+ve**2)
    slr=1-Ct/float(8)/np.square(kstar*st/wt.D+epsilon(beta(Ct)))
    if slr >= 0.0:
        C=1-np.sqrt(slr)
        newdef=C*np.exp(-1/2/np.square(kstar*(st)/wt.D+epsilon(beta(Ct)))*np.square((rp)/wt.D))
    if slr < 0.0:
        if abs(st) < abs(sp):
            newdef=0
        if abs(st) > abs(sp):
            newdef=1
            print 'problem, the argument under the square root is negative, even though the point of interest is far enough downstream.'
            print st, sp
    return newdef

def topHat(wt,Ct, f, d, kw):
    '''
    calculates the velocity deficit induced by the turbine wt, which has a thrust coefficient Ct.
    The velocity deficit is returned at the point f, for a wind direction d.
    The velocity deficit is assumed to have a top hat shape
    '''
    mfac=1
    st,sp,ve =move_coord(wt.loc3, f, d)
    rp=m.sqrt(sp**2+ve**2)
    newdef=(1-np.sqrt(1-Ct))/np.square(1+2*kw*st/wt.D)
    newdef=np.where((st>0) & (np.absolute(rp)<=(wt.D+2*kw*np.power(st,mfac))/2), newdef,0)
    return newdef

def cosineWake(wt,Ct,f,d,k0,I0):
    '''
    calculates the velocity deficit induced by the turbine wt, which has a thrust coefficient Ct.
    The velocity deficit is returned at the point f, for a wind direction d.
    The velocity deficit is assumed to have a cosine shape
    '''
    kw=k0*wt.TI/I0
    st,sp,ve =move_coord(wt.loc3, f, d)
    rw=kw*st+wt.D/2
    rp=m.sqrt(sp**2+ve**2)
    B=0
    if rp < rw:
        B=1
    J=(1-np.sqrt(1-Ct))/np.square(1+2*kw*st/wt.D)
    C=m.cos(np.pi*(1+rp/rw))
    newdef=B*J*(1-C)
    return newdef

def Jensen_windFarm(windDir, windSpeed, kw=0.075):
    '''
    computes an estimation of the power output for one single wind direction and one single wind speed
    for small farms, applying quadratic summation and a smaller growth rate may lead to better results.
    '''
    turbines=make_turbines(windDir, shift_origin=False)
    
    remain=turbines[:]
    for wt in turbines:
        remain.pop(0)
        #Vref=float(windSpeed*(1-wt.velDef))
	Vref=windSpeed-wt.velDef
        Ct=interpolate(wt,Vref,'Ct')
        for rm in remain:
            #linear summation is applied because it is supposed to be better in large wind farms
            #rm.velDef += topHat(wt, Ct, rm.loc3, windDir, kw)
            rm.velDef += windSpeed*topHat(wt, Ct, rm.loc3, windDir, kw)
        #print wt.num, " " , windSpeed*(1.0-topHat(wt, Ct, rm.loc3, windDir, kw))   #Carlos
        print wt.num, 100*windSpeed * (1.0 -(1.0-topHat(wt, Ct, rm.loc3, windDir, kw)))
    return turbines

def PA_windFarm(windDir, windSpeed,I0):
    '''
    compute the estimated power of a wind for one single wind direction and one
    single wind speed farm as function of the wind direction and intensity
    This script is based on:
    A new analytical model for wind farm power prediction
    Amin Niayifar and Fernando Porte-Agel
    doi:10.1088/1742-6596/625/1/012039
    '''
    turbines=make_turbines(windDir, shift_origin=False)
    
    for wt in turbines: # initialize the incoming turbulence intensity for all turbines
        wt.TI=I0
    
    '''
    the following loops compute the incoming velocity, as well as the turbulence intensity at each tubine location in the following way:
    the turbines are processed one by one, in the streamwise order. When the turbine i is being resolved, all previously processed turbines (upstream)
    are stored in "resolved", while all non processed turbines (downstream) are stored in "remain". The wake of turbine i influences all
    downstream turbines. all the resolved ones are however kept untouched.
    In order to compute the added turbulence intensity at turbine i, only the turbines in "resolved" are taken into account" and the most influent of those is kept.
    Thus, for each turbine (1st loop) one has to:
    -update the "resolved" and "remain" lists
    -compute the added turbulence intensity at that location by checking all upsteam turbines (3nd loop)
    -redefine the velocity deficit of all downstream turbines(2nd loop)
    '''
    remain=turbines[:]
    resolved=[]
    for wt in turbines:
        remain.pop(0)
        Vref=windSpeed-wt.velDef
        #Vref=float(windSpeed*(1-wt.velDef)) # disable the previous line and enable this line to always use the undisturbed wind velocity as reference when computing the velocity deficit
        Ct=interpolate(wt,Vref,'Ct')
        for rm in remain:
            IplusMax=0
            for re in resolved:
                Vref=windSpeed-re.velDef
                #Vref=float(windSpeed*(1-re.velDef))
                Ctre=interpolate(re,Vref,'Ct')
                st,sp,ve=move_coord(re.loc3, rm.loc3,windDir)
                perpDist=m.sqrt(sp**2+ve**2)
                kstar=0.3837*re.TI+0.003678
                sigma=kstar*st+re.D*epsilon(beta(Ctre))
                R=2*sigma
                r=rm.D/2
                Aw=get_area(R,r,perpDist)
                Iplustest=4*Aw/m.pi/re.D**2*Iplus(re,Ctre,I0,abs(st))
                if Iplustest>IplusMax:
                    IplusMax=Iplustest
                    rm.TI=m.sqrt(I0**2+Iplustest**2)
                # Note: gaussianWake returns the Delta U / Uinf in paper.
                #print "Waked wind speed from PA ", Vref*(1.0-gaussianWake(wt,Ct,rm.loc3,windDir))   #Carlos
            #rm.velDef += gaussianWake(wt,Ct,rm.loc3,windDir)
            rm.velDef += (windSpeed-wt.velDef)*gaussianWake(wt,Ct,rm.loc3,windDir)# disable the previous line and enable this line to always use the undisturbed wind velocity as reference when computing the velocity deficit
        #print "Waked wind speed from PA ", wt.num, " ", Vref*(1.0-gaussianWake(wt,Ct,rm.loc3,windDir))   #Carlos
        #print "Waked wind speed from PA ", wt.num, " ", Vref*(1.0 - rm.velDef/(windSpeed-wt.velDef))   #Carlos
        # print turbine number, undisturbed and disturbed speed
        #print wt.num," ", Vref, " ", Vref*(1.0 - rm.velDef/(windSpeed-wt.velDef))   #Carlos
        # print turbine number, undisturbed and disturbed TI
        print wt.num," ", I0, " ", rm.TI   #Carlos


        resolved.append(wt)
        #print "Waked wind speed ",rm.velDef
        #print "Waked wind speed from PA ", wt.num, " ", Vref*(1.0-gaussianWake(wt,Ct,rm.loc3,windDir))   #Carlos
    return turbines

def cosine_windFarm(windDir, windSpeed,I0,z0):
    '''
    compute the estimated power of a wind for one single wind direction and one
    single wind speed farm as function of the wind direction and intensity
    This script is based on:
    
    '''
    turbines=make_turbines(windDir, shift_origin=False)
    
    kn=0.4
    for wt in turbines:
        wt.TI=I0
    
    remain=turbines[:]
    resolved=[]
    for wt in turbines:
        remain.pop(0)
        k0=0.5/m.log(wt.loc3[2]/z0)
        #k0=0.075
	wt.velDef=windSpeed*np.sqrt(wt.velDef)
        #wt.velDef=np.sqrt(wt.velDef)
	Vref=windSpeed-wt.velDef
        #Vref=float(windSpeed*(1-wt.velDef))
        Ct=interpolate(wt,Vref,'Ct')
        for rm in remain:
            nCoor=move_coord(wt.loc3, rm.loc3, windDir)
            rm.TI += kn*Ct*wt.D/nCoor[0]/100
            rm.velDef += np.square(cosineWake(wt,Ct,rm.loc3,windDir,k0,I0))
        resolved.append(wt)
    return turbines

'''
post processing the power output of each turbines in order to get the averages over the rows of interest
'''

def veldef2power(turbines,windSpeed, rho):
    '''
    compute the total power output of the whole farm , based on the calculated velocity deficit at each turbine location.
    '''
    Ptot=0
    for wt in turbines:
	Vref=windSpeed-wt.velDef
        #Vref=float(windSpeed*(1-wt.velDef)) # disable the previous line and enable this line to always use the undisturbed wind velocity as reference when computing the velocity deficit
	CP=interpolate(wt,Vref,'Cp')
        A=np.pi*(wt.D/2)**2
        Pturbine=CP*(0.5*rho*A*Vref**3)
        Ptot += Pturbine
    return Ptot
