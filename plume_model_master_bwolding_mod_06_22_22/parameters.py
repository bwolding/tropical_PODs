'''
NAME: parameters.py

PURPOSE: To save parameters (constants, plot specifications etc.)

AUTHOR: Fiaz Ahmed

DATE: 02/09/18

'''
import numpy as np
import json
from netCDF4 import Dataset

diri_json="/glade/work/fiaz/analysis/"
fili_json="coords_lonflip.json"
### Read protected locations ###
coords=json.load(open(diri_json+fili_json))

latn=np.float_(coords['lat_n'])
lats=np.float_(coords['lat_s'])
lonw=np.float_(coords['lon_w'])
lone=np.float_(coords['lon_e'])

## Constants ##
Cp=1004. # J/Kg/K
Lv=2.43e6 # J/kg
gravity=9.8 #m/s^2
Tk0 = 273.15 # Reference temperature.
Es0 = 610.7 # Vapor pressure at Tk0.
Lv0 = 2500800 # Latent heat of evaporation at Tk0.
cpv = 1869.4 # Isobaric specific heat capacity of water vapor at tk0.
cl = 4218.0 # Specific heat capacity of liquid water at tk0.
R = 8.3144 # Universal gas constant.
Mw = 0.018015 # Molecular weight of water.
Rv = R/Mw # Gas constant for water vapor.
Ma = 0.028964 # Molecular weight of dry air.
Rd = R/Ma # Gas constant for dry air.
epsilon = Mw/Ma
g = 9.80665
cpd=1004.
Po=1025.
k=2./7

# buoy_wts={}
# buoy_wts['A']=0.25
# buoy_wts['B']=0.125
# buoy_wts['C']=0.235
# buoy_wts['D']=0.265
# buoy_wts['E']=0.125

# A->theta_BL/theta*_LT: 0.228216872136
# B->theta_LT/theta*_LT: 0.11294924436
# C->theta_BL/theta*_MT: 0.247323961071
# D->theta_LT/theta*_MTL 0.279759890392
# E:->theta_MT/theta*_MT: 0.131750032042


def es_calc(temp):


        #get some constants:
        tmelt  = 273.15

        #convert inputs to proper units, forms
        tempc = temp - tmelt # in C
        tempcorig = tempc
        c=np.array((0.6105851e+03,0.4440316e+02,0.1430341e+01,0.2641412e-01,0.2995057e-03,0.2031998e-05,0.6936113e-08,0.2564861e-11,-.3704404e-13))

        #calc. es in hPa (!!!)
        #es = 6.112*EXP(17.67*tempc/(243.5+tempc))
        es=c[0]+tempc*(c[1]+tempc*(c[2]+tempc*(c[3]+tempc*(c[4]+tempc*(c[5]+tempc*(c[6]+tempc*(c[7]+tempc*c[8])))))))
        es = es/100.
        return es
