
'''
PURPOSE: To run the plume model on moisture (specific humidity)
         and temperature inputs and to output plume virtual temp.
                  
AUTHOR: Fiaz Ahmed

DATE: 08/27/19
'''

import numpy as np
from netCDF4 import Dataset
from glob import glob
import datetime as dt
from dateutil.relativedelta import relativedelta
import time
import itertools
from sys import exit
from numpy import dtype
from thermodynamic_functions import *
from parameters import *
from thermo_functions import plume_lifting,calc_qsat,invert_theta_il
from scipy.interpolate import interp1d
import scipy.io as sio

start_time=time.time()

ARM_SITE='Nauru'

#### Set mixing preference ####
# Deep-inflow B mixing like in Holloway & Neelin 2009, JAS; Schiro et al. 2016, JAS
MIX='DIB'       

# No mixing case
# MIX='NOMIX'   

print('READING AND PREPARING FILES')

####### LOAD temp. & sp.hum DATA ########

dir='./ARM_Nauru/'
file_temp='tdry_profiles.mat'
file_hum='q_profiles.mat'
file_pres='pressure_dim_5hPa_interp.mat'
file_cwv='cwv_timeseries_5minavgs_Nauru_01Jan1998_31Dec2008.mat'

### Read temp in K ###
fil=sio.loadmat(dir+file_temp)
temp=fil['tdrycholloworig']

### Read sp.hum in g/kg ###
fil=sio.loadmat(dir+file_hum)
sphum=fil['qnmcholloworig'] 
sphum*=1e-3 ### Convert sp. humidity from g/kg -> kg/kg 

### Read pressure levels ###
fil=sio.loadmat(dir+file_pres)
lev=np.int_(np.squeeze(fil['pressure'])) ## convert from short to int

### Remove some levels with NaNs
ind_fin=np.where(np.isfinite(temp[0,:]))[0]
temp_fin=temp[:,ind_fin]
sphum_fin=sphum[:,ind_fin]
lev_fin=lev[ind_fin]

### Get time dimension ###
time_dim=np.arange(temp.shape[0])

### Obtain indices for mid and starting pressure levels 
i450=np.where(lev_fin==450)[0]
i1000=np.where(lev_fin==1000)[0]
####

## Launch plume from 1000 hPa ##
ind_launch=np.zeros((time_dim.size),dtype='int')
ind_launch[:]=i1000+1
        
### Prescribing mixing coefficients ###
c_mix_DIB = np.zeros((time_dim.size,lev_fin.size)) 
c_mix_NOMIX = np.zeros((time_dim.size,lev_fin.size)) 
     

## Compute Deep Inflow Mass-Flux ##
## The mass-flux (= vertical velocity) is a sine wave from near surface
## to 450 mb. Designed to be 1 at the level of launch
## and 0 above max. w i.e., no mixing in the upper trop.
assert(all(ind_launch>=1))
w_mean = np.sin(np.pi*0.5*(lev_fin[ind_launch-1][:,None]-lev_fin[None,:])/(lev_fin[ind_launch-1][:,None]-lev_fin[i450]))    
minomind = np.where(w_mean==np.nanmax(w_mean))[0][0]
c_mix_DIB = np.zeros((time_dim.size,lev_fin.size)) ## 0 above max. w
c_mix_DIB[:,1:-1]= (w_mean[:,2:] - w_mean[:,:-2])/(w_mean[:,2:]+w_mean[:,:-2])
c_mix_DIB[c_mix_DIB<0]=0.0
c_mix_DIB[np.isinf(c_mix_DIB)]=0.0


### Change data type ####
temp_fin=np.float_(temp_fin)
sphum_fin=np.float_(sphum_fin)

    
### Set output variables ####
temp_plume_DIB=np.zeros_like(temp_fin)    
temp_v_plume_DIB=np.zeros_like(temp_fin)    

temp_plume_NOMIX=np.zeros_like(temp_fin)    
temp_v_plume_NOMIX=np.zeros_like(temp_fin)    

## Launch plume ###
print('DOING DIB PLUME COMPUTATION')
plume_lifting(temp_fin, sphum_fin, temp_v_plume_DIB, temp_plume_DIB, 
c_mix_DIB, lev_fin, ind_launch)

print('DOING NOMIX PLUME COMPUTATION')
plume_lifting(temp_fin, sphum_fin, temp_v_plume_NOMIX, temp_plume_NOMIX, 
c_mix_NOMIX, lev_fin, ind_launch)

#### Optional thermodynamic computations ###

# qsat ##
qsat=qs_calc(lev_fin, temp_fin)
rh=(sphum_fin/qsat)*100.

## env. virtual temp. ###
# temp_v_env = temp_v_calc(temp_fin, hum_fin, 0.) ### Environmental virtual temp.

## thermal buoyancy ####
# buoy=9.8*(temp_v_plume-temp_plume)/(temp_plume)

################################################

print ('SAVING FILE')
fout=dir+'plume_properties_'+ARM_SITE+'.nc'

##### SAVE FILE ######

try:ncfile.close()
except:pass

ncfile = Dataset(fout, mode='w', format='NETCDF4')

ncfile.createDimension('time',None)
ncfile.createDimension('lev',None)

ti = ncfile.createVariable('time',dtype('float32').char,('time'))
lv = ncfile.createVariable('lev',dtype('float32').char,('lev'))

tp_dib = ncfile.createVariable('temp_plume_DIB',dtype('float32').char,('time','lev'),zlib=True)
tp_nmx = ncfile.createVariable('temp_plume_NOMIX',dtype('float32').char,('time','lev'),zlib=True)
tenv = ncfile.createVariable('temp_env',dtype('float32').char,('time','lev'),zlib=True)
qenv = ncfile.createVariable('q_env',dtype('float32').char,('time','lev'),zlib=True)
relh = ncfile.createVariable('rh_env',dtype('float32').char,('time','lev'),zlib=True)

ti[:]=time_dim
lv[:]=lev_fin
tp_dib[:]=temp_plume_DIB
tp_nmx[:]=temp_plume_NOMIX
tenv[:]=temp_fin
qenv[:]=sphum_fin

relh[:]=rh

print ('FILE WRITTEN')
print ('TOOK %.2f MINUTES'%((time.time()-start_time)/60))
