import numpy as np
cimport numpy as np
from libc.math cimport exp,log,pow, sqrt
import cython
# from sys import exit

cdef extern from "math.h":
    bint isfinite(double x)


DTYPE = np.float64
DTYPE1 = np.int64 # Updated by B. Wolding 10/31/2023
ctypedef np.float_t DTYPE_t
ctypedef np.int_t DTYPE1_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
# @cython.cdivision(True)

#### Thermodyanmic constants ###


cdef temp_v_calc (double temp, double q, double ql):

    cdef double r,rl,rt
    cdef double RV=461.5
    cdef double RD=287.04
    cdef double EPS
    cdef double temp_v
    
    EPS=RD/RV

    r = q/(1-q-ql)
    rl =  ql/(1-q-ql)
    rt = r + rl
    
    temp_v = temp * (1 + (r/EPS)) 
    temp_v = temp_v/(1 + rt)    
    
    return temp_v

cdef es_calc_bolton(double temp):
    # in hPa

    cdef double tmelt  = 273.15
    cdef double tempc, es  
    tempc = temp - tmelt 
    es = 6.112*exp(17.67*tempc/(243.5+tempc))
    return es


cdef es_calc(double temp):

    cdef double tmelt  = 273.15
    cdef double tempc,tempcorig 
    cdef double c0,c1,c2,c3,c4,c5,c6,c7,c8
    cdef double es
    
    c0=0.6105851e+03
    c1=0.4440316e+02
    c2=0.1430341e+01
    c3=0.2641412e-01
    c4=0.2995057e-03
    c5=0.2031998e-05
    c6=0.6936113e-08
    c7=0.2564861e-11
    c8=-.3704404e-13

    tempc = temp - tmelt 
    tempcorig = tempc
    
    if tempc < -80:
        # in hPa
        es=es_calc_bolton(temp)
    else:
        # in Pa: convert to hPa
        es=c0+tempc*(c1+tempc*(c2+tempc*(c3+tempc*(c4+tempc*(c5+tempc*(c6+tempc*(c7+tempc*c8)))))))
        es=es/100
    
    return es

cdef esi_calc(double temp):
    cdef double esi
    esi = exp(23.33086 - (6111.72784/temp) + (0.15215 * log(temp)))
    return esi
    
cdef qs_calc(double press_hPa, double temp):

    cdef double tmelt  = 273.15
    cdef double RV=461.5
    cdef double RD=287.04
    cdef double EPS, press, tempc, es, qs

    EPS=RD/RV

    press = press_hPa * 100. 
    tempc = temp - tmelt 

    es=es_calc(temp) 
    es=es * 100. #hPa
    qs = (EPS * es) / (press + ((EPS-1.)*es))
    return qs

cdef qsi_calc(double press_hPa, double temp):

    cdef double tmelt  = 273.15
    cdef double RV=461.5
    cdef double RD=287.04
    cdef double EPS
    cdef double press,esi,qsi
        
    EPS=RD/RV
    press = press_hPa * 100
    esi=esi_calc(temp) 
    esi=esi*100. #hPa 
    qsi = (EPS * esi) / (press + ((EPS-1.)*esi))
    return qsi

cdef theta_v_calc(double press_hPa, double temp,double q,double ql):

    cdef double pref = 100000.
    cdef double tmelt  = 273.15
    cdef double CPD=1005.7
    cdef double  RD=287.04
    cdef double pres, temp_v,theta_v

    press = press_hPa * 100.
    temp_v=temp_v_calc(temp, q, ql)
    theta_v = temp_v * pow((pref/press),(RD/CPD))
    return theta_v

cdef theta_il_calc(double press_hPa,double temp,double q,double ql,double qi):

    cdef double pref = 100000.
    cdef double tmelt  = 273.15
    cdef double CPD=1005.7
    cdef double CPV=1870.0
    cdef double CPVMCL=2320.0
    cdef double RV=461.5
    cdef double RD=287.04
    cdef double EPS=RD/RV
    cdef double ALV0=2.501E6
    cdef double LS=2.834E6
    cdef double press,tempc
    cdef double r, rl, ri, rt
    cdef double ALV, chi, gam
    cdef double theta_il

    press = press_hPa * 100. 
    tempc = temp - tmelt 
    r = q / (1. - q - ql - qi)
    rl =  ql / (1. - q - ql - qi)
    ri =  qi / (1. - q - ql - qi)
    rt = r + rl + ri

    ALV = ALV0 - (CPVMCL * tempc)

    chi = (RD + (rt * RV)) / (CPD + (rt * CPV))
    gam = (rt * RV) / (CPD + (rt * CPV))


    ### Handling zero moisture environments

    if q==0:
        theta_il = temp * pow((pref / press),chi) * pow((1. - ((rl + ri)/(EPS + rt))),chi)*exp(((-ALV)*rl - LS*ri)/(CPD+(rt*CPV))/temp)
    else:
        theta_il = temp * pow((pref / press),chi) * pow((1. - ((rl + ri)/(EPS + rt))),chi)*pow((1. - ((rl + ri)/rt)),-gam)*exp(((-ALV)*rl - LS*ri)/(CPD+(rt*CPV))/temp)


    return theta_il

cdef theta_e_calc (double press_hPa,double temp, double q):

    cdef double pref = 100000.
    cdef double tmelt  = 273.15
    cdef double CPD=1005.7
    cdef double CPV=1870.0
    cdef double CPVMCL=2320.0
    cdef double RV=461.5
    cdef double RD=287.04
    cdef double EPS=RD/RV
    cdef double ALV0=2.501E6
    cdef double press, tempc,theta_e
    cdef double r,ev_hPa, TL, chi_e
    
    press = press_hPa * 100. # in Pa
    tempc = temp - tmelt # in C

    r = q / (1. - q)

    # get ev in hPa 
    ev_hPa = press_hPa * r / (EPS + r)

    #get TL
    TL = (2840. / ((3.5*log(temp)) - (log(ev_hPa)) - 4.805)) + 55.

    #calc chi_e:
    chi_e = 0.2854 * (1. - (0.28*r))

    theta_e = temp * pow((pref / press),chi_e) * exp(((3.376/TL) - 0.00254) * r * 1000. * (1. + (0.81 * r)))
    return theta_e

# cdef theta_calc(double press_hPa, double temp, double theta):
#     cdef double pref = 100000.
#     cdef double CPD=1005.7
#     cdef double RD=287.04
#     cdef double press
#     press = press_hPa * 100. 
# 
#     theta = temp * pow((pref/press),(RD/CPD))

cdef theta_l_calc(double press_hPa, double temp,double q,double ql):
    cdef double pref = 100000.
    cdef double tmelt  = 273.15
    cdef double CPD=1005.7
    cdef double CPV=1870.0
    cdef double CPVMCL=2320.0
    cdef double RV=461.5
    cdef double RD=287.04
    cdef double EPS=RD/RV
    cdef double ALV0=2.501E6
    cdef double press,tempc,theta_l
    cdef double r,rl,rt
    cdef double ALV,chi, gamma
    
    press = press_hPa * 100.
    tempc = temp - tmelt 
    r = q / (1. - q - ql)
    rl =  ql / (1. - q - ql)
    rt = r + rl

    ALV = ALV0 - (CPVMCL * tempc)

    chi = (RD + (rt * RV)) / (CPD + (rt * CPV))
    gam = (rt * RV) / (CPD + (rt * CPV))

#     print 'TEMP:',temp
#     print 'PRESS:',press
#     print 'EPS+rt:',EPS+rt
#     print 'q,ql:',q,ql
#     print 'rt:',rt
#     print 'denom:',(CPD+(rt*CPV))*temp
    theta_l = temp * pow((pref / press),chi) * pow((1. - (rl/(EPS + rt))),chi)*pow((1. - (rl/rt)),(-gam))*exp((-ALV)*rl/((CPD+(rt*CPV))*temp))
    return theta_l

cdef temp_calc(double press_hPa,double theta_l,double qt):

    ## Only designed to work on single elements for now ##
    ## get some constants ##
    cdef double pref = 100000.
    cdef double tmelt  = 273.15
    cdef double CPD=1005.7
    cdef double CPVMCL=2320.0
    cdef double RV=461.5
    cdef double RD=287.04
    cdef double EPS=RD/RV
    cdef double ALV0=2.501E6
    cdef double pres, rs, rl, theta_l_new, temp
    cdef double rt, TL, qs, ql,q,x1, diff_tl
    cdef double deltaT=999.0

    press = press_hPa * 100. 
    rt = qt / (1. - qt)

#     #get TL, tentative T, rs, ql
    TL = theta_l * pow((press / pref),(RD/CPD))
    qs=qs_calc(press_hPa, TL)

    if qt>qs:
        ql=qt-qs
    else:
        ql=0
        
    rs = qs / (1. - qs - ql)
    
    if rt>rs:
        rl=rt-rs
    else:
        rl=0

    temp = TL
    
#     ## Iterative procedure to get T

    while (abs(deltaT) >= 1e-3):
        ALV= ALV0
        x1=ALV/(CPD*temp)
        deltaT = (temp - TL*(1.+x1*rl))/(1. + x1*TL*(rl/temp+(1.+rs*EPS)*rs*ALV/(RV*temp*temp)))
        temp = temp -deltaT
        qs=qs_calc(press_hPa, temp)
        
        if qt>qs:
            ql = qt - qs  
        else:
            ql=0
        
        if qt<qs:
            q=qt
        else:
            q=qs

        rs = qs/(1. - qs - ql)
        
        if rt>rs:
            rl=rt-rs
        else:
            rl=0

#     ##do additional iteration to make sure final theta_l
#     ##agrees with the initial theta_l

    theta_l_new=theta_l_calc(press_hPa,temp, q,ql)
    diff_tl = theta_l_new - theta_l

    while(abs(diff_tl) > 0.05):
        if (abs(diff_tl) < 1.):
            deltaT = 0.001 * diff_tl/(abs(diff_tl))
        else:
            deltaT = diff_tl/3.

        temp=temp-deltaT
        qs=qs_calc(press_hPa, temp)
        
        if qt>qs:
            ql=qt-qs
        else:
            ql=0.0
        
        if qt<qs:
            q=qt
        else:
            q=qs

        theta_l_new=theta_l_calc(press_hPa,temp, q,ql)
        diff_tl = theta_l_new - theta_l
        
    return temp

cdef temp_i_calc(double press_hPa, double theta_il, double qt):
#     ## Get temperature from theta_il and all condensed matter in qt is ice ##

    cdef double pref = 100000.
    cdef double tmelt  = 273.15
    cdef double CPD=1005.7
    cdef double CPVMCL=2320.0
    cdef double RV=461.5
    cdef double RD=287.04
    cdef double EPS=RD/RV
    cdef double ALV0=2.501E6
    cdef double LS=2.834E6
    cdef double press,rt,TL,theta_il_new
    cdef double qs,rs,qi,ri,x1,diff_tl
    cdef double deltaT=999.0
    cdef double temp
    
    qs=0.0
    theta_il_new=0.0

#     #convert inputs to proper units, forms 
    press = press_hPa * 100. 
    rt = qt / (1. - qt)

#     #get TL, tentative T, rs, qi
    TL = theta_il * ((press / pref)**(RD/CPD))
    qs=qsi_calc(press_hPa, TL)
    
    if qt>qs:
        qi=qt-qs
    else:
        qi=0.0

    rs = qs / (1. - qs - qi)
     
    if rt>rs:
        ri=rt-rs
    else:
        ri=0.0
    
    temp = TL

    while (abs(deltaT) >= 1e-3):
        x1=LS/(CPD*temp)
        deltaT = (temp - TL*(1.+x1*ri))/(1. + x1*TL*(ri/temp+(1.+rs*EPS)*rs*LS/(RV*temp*temp)))
        temp = temp - deltaT
        qs=qsi_calc(press_hPa, temp)
        
        if qt>qs:
            qi=qt-qs
        else:
            qi=0.0
        
        if qt<qs:
            q=qt
        else:
            q=qs

        rs = qs / (1. - qs - qi)
        ri = rt - rs if rt>rs else 0

#     # do additional iteration to make sure final theta_il
#     # agrees with the initial theta_il

    theta_il_new=theta_il_calc(press_hPa,temp,q,0.0,qi)
    
    diff_tl = theta_il_new - theta_il
    while(abs(diff_tl) >= 0.05):
            if (abs(diff_tl) < 1.):
                deltaT = 0.001 * diff_tl/(abs(diff_tl))
            else:
                deltaT = diff_tl/3.

            temp=temp-deltaT
            qs=qsi_calc(press_hPa, temp)
            
            if qt >qs:
                qi = qt - qs  
            else:
                qi=0.0
            
            if qt<qs:
                q=qt
            else:
                q=qs
                
            theta_il_new=theta_il_calc(press_hPa,temp,q,0.0,qi)
            diff_tl = theta_il_new - theta_il
    
    return temp


def plume_lifting_bwolding(np.ndarray[DTYPE_t, ndim=2] temp_env,
np.ndarray[DTYPE_t, ndim=2] q_env,
np.ndarray[DTYPE_t, ndim=2] temp_v_plume,
np.ndarray[DTYPE_t, ndim=2] temp_plume,
np.ndarray[DTYPE_t, ndim=2] c_mix,
np.ndarray[DTYPE1_t, ndim=1] pres, 
np.ndarray[DTYPE1_t, ndim=1] ind_init):

    ### Only input is environmental T,q and lifting level -- Everything else computed in-house ##

    cdef unsigned int time_size = temp_env.shape[0]        
    cdef unsigned int height_size = temp_env.shape[1]            
    cdef unsigned int freeze
    
    cdef np.ndarray[DTYPE_t, ndim=1] theta_il_env= np.zeros(height_size)
    cdef np.ndarray[DTYPE_t, ndim=1] theta_il_plume= np.zeros(height_size)
    cdef np.ndarray[DTYPE_t, ndim=1] qt_plume= np.zeros(height_size)

    cdef double q_plume
    cdef double ql_plume
    cdef double theta_v_plume
    cdef double theta_e_plume
    cdef double NAN
    NAN = float("NaN")

    cdef double qi_plume, qs_plume
    cdef double alpha, bet, b1, b2, LV0, LF0
    cdef double qs_fr, q_fr, ql_fr , q2_fr, qi_fr, qsi_fr
    cdef double rl_fr, rs_fr, rt_fr
    cdef double a_fr, b_fr, c_fr,deltaTplus
    cdef double q1, ql1, qs1
    cdef double x
    
    cdef Py_ssize_t i,j 
    
    cdef double tmelt  = 273.15
    cdef double CPD=1005.7
    cdef double  CL=4190.0
    cdef double RV=461.5
        
        
    for j in range(time_size):
        
        for i in range(0,height_size):  
            theta_il_env[i]=theta_il_calc(pres[i],temp_env[j,i],q_env[j,i],0.0,0.0)
        
        theta_il_plume[ind_init[j]]=theta_il_env[ind_init[j]]
        qt_plume[ind_init[j]]=q_env[j,ind_init[j]]
        temp_plume[j,ind_init[j]]=temp_env[j,ind_init[j]]
        temp_v_plume[j,ind_init[j]]=temp_v_calc(temp_env[j,ind_init[j]], q_env[j,ind_init[j]], 0.) ### Set plume environmental virtual temp. equal to environmental virtual temperature at laucnh level # Added by B. Wolding 06/04/2021
    
        freeze = 0
        
        #print('Here 1')
        
        for i in range(ind_init[j]+1,height_size):
        
            ## Mix the liquid water potential temperature and the total water ##
            
            if c_mix[j,i]>0: # This section has been modified from original code by B. Wolding on 06/22/2022 in order to make local mixing instantatneous. In the original version, the plume properties at a given level were determined by mixing occuring at previous level, not mixing  occuring at current level. Essentially mixing occured from previous level, plume thermodynamics were diagnosed at current level, and then mixing at current level occured. In other words, plume properties at the current level were not impacted by mixing at the current level. This will bias CAPE high relative to if mixing occured locally (at current level) and instantaneuosly and then plume properties were diagnosed. In this version, plume mixing happens locally (at current level) and instantaenously. Here the definition of c_mix is such that c_mix(n) indicates the mixing of environmental air at level "n" into the plume.
                
                #print('Here 2')
                theta_il_plume[i] = (theta_il_plume[i-1] * (1.-c_mix[j,i])) + (theta_il_env[i] * c_mix[j,i]) # Modified by B. Wolding on 06/22/2022 
                qt_plume[i] = (qt_plume[i-1] * (1.-c_mix[j,i])) + (q_env[j,i] * c_mix[j,i])
                #print('Here 3')
            else: 
                #print('Here 4')
                theta_il_plume[i] = theta_il_plume[i-1]
                qt_plume[i] = qt_plume[i-1]
                #print('Here 5')
       
            if (isfinite(theta_il_plume[i]) & isfinite(qt_plume[i])):
            
                if (freeze==0):
                    
                    temp_plume[j,i]=temp_calc(pres[i], theta_il_plume[i], qt_plume[i])
                    
                    if (temp_plume[j,i]<=tmelt):
                        ## Turning off freezing ##
                        # convert liquid water to ice in one (irreversible) step (Eman. 1994, p. 139):
                        LF0 = 0.3337E6
                        LV0 = 2.501E6
                        qs_fr=qs_calc(pres[i], temp_plume[j,i])
                        
                        if qt_plume[i] < qs_fr:
                            q_fr=qt_plume[i]
                        else:
                            q_fr=qs_fr
                        
                        ql_fr = qt_plume[i] - q_fr
                        rl_fr = ql_fr/(1. - (q_fr ) - (ql_fr ))
                        rs_fr = qs_fr/(1. - (qs_fr))
                        rt_fr = rl_fr + rs_fr
                        alpha = 0.009705  ## linearized e#/e* around 0C (to -1C)
                        b1=esi_calc(temp_plume[j,i])
                        b2=es_calc(temp_plume[j,i])
                        bet = b1 / b2
                        a_fr = (LV0 + LF0) * alpha * LV0 * rs_fr / (RV *pow(temp_plume[j,i],2))
                        b_fr = CPD + (CL * rt_fr) + (alpha * (LV0 + LF0) * rs_fr) +(bet * (LV0 + LF0) * LV0 * rs_fr /(RV *pow(temp_plume[j,i],2)))
                        c_fr = ((-1.) * LV0 * rs_fr) - (LF0 * rt_fr) + (bet * (LV0 + LF0) * rs_fr)
                        deltaTplus = (-b_fr + sqrt(pow(b_fr,2) - (4 * a_fr * c_fr))) / (2 * a_fr)
                        temp_plume[j,i] = temp_plume[j,i] + deltaTplus
                        
                        qsi_fr=qsi_calc(pres[i], temp_plume[j,i])
                        qs_plume = qsi_fr
                        
                        if qt_plume[i]<qsi_fr:
                            q_plume = qt_plume[i]  
                        else:
                            q_plume=qsi_fr
                            
                        q2_fr = q_plume
                        qi_fr = qt_plume[i] - q2_fr
                        qi_plume = qi_fr
                        theta_il_plume[i]=theta_il_calc(pres[i], temp_plume[j,i], q2_fr, 0., qi_fr)
                        temp_v_plume[j,i]=temp_v_calc(temp_plume[j,i], q2_fr, qi_fr)
                        freeze = 1

                    else:

                        ## calc. other values with liquid, as usual
                        qs1 = qs_calc(pres[i], temp_plume[j,i])
                        qs_plume = qs1
                        
                        if qt_plume[i]<qs1:
                            q_plume=qt_plume[i]
                        else:
                            q_plume=qs1
                               
                        q1 = q_plume
                        ql1 = qt_plume[i] - q1
                        ql_plume = ql1
                        
                        if ql1>.001:   # Top out the liquid water content at 1g/kg

                            qt_plume[i]=qt_plume[i]-ql1+.001
                            theta_il_plume[i]=theta_il_calc(pres[i], temp_plume[j,i], q1, 0.001, 0.)
                            ql1=.001

                        temp_v_plume[j,i]=temp_v_calc(temp_plume[j,i], q1, ql1)

                        #!!!!!!!!! RAINING OUT LIQUID WATER !!!!!!!!#                            
#                         if ql1>0:
#                             qt_plume[i]=qt_plume[i]-ql1
#                             theta_il_plume[i]=theta_il_calc(pres[i], temp_plume[j,i], q1, 0., 0.)
                        #!!!!!!!!! REMOVING WATER LOADING IN TEMP_V COMPUTATION !!!!!!!!#                            
#                         temp_v_plume[j,i]=temp_v_calc(temp_plume[j,i], q1, 0.0)
                        #!!!!!!!!! REMOVING WATER LOADING IN TEMP_V COMPUTATION !!!!!!!!# 

                else:
                
                    #continue adiabatic ascent with all additional condensation as ice
                    temp_plume[j,i]=temp_i_calc(pres[i], theta_il_plume[i], qt_plume[i])
                    
                    ##calc. other values
                    qsi_fr=qsi_calc(pres[i], temp_plume[j,i])
                    qs_plume = qsi_fr
                
                    if qt_plume[i]<qsi_fr:
                        q_plume = qt_plume[i] 
                    else:
                        q_plume=qsi_fr
                
                    q2_fr = q_plume
                    qi_fr = qt_plume[i] - q2_fr
                    qi_plume = qi_fr
                    
                    if qi_fr>.001:   # Top out the ice content at 1g/kg

                        qt_plume[i]=qt_plume[i]-qi_fr+.001
                        theta_il_plume[i]=theta_il_calc(pres[i], temp_plume[j,i], q2_fr, 0.0, 0.001)
                        qi_fr=.001
                                        
                    temp_v_plume[j,i]=temp_v_calc(temp_plume[j,i], q2_fr, qi_fr)

def invert_theta_il(np.ndarray[DTYPE_t, ndim=2] theta_il_plume,
np.ndarray[DTYPE_t, ndim=2] qt_il_plume,
np.ndarray[DTYPE_t, ndim=2] temp_plume,
np.ndarray[DTYPE1_t, ndim=1] pres):

    cdef unsigned int time_size = theta_il_plume.shape[0]        
    cdef unsigned int height_size = theta_il_plume.shape[1]            
    cdef double NAN
    NAN = float("NaN")    
    cdef Py_ssize_t i,j 
                
    for j in range(time_size):
        for i in range(0,height_size): 
            if (theta_il_plume[j,i]>0):
                temp_plume[j,i]=temp_calc(pres[i], theta_il_plume[j,i], qt_il_plume[j,i])

                    
def calc_qsat(np.ndarray[DTYPE_t, ndim=2] temp_env,
np.ndarray[DTYPE_t, ndim=2] qs_env,
np.ndarray[DTYPE1_t, ndim=1] pres):

    cdef unsigned int time_size = temp_env.shape[0]        
    cdef unsigned int height_size = temp_env.shape[1]            
    cdef unsigned int freeze

    cdef double NAN
    NAN = float("NaN")
    
    cdef Py_ssize_t i,j 

    cdef double tmelt  = 273.15
    cdef double CPD=1005.7
    cdef double  CL=4190.0
    cdef double RV=461.5
        
    for j in range(time_size):

        for i in range(height_size):

            if (temp_env[j,i]>tmelt):
                ## Turning off freezing ##
                # convert liquid water to ice in one (irreversible) step (Eman. 1994, p. 139):
                LF0 = 0.3337E6
                LV0 = 2.501E6
                qs_env[j,i]=qs_calc(pres[i], temp_env[j,i])
                

            else:
    
                qs_env[j,i]=qsi_calc(pres[i], temp_env[j,i])

    

