import math
import numpy
import scipy
from scipy import integrate
import time
import json
import pickle 

R = 0.001

f = 600*math.pow(10,6)
waveL = 3.0*math.pow(10,8)/f 
mu = 1
sigma = 6.31*math.pow(10,-12) ##CCA Threshold = -82 dB
i_max_dens = 0
RMIN = 0.001 

NO = 2.401199*math.pow(10,-14) ##For 6 MHz

beta_range_dB = numpy.arange(40.5,60.5,0.5)
beta_range = [math.pow(10,b/10.0) for b in beta_range_dB]
beta_range_new = [math.log(1+b,2) for b in beta_range]

def PSI(x):
    return math.exp(-x*NO)

def PHI(x):
    return 1/(1+x)

def MIN(a,b):
    if (a < b):
        return a
    else:
        return b

def MAX(a,b):
    if (a > b):
        return a
    else:
        return b

def G(a):
    return math.exp(-a)

def pathloss_AP(h1,d):
    R_bp = (1/waveL)*math.sqrt((4*h1**2)**2 - 2*(h1**2)*(waveL**2) + (waveL/2)**4)
    R_bp = R_bp/1000.0
    P_LOS = abs(20*math.log10((waveL**2)/(8*math.pi*(h1**2))))
    if d < R_bp:
        PL = P_LOS + 20 + 25*math.log10(d/R_bp) 
    else:
        PL = P_LOS + 20 + 40*math.log10(d/R_bp)
    PL = min(PL,180) 
    return math.pow(10,-PL/10.0) 

def pathloss_Client(h1,d):
    R_bp = (1/waveL)*math.sqrt((4*h1)**2 - ((h1**2)+1)*(waveL**2) + (waveL/2)**4)
    R_bp = R_bp/1000.0 
    P_LOS = abs(20*math.log10((waveL**2)/(8*math.pi*h1)))
    if d < R_bp:
        PL = P_LOS + 20 + 25*math.log10(d/R_bp)
    else:
        PL = P_LOS + 20 + 40*math.log10(d/R_bp)
    PL = min(PL,180)
    return math.pow(10,-PL/10.0) 

def integral(g,r):
    integr = 0.0
    delta = 5.0*step
    delta2 = delta*delta
    x = -100*delta+r/2
    while x<100.0*delta+r/2:
        y = 0.0
        while y<100.0*delta:
            integr = integr + 2.0*g(x+delta/2.0,y+delta/2,r)*delta2
            y = y+delta
        x = x+delta
    return integr 

def l1(x,y,r):
    x2 = x*x
    y2 = y*y
    dist1 = math.sqrt(x2 + y2)
    dist2 = math.sqrt((x-r)*(x-r) + y2)
    if ((dist1<R) and (dist2<R)):
        return 1.0
    else:
        return 0.0 

def integra(h1,g,r):
    integr = 0.0
    delta = 5.0*step
    delta2 = delta*delta
    x = -100.0*delta + r/2.0
    while x < 100.0*delta + r/2.0:
        y = 0.0
        while y < 100.0*delta:
            integr = integr + 2.0*g(h1,x+delta/2.0,y+delta/2.0,r)*delta
            y = y+delta
        x = x+delta
    return integr 

def f2(h1,x,y,r):
    x2 = x*x
    y2 = y*y
    dist1 = math.sqrt(x2 + y2)
    dist2 = math.sqrt((x-r)*(x-r)+y2)
    mindist = MIN(dist1,dist2)
    rdist = int(round(mindist/step,0))
    if  (rdist>i_max_dens):
        rdist=i_max_dens 
    return dens_acc[rdist]*math.exp(-mu*sigma/(P*pathloss_AP(h1,dist1)))

def f1(h1,x,y,r):
    x2 = x*x
    y2 = y*y
    dist1 = math.sqrt(x2+y2)
    dist2 = math.sqrt((x-r)*(x-r)+y2)
    mindist = MIN(dist1,dist2)
    rdist = int(round(mindist/step,0))
    if  (rdist>i_max_dens):
        rdist=i_max_dens
    return dens_acc[rdist]*math.exp(-mu*sigma/(P*pathloss_AP(h1,dist1)))*math.exp(-mu*sigma/(P*pathloss_AP(h1,dist2)))

def f(h1,r):
    B = integra(h1,f2,r)
    A = B-integra(h1,f1,r)
    ##print 'Pathloss %.15f'%(pathloss_AP(h1,r))
    C = 1.0-math.exp(-mu*sigma/(P*pathloss_AP(h1,r)))
    ##time.sleep(5)
    ##return C*2.0/(1.0/(intensity*B)*((1.0-math.exp(-intensity*B)*C)-1.0/(intensity*B)*(1.0-math.exp(-intensity*B))*(1.0-C)))*1.0/(intensity*A)*((1.0-math.exp(-intensity*(B)))/(intensity*(B))-(1.0-math.exp(-intensity*(B+A)))/(intensity*(B+A)))
    return C*2.0/(1.0/(intensity*B)*((1.0-math.exp(-intensity*B)*C)-1.0/(intensity*B)*(1.0-math.exp(-intensity*B))*(1.0-C)))*1.0/(intensity*A)*((1.0-math.exp(-intensity*(B)))/(intensity*(B))-(1.0-math.exp(-intensity*(B+A)))/(intensity*(B +A)))

def lacc(r):
    B = math.pi*R*R
    A = B - integral(l1,r)
    if r<R:
        C = 0.0
    else:
        C = 1.0
    return C*2.0/(1.0/(intensity*B)*((1.0-math.exp(-intensity*B)*C)-1.0/(intensity*B)*
                                     (1.0-math.exp(-intensity*B))*(1.0-C)))*1.0/(intensity*A)*((1.0-math.exp(-intensity*B))/(intensity*B)-(1.0-math.exp(-intensity*(B+A)))/(intensity*(B+A)))



def P_T(h1):
    cont = 0.0
    y = 0.001
    while y<5000.0*step:
        rdist = int(round(y/step,0))
        if rdist>i_max_dens:
            rdist = i_max_dens
        cont = cont + 2.0*math.pi*y*dens_acc[rdist]*G(mu*sigma/(P*pathloss_AP(h1,y)))*step
        y = y+step
    return (1.0 - math.exp(-intensity*cont))/(intensity*cont) 

def integra2(h1,g,r):
    integr = 0.0
    delta = 5.0*step
    delta2 = delta*delta
    x = -100.0
    while x<100.0*delta:
        y = 0.0
        while y<100.0*delta:
            if (x*x + y*y > r*r):
                integr = integr + 2.0*g(h1,x+delta/2.0,y+delta/2.0,r)*delta2
            y = y+delta
        x = x+delta 
    return integr

def h(h1,x,y,r):
    x2 = x*x
    y2 = y*y
    dist1 = math.sqrt(x2+y2)
    dist2 = math.sqrt((x-r)*(x-r)+y2)
    rdist2 = int(round(dist2/step,0))
    if rdist2>i_max_dens:
        rdist2 = i_max_dens
    return dens_acc[rdist2]*dens[rdist2]*(1.0-PHI(mu*beta*pathloss_AP(h1,dist1)/pathloss_Client(h1,r)))

def psuccess(h1,r):
    return math.exp(-intensity*integra2(h1,h,r))*PSI(mu*beta/(P*pathloss_Client(h1,r))) 


##########New code##################
def integra_n2(h1,g,r):
    integr = 0.0
    delta = 0.2
    delta2 = delta*delta
    x = -20.0
    while x<20.0:
        y = 0.0
        while y<20.0:
            if ((x-r)*(x-r)+(y*y) >= (r*r)):
                integr = integr+2.0*g(h1,x+delta/2.0,y+delta/2.0,r)*delta2
            y += delta
        x += delta
    return integr 

def Pt(h1,r):
    A = integra_n2(h1,f2,r)
    return (1-math.exp(-intensity*A))/(intensity*A) 

#######################################
results = [] 

def compute_triplet(P_in,intensity_in,h1_in,L_in,M_in):
    global P
    P = P_in
    global intensity
    intensity = intensity_in/M_in ##Divide intensity by M number of layers 
    global h1
    h1 = h1_in
    global RMAX 
    RMAX = 2.0/math.sqrt(intensity)
    global step 
    step = RMAX/100.0 
    print 'step = %.5f'%step 
    global dens_acc 
    global dens 
    global p_suc
    dens = [0]*120
    dens_acc = [0]*120
    p_suc = [0]*120

    psuc_near = [[0 for j in range(0,40)] for i in range(0,120)]

    i = 0 
    r = RMIN
    while r<RMAX:
        dens_acc[i] = lacc(r)
        i_max_dens = i
        i += 1
        r = r+step

    print 'First while loop done!' 
    pt = [0]*120
    i = 0 
    r = RMIN
    while r<RMAX:
        ##print 'r = %.5f'%r
        dens[i] = f(h1,r)
        pt[i] = Pt(h1,r) 
        i += 1
        r = r+step

    print 'Second while loop done!'
    
    i = 0
    j = 0
    r = RMIN 
    while r<RMAX:
        j = 0 
        global beta 
        for beta in beta_range:
            psuc_near[i][j] = psuccess(h1,r)
            j = j+1
        r = r+step
        i = i+1
        print '%.2f'%i 

    print 'Third while loop done!'
    
    i = 0
    r = RMIN
    mean = [0]*120
    while r<RMAX:
        mean[i] = scipy.integrate.simps(psuc_near[i],beta_range_new)
        print 'Expected rate at i = %.2f P = %.2f Intensity = %.2f is = %.6f'%(i,P,intensity,mean[i])
        r = r+step
        i+=1

    cdf_sinr = [0]*40
    j = 0 
    for beta in beta_range:
        i = 0
        r = RMIN
        while r<RMAX:
            cdf_sinr[j] = cdf_sinr[j] + (psuc_near[i][j]*2.0*intensity*math.pi*r*math.exp(-intensity*math.pi*r*r)*step)
            r = r+step
            i = i+1
        j = j+1 

    i = 0
    r = RMIN
    ave_rate_1 = 0.0
    ave_rate_2 = 0.0 
    Pt_ = 0.0
    while r<RMAX:
        Pt_ = Pt_ + (pt[i]*2.0*math.pi*r*intensity_in*math.exp(-intensity_in*math.pi*r*r)*step) 
        ave_rate_2 = ave_rate_2 + (pt[i]*mean[i]*G(mu*sigma/(0.1*pathloss_Client(h1,r)))*2.0*intensity_in*math.pi*r*math.exp(-intensity_in*math.pi*r*r)*step) ##With uplink viability factored in 
        ave_rate_1 = ave_rate_1 + (pt[i]*mean[i]*2.0*intensity_in*math.pi*r*math.exp(-intensity_in*math.pi*r*r)*step)
        i += 1
        r += step
    temp = []
    temp.append(P)
    temp.append(intensity)
    temp.append(h1)
    temp.append(Pt_)
    temp.append(ave_rate_1)
    temp.append(ave_rate_2)
    temp.append(cdf_sinr)
    results.append(temp)
    fileName = 'Logs-P-'+str(P)+'-intensity-'+str(intensity_in)+'-L-'+str(L_in)+'-M-'+str(M_in)+'-h-'+str(h1)+'-exhaustivedata.txt'
    with open(fileName,'wb') as myfile:
        pickle.dump(results,myfile)
    print 'Spatial Average of probability of transmission at P = %.2f intensity %.2f and AP Altitude = %.2f is %.5f \n'%(P,intensity,h1,Pt_)
    print 'ASE at P = %.2f intensity %.2f and AP Altitude = %.2f is %.15f \n'%(P,intensity,h1,ave_rate_1)
    print 'ASE_UP at P = %.2f intensity %.2f and AP Altitude = %.2f is %.15f \n'%(P,intensity,h1,ave_rate_2)
    print 'AP Altitude completed = %.2f \n'%h1
    print 'Intensity completed = %.2f \n'%intensity 
    print 'Power completed = %.2f \n'%P
