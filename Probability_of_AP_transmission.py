import math
import numpy
import scipy
from scipy import integrate
import time
import json
import pickle
##from mpl_toolkits.mplot3d import Axes3D
##from matplotlib.ticker import LinearLocator, FormatStrFormatter
##from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.interpolate import interp1d
import itertools
import matplotlib
from scipy import ndimage


marker = itertools.cycle(('o', '.', '^', 'x', 'v', '*','d'))

cmap = plt.get_cmap('jet')
colors = [cmap(i) for i in numpy.linspace(0, 1, 9)]

R = 0.001

f = 600*math.pow(10,6)
waveL = 3.0*math.pow(10,8)/f 
mu = 1
sigma = 6.31*math.pow(10,-12) ##CCA Threshold = -82 dB
i_max_dens = 0
RMIN = 0.001 

##height_range = [1.5,3.0,6.0,9.0,15.0,30.0]
height_range = [1] 

NO = 2.401199*math.pow(10,-14) ##For 6 MHz

beta_range_dB = numpy.arange(0,40.5,0.5)
beta_range = [math.pow(10,b/10.0) for b in beta_range_dB]
beta_range_new = [math.log(1+b,2) for b in beta_range]

##intensity_range = [0.01,0.05,0.1,0.2,0.5,1.0,2.0,5.0,10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0,105.0,110.0,130.0,150.0,
##                   170.0,190.0,200.0]

##intensity_range_plot = list(numpy.arange(0.01,0.05,0.01)) +[0.1,0.2,0.5,1.0,2.0,5.0,10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0,105.0,110.0,130.0,150.0,170.0,190.0,200.0]

power_range = [0.1,0.5,1.0,2.0,4.0]
power_range_plot = numpy.arange(0.1,4.1,0.1) 

fig,ax = plt.subplots(figsize = (4.5,4.5))

def my_legend(axis = None):

    if axis == None:
        axis = plt.gca()

    N = 32
    Nlines = len(axis.lines)
    print Nlines

    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()

    # the 'point of presence' matrix
    pop = numpy.zeros((Nlines, N, N), dtype=numpy.float)    

    for l in range(Nlines):
        # get xy data and scale it to the NxN squares
        xy = axis.lines[l].get_xydata()
        xy = (xy - [xmin,ymin]) / ([xmax-xmin, ymax-ymin]) * N
        xy = xy.astype(numpy.int32)
        # mask stuff outside plot        
        mask = (xy[:,0] >= 0) & (xy[:,0] < N) & (xy[:,1] >= 0) & (xy[:,1] < N)
        xy = xy[mask]
        # add to pop
        for p in xy:
            pop[l][tuple(p)] = 1.0

    # find whitespace, nice place for labels
    ws = 1.0 - (numpy.sum(pop, axis=0) > 0) * 1.0 
    # don't use the borders
    ws[:,0]   = 0
    ws[:,N-1] = 0
    ws[0,:]   = 0  
    ws[N-1,:] = 0  

    # blur the pop's
    for l in range(Nlines):
        pop[l] = ndimage.gaussian_filter(pop[l], sigma=N/5)

    for l in range(Nlines):
        # positive weights for current line, negative weight for others....
        w = -0.3 * numpy.ones(Nlines, dtype=numpy.float)
        w[l] = 0.5

        # calculate a field         
        p = ws + numpy.sum(w[:, numpy.newaxis, numpy.newaxis] * pop, axis=0)
        plt.figure()
        plt.imshow(p, interpolation='nearest')
        plt.title(axis.lines[l].get_label())

        pos = numpy.argmax(p)  # note, argmax flattens the array first 
        best_x, best_y =  (pos / N, pos % N) 
        x = xmin + (xmax-xmin) * best_x / N       
        y = ymin + (ymax-ymin) * best_y / N       


        axis.text(x, y, axis.lines[l].get_label(), 
                  horizontalalignment='center',
                  verticalalignment='center')


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

def PT(h1):   #### Code for LBT!!
    threshold = -62 ## ED threshold for LTE = -62 dBm
    threshold = math.pow(10,threshold/10.0)*math.pow(10,-3)
    cont = 0.0
    y = 0.001
    while y<5000.0*step:
        rdist = int(round(y/step,0))
        if rdist>i_max_dens:
            rdist = i_max_dens
        cont = cont + 2.0*math.pi*y*dens_acc[rdist]*(0.5 - 0.5*scipy.special.erf(10*math.log10(threshold/(P*(r**-3.25)))/(math.sqrt(2)*sigma)))*step
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
results_exhaustive = []
##P = 1.0
intensity = 100.0

for h1 in height_range:
    results = [] 
    ##for intensity in intensity_range:
    for P in power_range: 
        RMAX = 1/math.sqrt(intensity)
        step = RMAX/100.0 
        dens = [0]*120
        dens_acc = [0]*120
        p_suc = [0]*120

        psuc_near = [[0 for j in range(0,81)] for i in range(0,120)]

        i = 0 
        r = RMIN
        while r<RMAX:
            dens_acc[i] = lacc(r)
            i_max_dens = i
            i += 1
            r = r+step

        ##print 'First while loop done!' 

        PT_NEW = PT(h1) 

        results.append(PT_NEW)
        
        ##print 'AP intensity completed = %.2f \n'%intensity
        print 'AP Power completed = %.1f \n'%P
        print 'PT = %.5f \n'%PT_NEW
    ##f = interp1d(intensity_range,results)
    f = interp1d(power_range,results)         
    ##plt.plot(intensity_range_plot,f(intensity_range_plot),label = 'AP Alt. = %.2f m'%h1, color = 'k', marker = marker.next(),ms = 9.5)
    ##plt.plot(power_range_plot,f(power_range_plot),label = 'AP Height = %.1f m'%h1, color = 'k', marker = marker.next(),ms = 9.5)
    plt.plot(power_range,results,label = '$h_{AP}$ = %.1f m'%h1, color = 'k', marker = marker.next())
    print 'AP Height completed = %.2f \n'%h1
    results_exhaustive.append(f(power_range_plot))

##PIK = 'PT_density_10.txt'
##
##with open(PIK,"wb") as myFile:
##    pickle.dump(results_exhaustive,myFile) 


##fig = plt.figure()
##ax = fig.add_subplot(111, projection='3d')
##X, Y = numpy.meshgrid(height_range,intensity_range) 
####ax.plot_wireframe(X, Y, e, rstride=1, cstride=1,color = 'red')
##ax.plot_surface(Y,X,results_exhaustive, cmap=cm.jet, rstride=1, cstride=1)

##ax.set_ylabel('AP Altitude (m)')
##ax.set_xlabel('No. of APs/$km^2$')
##ax.set_zlabel('Probability of AP transmission')

##ax.set_xscale('log')
plt.xlabel('AP Power (in W)',fontsize = 10)
plt.ylabel('Probability of AP Transmission',fontsize = 10)
##plt.xlim([0.01,220])

##matplotlib.rcParams.update({'font.size': 12.5})

##fig.suptitle('$P_{AP} = 1.0W$',fontsize = 16)
fig.suptitle('AP Density = 1/$\mathrm{km}^2$',fontsize = 10)

legend = ax.legend(loc = 'upper right', shadow = True, fontsize=10).draggable() 
##frame = legend.get_frame()
##frame.set_facecolor('0.90')
plt.ylim([0,1])

##box = ax.get_position()
##ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
##plt.legend(bbox_to_anchor=(0, 1), loc='upper right')
##my_legend() 

plt.tick_params(labelsize=10)

plt.grid(True,which = 'both')
plt.show() 

