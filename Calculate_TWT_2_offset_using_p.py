# import modules
import numpy as np
from tabulate import tabulate


dz = np.array([0, 0.125, 0.125, 0.15, 0.3, 0.8, 0.2, 0.3, 0.2, 0.25, 0.25, 0.3, 0.25, 0.75]
              )

vp = np.array([1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800])/1000

p = np.linspace(0, 0.5, 10)

max_offset = 10

def tp_xp(dz, vp, p):
    tp = np.zeros(len(p))
    
    xp = np.zeros(len(p))
    
    for k in range(len(p)):
        pp = p[k]
        for i in range(len(dz)):
            bracket_term_tp = (2*dz[i])/(vp[i]*np.sqrt(abs(1-pp**2*vp[i]**2)))
            bracket_term_xp = (2*dz[i]*vp[i]*pp)/(np.sqrt(abs(1-pp**2*vp[i]**2)))
                
            tp[k]=tp[k]+bracket_term_tp
            xp[k]=xp[k]+bracket_term_xp
            
        if xp[k]>max_offset:
            break
        
    return tp, xp, k+1

#Calculate data
tp, xp, k = tp_xp(dz, vp, p)

tp = tp[:k]
tp = np.round(tp,2)

xp = xp[:k]
xp = np.round(xp,2)

#Prepare data to plot
result_to_plot = np.zeros((len(tp),2))

result_to_plot[:,0] = xp
result_to_plot[:,1] = tp

#Create header
head = ["Offset, km", "Traveltime, sec"]

# display table
print(tabulate(result_to_plot, headers=head, tablefmt="grid"))




