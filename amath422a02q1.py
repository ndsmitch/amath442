"""
Created on Fri Sep 27 10:08:25 2019

@author: ndsmitch
"""
from decimal import Decimal
import math
import matplotlib.pyplot as plt

c = Decimal(0.8)  # For all tests

# Linear Advection Methods with Periodic BCs
# un = solution at time n
# n = total time steps
# xn = range(n)

def Gd(un, n, xn):  # Godunov
    return[un[i] + c*(un[(i-1)%n] - un[i])
           for i in xn]

def LF(un, n, xn):  # Lax-Friedrichs
    return [((1+c)/2)*un[(i-1)%n] + ((1-c)/2)*un[(i+1)%n]
            for i in xn]

def LW(un, n, xn):  # Lax-Wendroff
    return [(c*(c+1)/2)*un[(i-1)%n] + (1-c*c)*un[i] - (c*(1-c)/2)*un[(i+1)%n]
            for i in xn]

def BW(un, n, xn):  # Beam-Warming
    return [(c*(c-1)/2)*un[(i-2)%n] + (c*(2-c))*un[(i-1)%n] + ((c-1)*(c-2)/2)*un[i]
            for i in xn]
    
# Burgers Methods with Fixed BCs
# un = solution at time n
# xn = range(total_time_steps)
    
def BGd(un, xn):  # Godunov
    def F(x,y):  # Godunov flux for Burgers equation
        if x >= y:
            z = x if x+y>0 else y
        else:
            z = x if x>0 else y if y<0 else 0
        return z*z

    result = [un[i] + (c/4)*(F(un[i-1], un[i]) - F(un[i], un[i+1]))
              for i in xn]
    result.append(Decimal(0))
    result.insert(0, Decimal(-0.5))
    return result
    
def BLF(un, xn):  # Lax-Friedrich
    result = [
        Decimal(0.5)*(un[i-1] + un[i+1]) +
        (c/4)*(un[i-1]*un[i-1] - un[i+1]*un[i+1])
        for i in xn]
    result.append(Decimal(0))
    result.insert(0, Decimal(-0.5))
    return result


###############################################################################
# Test 1&2 - Linear Advection Equation
###############################################################################

# Spatial domain -1<x<1, 101 pts
N = 101 # x-pts
x0 = [x / (N-1) for x in range(-N+1, N, 2)]
xN = range(N)

# Initial conditions
u0_test1 = [Decimal(math.exp(-8*x*x)) for x in x0]
u0_test2 = [Decimal(1 if -0.4<=x<=0.4 else 0) for x in x0]
    
def subplot(ax, x, results, title, comparison):
    ax.plot(x, comparison, label="exact", c="black")
    ax.plot(x, results, label="approximate", linestyle="--", c="black")
    ax.set_title(title)
    
    #ax.set_xlabel("x")
    #ax.set_ylabel("u")

# Initial condition loop for test 1 and 2
for u0 in (u0_test1, u0_test2):

    # Outer loop - time
    tN = range(12500)
    unGd = unLF = unLW = unBW = u0 # copy initial solution
    for n in tN:  # Step through each method once per time step
        unGd = Gd(unGd, N, xN)
        unLF = LF(unLF, N, xN)
        unLW = LW(unLW, N, xN)
        unBW = BW(unBW, N, xN)
    
        # Plot solutions at time t={2,20,200} units
        if n in (124, 1261, 12499):
            time = {124: 2, 1261: 20.176, 12499: 200}
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(5,5))
            subplot(ax1, x0, unGd, "Godunov", u0)
            subplot(ax2, x0, unLF, "Lax-Friedrichs", u0)
            subplot(ax3, x0, unLW, "Lax-Wendroff", u0)
            subplot(ax4, x0, unBW, "Beam-Warming", u0)
            ax1.set_ylabel("u")
            ax3.set_ylabel("u")
            ax3.set_xlabel("x")
            ax4.set_xlabel("x")
            fig.suptitle('Linear Advection: Time t={}'.format(time[n]), fontsize=16)
            plt.subplots_adjust(hspace=0.275, wspace=0.275)  # fix label overlap
            fig.savefig('temp.png', dpi=fig.dpi)
            plt.show()
            
###############################################################################
# Test 3 - Inviscid Burgers Equation
###############################################################################

# Spatial domain 0<x<1.5, 101 pts
N = 76 # x-pts
x0 = [x / 100 for x in range(0, 2*N-1, 2)]
xN = range(N)[1:-1]  # crop ends for fixed BCs

# Initial conditions
u0 = [Decimal(-0.5 if x<=0.5 else 1 if x <= 1 else 0) for x in x0]

# Exact solution for plotting comparisons
def soln(x):
    if x <= 0.25:
        return -0.5
    if x <= 1:
        return 2*x - 1
    if x<= 1.25:
        return 1
    else:
        return 0
u_0p5 = [Decimal(soln(x)) for x in x0]

# Outer loop - time
tN = range(32)
unGd = unLF = u0 # copy initial solution
for n in tN:  # Step through each method once per time step
    unGd = BGd(unGd, xN)
    unLF = BLF(unLF, xN)
  
# Plot solutions at final time
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(5,2.5))
subplot(ax1, x0, unGd, "Godunov", u_0p5)
subplot(ax2, x0, unLF, "Lax-Friedrichs", u_0p5)
fig.suptitle('Burgers Equation: Time t=0.5',fontsize=16)
plt.subplots_adjust(top=0.8)  # fix subtitle/suptitle overlap
plt.show()
        
###############################################################################
# Test 4 - Linear Advection with Varying Spatial Discretization
###############################################################################

# Outer Test Loop - Number of Points
for N in (51, 101, 201, 401):
    x0 = [x / (N-1) for x in range(-N+1, N, 2)]
    xN = range(N)

    # Initial conditions - same as test 1&2
    u0_test1 = [Decimal(math.exp(-8*x*x)) for x in x0]
    u0_test2 = [Decimal(1 if -0.4<=x<=0.4 else 0) for x in x0]
    
    # Initial condition loop for continuous/discontinuous
    for u0 in (u0_test1, u0_test2):

        # Outer loop - time [1 unit]       
        tN = range(int(math.ceil((N-1)/0.8 + 1)))  # x=0.8 => tN = (N-1)/0.8 + 1
        unLF = unLW = u0 # copy initial solution
        
        for n in tN: # Step through each method once per time step
            unLF = LF(unLF, N, xN)
            unLW = LW(unLW, N, xN)
    
        # Plot solutions at final time
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(5,2.5))
        subplot(ax1, x0, unLF, "Lax-Friedrichs", u0)
        subplot(ax2, x0, unLW, "Lax-Wendroff", u0)
        fig.suptitle('Linear Advection: {}pts'.format(N),
                             fontsize=16)
        plt.subplots_adjust(top=0.8, wspace=0.3)  # fix subtitle/suptitle overlap
        plt.show()
