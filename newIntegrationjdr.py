# import numpy as np
# import mpmath as mp
# from phase1 import diffusionConstants, mobility, srhLifetime

q=1.6e-19
a =5.2e-2
a*=6.15
E = (2.67*3.4) + 0.87
bandGap = 1.58956743543344983e-18

xj = 1e-8
l=3
flux = 2.97e7
photonFlux=2e11
hv=2.4e-14
perimittivity = 8.9
absolutePermittivity = perimittivity * 8.85e-12
maxMobilityElectrons = 1265
lifetimeElectron = 1e-8



# def ElecFieldAsXLoc(x, nSide, pSide, iSide, xn, W, xp):
#     x -= xn
#     if x < 0:
#         return -((-q*nSide)/absolutePermittivity)*(x+xn)
#     if x >= 0 and x <= (W):
#         return -((((-q*nSide)/absolutePermittivity)*(xn))-(((q*iSide)/absolutePermittivity)*x))

#     if x > (W):
#         return -(((q*pSide)/absolutePermittivity)*(x-(W+xp)))
    


# N = 1000  # or more if you want better resolution
# L = xn+xp+W # cm
# h=1e-8
# x = np.linspace(L - h, h, N)  # goes from right to left

# h = x[1] - x[0]



# def genDevice(x, width, hv, photonFlux, E, a):
#     def Q(x):
#         distance = x  # Now using absolute position directly
#         gen = (((hv*photonFlux)/E))*mp.exp(-(a*distance))
#         return gen
    
#     # Integrate generation over the entire device width

#     return Q(x)

# A = np.zeros((N, N))
# b = np.zeros(N)

# E_array = np.array([ElecFieldAsXLoc(xi, nSide, pSide, iSide, xn, W, xp) for xi in x])
# G_array = np.array([genDevice(xi, L, hv, photonFlux, bandGap, a) for xi in x])

# print(E_array)
# print(G_array)

# D = diffusionConstants(300, maxMobilityElectrons)

# for i in range(1, N - 1):
#     A[i, i]     = -(maxMobilityElectrons * E_array[i]) / h - 1 / lifetimeElectron
#     A[i, i+1]   = (maxMobilityElectrons * E_array[i+1]) / h
#     b[i]        = G_array[i]

 

# A[0, 0] = 1
# b[0] = 0

# A[N - 1, N - 1] = 1
# b[N - 1] = 0  # or known n at x = 0      # trim b too



    


# n = np.linalg.solve(A, b)

# lastCurrent = n[-2]

# print(n)

# #Finding current


# recombinationRefValues = {
#     'radiativeRecombinationCoefficient': 1.1e-8, #cm3s-1
# }

# def radiativeRecombintionMaj(concMinority, concMajority):
#     radiativeRecombinationCoefficient = recombinationRefValues['radiativeRecombinationCoefficient']
#     lifetimeMajority = 1/(radiativeRecombinationCoefficient*concMinority)

#     return lifetimeMajority

# refAuger = {
#     'simData':1e-30,
#     'direct':3.5e-34,
#     'indirect':2e-30
# }


# def augerLifetimeMaj(concMinority, concMajority): #Majority or minority
#     augerCoefficient = refAuger['simData']
#     augerMinority = (concMajority/(augerCoefficient*(((concMajority**2)*concMinority)+((concMajority**2)*concMinority))))
#     return augerMinority

# def bulkRecombinationLifetimeMaj(concMinority,concMajority, type):
#     radiative = radiativeRecombintionMaj(concMinority, concMajority)
#     auger = augerLifetimeMaj(concMinority, concMajority)
#     srh = srhLifetime()


#     if type == 'n':
#         #For electron
#         # minorityTotalBulkLifetime = 1/((1/radiative)+(1/auger)+(1/srh[0]))
#         majorityTotalBulkLifetime = 1/((1/radiative)+(1/auger)+(1/srh[0]))

#         return [majorityTotalBulkLifetime, radiative, auger, srh[0]]
    
#     if type == 'p':
#         #For hole
#         majorityTotalBulkLifetime = 1/((1/radiative)+(1/auger)+(1/srh[1]))


#         return [majorityTotalBulkLifetime, radiative, auger, srh[1]]

# #Testing lifetimes
# print("Lifetime 1e15 N:"+str(bulkRecombinationLifetimeMaj(1e15, 1e15,'n')))
# print("Lifetime 1e15 P:"+str(bulkRecombinationLifetimeMaj(1e15, 1e15,'p')))

# def totalMajorityLength(temp, typeOfMajority, majorityConc):
#     majority = majorityConc
#     minority = (1.6e-10**2)/majority
#     diffusion = diffusionConstants(temp, mobility(typeOfMajority,(majority),temp))
#     totalMajorityLifetimeVal = bulkRecombinationLifetimeMaj(minority,majority,typeOfMajority)
#     totalLength = mp.sqrt(diffusion*totalMajorityLifetimeVal[0])

#     return totalLength

# # print("Diffusion length : " + str(totalMajorityLength(300, 'n', nSide)))

# def diffusionCurrent(temp, typeOfMajority, majorityConc, injection):
    
#     diffusion = diffusionConstants(temp, mobility(typeOfMajority, majorityConc, 300))
#     # print(diffusion)
#     diffusionLength = totalMajorityLength(temp,  typeOfMajority, majorityConc)
#     # print(diffusionLength)

#     val1 = (injection)
#     # print(val1)
#     val2 = val1/diffusionLength

#     return (q*diffusion*val2)


# lastElectricField = E_array[-1]
# lastCurrent = lastCurrent
# print(lastCurrent, lastElectricField)

# # finalCurrent = -maxMobilityElectrons*q*lastCurrent*lastElectricField + diffusionCurrent(300, 'n', 1e18, lastCurrent)
# # print(finalCurrent)

# D = diffusionConstants(300, maxMobilityElectrons)
# tau = lifetimeElectron
# G_edge = float(G_array[-2])
# E_edge = float(E_array[-2])

# E_safe = max(abs(E_edge), 1e1)

# peak_index = np.argmax(n)
# E_peak = E_array[peak_index]
# n_peak = n[peak_index]




# # drift_term = q * maxMobilityElectrons * lastCurrent * E_safe
# # diff_term = q * (D / (maxMobilityElectrons * E_safe)) * (G_edge - (lastCurrent / lifetimeElectron))

# # finalCurrent = drift_term + diff_term

# # print(finalCurrent)

# index = -2
# Dn = diffusionConstants(300, maxMobilityElectrons)
# E_here = E_array[index]
# n_here = lastCurrent
# dn_dx = (G_edge - (lastCurrent / lifetimeElectron))


# # J_drift = q * maxMobilityElectrons * n_here * E_here
# # J_diff = q * Dn * dn_dx

# # print("Drift =", J_drift)
# # print("Diff =", J_diff)
# # print("Total Current =", -J_drift +J_diff)

# # Jn = []


# print(E_array[-1])
# print(lastCurrent)


# finalCurrentElectron = maxMobilityElectrons*q*lastCurrent*E_array[-1] - diffusionCurrent(300, 'n', nSide, lastCurrent)
# print(finalCurrentElectron)


# Finite Difference Solution to:
# 0 = G(x) - n/tau + n * mu * dE/dx + mu * E * dn/dx + Dn * d^2n/dx^2

import numpy as np
import mpmath as mp
from phase1 import diffusionConstants, mobility, srhLifetime

q = 1.6e-19
T = 300
mu_n = 1265  # cm^2/Vs
Dn = diffusionConstants(T, mu_n)
tau = 1e-8  # s

# Doping
nSide, pSide, iSide = 1e17, 1e17, 1e14
xn, xp, W = 1e-6, 1e-6, 1e-6
L = xn + W + xp

# Numerical params
N = 1000

x = np.linspace(0, L, N)
h = x[1] - x[0]

# Generation
def G(x):
    return float(((hv * photonFlux) / bandGap) * mp.exp(-a * x))

G_array = np.array([G(xi) for xi in x])

print(G_array)

# Electric field profile
def ElecFieldAsXLoc(x):
    x = x - xn
    if x < 0:
        return -(-q * nSide / absolutePermittivity) * (x + xn)
    elif 0 <= x <= W:
        return -((-q * nSide * xn / absolutePermittivity) - (q * iSide / absolutePermittivity) * x)
    else:
        return -(q * pSide / absolutePermittivity) * (x - (W + xp))

E_array = np.array([ElecFieldAsXLoc(xi) for xi in x])


# Field derivative (backward difference)
dE_dx = np.zeros(N)
dE_dx[1:] = (E_array[1:] - E_array[:-1]) / h
dE_dx[0] = dE_dx[1]  # one-sided estimate at left edge

# Constants
mu_n = 1265  # cm^2/Vs
D_n = diffusionConstants(300, mu_n)
tau_n = 1e-8

# System
A = np.zeros((N, N))
b = np.zeros(N)

for i in range(1, N - 1):
    Ei = E_array[i]

    # Drift (upwind)
    if Ei >= 0:
        A[i, i]   += mu_n * Ei / h
        A[i, i-1] += -mu_n * Ei / h
    else:
        A[i, i]   += -mu_n * Ei / h
        A[i, i+1] += mu_n * Ei / h

    # Diffusion (central)
    A[i, i-1] += Dn / h**2
    A[i, i]   += -2 * Dn / h**2
    A[i, i+1] += Dn / h**2

    # Electric field gradient term
    gradE = (E_array[i] - E_array[i - 1]) / h
    A[i, i] += mu_n * gradE

    # Recombination
    A[i, i] += -1 / tau

    # Generation
    b[i] = G_array[i]
# Boundary conditions (n = 0 at both ends)
A[0, 0] = 1
A[N-1, N-1] = 1
b[0] = b[N-1] = 0

# Solve
n = np.linalg.solve(A, b)
print(n)

print(np.min(E_array), np.max(E_array))

