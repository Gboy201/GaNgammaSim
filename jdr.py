from phase1 import mobility, srhLifetime, depletionWidth, builtInPotential, diffusionConstants
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from scipy.integrate import quad, trapz
import csv

q=1.6e-19
print(srhLifetime())
a =5.2e-2
a*=6.15
E = (2.67*3.4) + 0.87
E = 1.58956743543344983e-18

xj = 1e-8
l=3
flux = 2.97e7
photonFlux=2e11
hv=2.4e-14
perimittivity = 8.9
absolutePermittivity = perimittivity * 8.85e-12
print(mobility('n', 0, 300))
maxMobilityElectrons = 1000

def ordered_integrate(func, x_start, x_end, num_points=100):

    x_vals = np.linspace(x_start, x_end, num_points)
    total = 0.0

    # Step through adjacent x values in order
    for i in range(len(x_vals) - 1):
        x0 = x_vals[i]
        x1 = x_vals[i+1]
        y0 = func(x0)
        y1 = func(x1)

        dx = abs(x1 - x0)
        area = 0.5 * (y0 + y1) * dx
        total += area

    return total

#Calculating photon flux

def photonFlux(energy, powerDensity):
    wavelength = 4.1357e-15*(3.8e8)/(energy)
    return powerDensity*wavelength/(q*1.24)

photonFlux = photonFlux(1250000,flux)
genRate = (a*photonFlux*np.exp(-a*xj))
print(genRate)

#Electric field

def electricField(distance, dopingAcceptor, dopingDoner, pDistance, nDistance):
    xp = ((-q*dopingAcceptor*distance)/absolutePermittivity)-((q*dopingAcceptor*pDistance)/absolutePermittivity)
    xn = ((q*dopingDoner*distance)/absolutePermittivity)-((q*dopingAcceptor*nDistance)/absolutePermittivity)
    if distance < 0:
        return xp
    else:
        return xn
    
dopingA = 1e22
dopingD = 1e15
nDistance = 30
pDistance= 15
    
def carrier(distance):
    eField =  electricField(distance,dopingA,dopingD,pDistance,nDistance)

distance = 0 #cm

electricFieldVal = []
dVal = []
for d in range(-15, 16, 1):
    electricFieldVal.append(electricField(d,dopingA,dopingD,pDistance,nDistance))
    dVal.append(d)

print(electricFieldVal)

# res, err = quad(carrier,0,30)
# print("The numerical result is {:f} (+-{:g})"
#     .format(res, err))

#Doing depletion region width calculations

refValuesDepletion = {
    'intrinsicCarrier':1.6e-10,
    'permittivity':8.9, #Also 9.7 for wurtzite structure
    'vacumnPermittivity': 8.85418782e-12
}

print("Depletion width:" +str(depletionWidth(builtInPotential(300,1e17,1e17),1e17,1e17,0)))

#Let's start with intrinsic region
#Parameters: Intirnsic doping: 1e12 for now

nSideGlob = 1e18
pSideGlob = 1e15
iSide = 1e14

# builtInPotentialIN = builtInPotential(300, (1.6e-10)/iSide, nSide)
# print("The built in potential  INis: "+ str(builtInPotentialIN))
builtInPotentialGlob = builtInPotential(300, nSideGlob, pSideGlob)
# print("The built in potential IP is: "+ str(builtInPotentialIP))

depletionWidthPGlob = depletionWidth(builtInPotentialGlob,pSideGlob,iSide,0)
# depletionWidthN = depletionWidth(builtInPotentialIN, nSide, (((1.6e-10)**2)/iSide), 0)

# print("Depletion Width on N-side:" + str(depletionWidthN[1]))
# print("Depletion width on I-N-Side:" + str(depletionWidthN[2]))
print("Depletion width on P-I-Side:" + str(depletionWidthPGlob[2]))
# print("Depletion width on P-Side:" + str(depletionWidthP[1]))

#Electric field equations of a PIN junction

#In centimeters
xp =depletionWidthPGlob[1]
W = depletionWidthPGlob[2]*(0.5)
xn = ((pSideGlob*xp)-(iSide*W))/(nSideGlob)


def ElectricField(iSide, pSide, nSide):
    x = -xn

    electricVal = []
    pos=[]


    while x < (W+xp):
        if x < 0:
            electricVal.append(((-q*nSide)/absolutePermittivity)*(x+xn))
            pos.append(x)
        if x > 0 and x < (W):
            electricVal.append((((-q*nSide)/absolutePermittivity)*(xn))-(((q*iSide)/absolutePermittivity)*x))
            pos.append(x)
        if x > (W):
            electricVal.append(((q*pSide)/absolutePermittivity)*(x-(W+xp)))
            pos.append(x)

        x+=1e-8

    return [electricVal,pos]

#Xn

# XN = ((iSide*0.055)+(pSide*1e-5))/nSide
# print("XN:" + str(XN))

# W = ((nSide*0.0002)-(pSide*0.0002))/iSide
# print("W: " + str(W))

electricVal = ElectricField(iSide,nSideGlob,pSideGlob)[0]
pos = ElectricField(iSide,nSideGlob,pSideGlob)[1]


# xp =depletionWidthP[1]
# W = depletionWidthP[2]
# xn =depletionWidthN[1]

print("xp, W, xn")
print(xp,W,xn)
print("****")
def ElecFieldAsX(x, nSide, pSide, iSide, xp, W, xn):
    if x < 0:
        return((-q*nSide)/absolutePermittivity)*(x+xn)
    if x >= 0 and x <= (W):
        return ((((-q*nSide)/absolutePermittivity)*(xn))-(((q*iSide)/absolutePermittivity)*x))
    if x > (W):
        return (((q*pSide)/absolutePermittivity)*(x-(W+xp)))

def totalElectricField(nSide, pSide, iSide, xp, W, xn, start, end):
    def ElecFieldAsXLoc(x):
        return ElecFieldAsX(x, nSide, pSide, iSide, xp, W, xn)
    
    I, err = quad(ElecFieldAsXLoc, start, end)
    avgField = I/(end-start)
    return avgField





#Average electric field

# vd = -1000 * avgField

# print("The drift velcoity is: " + str(vd))
# print(vd*9e-8)

# print("SRH LIFETIME:" + str(srhLifetime()))


#EMAX


# print("Emax: " + str(emax))

# depletionSize = 3.7e-5
# emax = (q*nSide*depletionSize)/absolutePermittivity

# print("Emax 2: " + str(emax))


#PDE for JDR

#Electrons:

#Majority recombination functions

#Radiative recombination

recombinationRefValues = {
    'radiativeRecombinationCoefficient': 1.1e-8, #cm3s-1
}

def radiativeRecombintionMaj(concMinority, concMajority):
    radiativeRecombinationCoefficient = recombinationRefValues['radiativeRecombinationCoefficient']
    lifetimeMajority = 1/(radiativeRecombinationCoefficient*concMinority)

    return lifetimeMajority

refAuger = {
    'simData':1e-30,
    'direct':3.5e-34,
    'indirect':2e-30
}


def augerLifetimeMaj(concMinority, concMajority): #Majority or minority
    augerCoefficient = refAuger['simData']
    augerMinority = (concMajority/(augerCoefficient*(((concMajority**2)*concMinority)+((concMajority**2)*concMinority))))
    return augerMinority

def bulkRecombinationLifetimeMaj(concMinority,concMajority, type):
    radiative = radiativeRecombintionMaj(concMinority, concMajority)
    auger = augerLifetimeMaj(concMinority, concMajority)
    srh = srhLifetime()


    if type == 'n':
        #For electron
        # minorityTotalBulkLifetime = 1/((1/radiative)+(1/auger)+(1/srh[0]))
        majorityTotalBulkLifetime = 1/((1/radiative)+(1/auger)+(1/srh[0]))

        return [majorityTotalBulkLifetime, radiative, auger, srh[0]]
    
    if type == 'p':
        #For hole
        majorityTotalBulkLifetime = 1/((1/radiative)+(1/auger)+(1/srh[1]))


        return [majorityTotalBulkLifetime, radiative, auger, srh[1]]

#Testing lifetimes
print("Lifetime 1e15 N:"+str(bulkRecombinationLifetimeMaj(1e15, 1e15,'n')))
print("Lifetime 1e15 P:"+str(bulkRecombinationLifetimeMaj(1e15, 1e15,'p')))

def totalMajorityLength(temp, typeOfMajority, majorityConc):
    majority = majorityConc
    minority = (1.6e-10**2)/majority
    diffusion = diffusionConstants(temp, mobility(typeOfMajority,(majority),temp))
    totalMajorityLifetimeVal = bulkRecombinationLifetimeMaj(minority,majority,typeOfMajority)
    totalLength = mp.sqrt(diffusion*totalMajorityLifetimeVal[0])

    return totalLength

# print("Diffusion length : " + str(totalMajorityLength(300, 'n', nSide)))

def diffusionCurrent(temp, typeOfMajority, majorityConc, injection):
    
    diffusion = diffusionConstants(temp, mobility(typeOfMajority, majorityConc, 300))
    # print(diffusion)
    diffusionLength = totalMajorityLength(temp,  typeOfMajority, majorityConc)
    # print(diffusionLength)

    val1 = (injection)
    # print(val1)
    val2 = val1/diffusionLength

    return (q*diffusion*val2)

def JDR(xVal1,xVal2, xn, xp, W, lifetimeN, mobilityN, lifetimeP, mobilityP,nSide, pSide, iSide, percent, hv, photonFlux):

    # builtInPotentialLocal = builtInPotential(300, nSide, pSide)

    # depletionWidthPLocal = depletionWidth(builtInPotentialLocal,pSide,iSide,0)



    # xp =depletionWidthPLocal[1]
    # W = depletionWidthPLocal[2]*(percent)
    # xn = ((pSide*xp)-(iSide*W))/(nSide)



    # print(xp, W, xn, xVal)


    depWidth = W+xn+xp





    def ElecFieldAsXLoc(x):
        if x < 0:
            return((-q*nSide)/absolutePermittivity)*(x+xn)
        if x >= 0 and x <= (W):
            return ((((-q*nSide)/absolutePermittivity)*(xn))-(((q*iSide)/absolutePermittivity)*x))

        if x > (W):
            return (((q*pSide)/absolutePermittivity)*(x-(W+xp)))
        


    I, err = quad(ElecFieldAsXLoc, -xn, xp+W, limit=1000)
    avgField = I/depWidth
            

    # print("depwidth: " + str(depWidth))

    #I(x) ver 1 with electric field'

    def intElecN(x):
        eF = ElecFieldAsXLoc(x)
        epsilon = -500  # Adjust based on scale (e.g., 1000 V/m)
        # print(epsilon)
        if abs(eF) < abs(epsilon):
            eF = epsilon

        if eF is None or mobilityN is None or lifetimeN is None:
            return 0  # or some safe fallback
        else:
            return 1 / (mobilityN * lifetimeN * eF)
        
    def intElecP(x):
        eF = ElecFieldAsXLoc(x)
        epsilon = -500  # Adjust based on scale (e.g., 1000 V/m)
        # print(epsilon)
        if abs(eF) < abs(epsilon):
            eF = epsilon

        if eF is None or mobilityP is None or lifetimeP is None:
            return 0  # or some safe fallback
        else:
            return 1 / (mobilityP * lifetimeP * eF)

    # graphE = []
    # graphI = []
    # i=-xn

    # while i < W+xp:
    #     graphE.append(intElecN(i))
    #     i+=0.0000001
    #     graphI.append(i)


        

    def In1(x):

        # print('x:'+str(x))

        accumulation = ordered_integrate(intElecN,xp+W, x)
        accumulation = mp.mpf(accumulation)

        eF = ElecFieldAsXLoc(x)
        epsilon = -500
        if abs(eF) < abs(epsilon):
            eF = epsilon


        

        # print("mp.exp(-accumulation: "+ str(mp.exp(-accumulation)))
        # print("eF:"+str(eF))

        return (eF * mp.exp(-accumulation))
    
    def Ip1(x):
        # print('x:'+str(x))

        accumulation = ordered_integrate(intElecP, x, -xn)
        accumulation = float(accumulation)
        # print(accumulation)

        eF = ElecFieldAsXLoc(x)
        epsilon = -500
        if abs(eF) < abs(epsilon):
            eF = epsilon


        

        # print("mp.exp(-accumulation: "+ str(mp.exp(accumulation)))
        # print("eF:"+str(eF))

        return (eF * mp.exp(abs(accumulation)))


    #Q(x) no electric field
    def Qn(x):
        distance = x+xn
        gen = (((hv*photonFlux)/E))*mp.exp(-(a*distance))
        # print('gen: ' + str(gen))


        return (-gen/mobilityN)
    
    def Qp(x):
        distance = x+xn
        gen = (((hv*photonFlux)/E))*mp.exp(-(a*distance))
        # print('gen: ' + str(gen))


        return (gen/mobilityP)


    #I(x) ver 2 with NO electric field

    def QIn(x):
        Q = Qn(x)

        accumulation = ordered_integrate(intElecN, xp+W, x)
        accumulation = mp.mpf(accumulation)
        I = mp.exp(accumulation)



        return (Q*I)
    
    def QIp(x):
        Q = Qp(x)
        accumulation = ordered_integrate(intElecP, x, -xn)
        # print(accumulation)

        accumulation = mp.mpf(abs(accumulation))
        I = mp.exp(accumulation)

        return (Q*I)

    #n(x) *** Include boundary constant

    def n(x):

        if x == depWidth:
            return 4.67e-8

        # print(x)

        val1 = 1/In1(x)
        val2 = ordered_integrate(QIn, xp+W, x)

        # print("Err:"+str(err))

        # print("Val 1: " + str(val1))
        # print("Val 2: " + str(val2))

        return (val1 * val2)
    
    def p(x):
        if x == -xn:
            return 4.67e-8
        # print(x)
        val1 = 1/Ip1(x)
        val2 = ordered_integrate(QIp, -xn, x)

        # print(val1, val2)

        return (val1 * val2)


    eMin = depWidth/(mobilityN*lifetimeN)

    densitiesElectrons = []
    densitiesHoles = []

    def allDensities(start, end):
        allPosN = np.linspace(start, end, 25)
        allPosP = np.linspace(start, end, 25)
        for i in allPosN:
            densitiesElectrons.append((abs(float(n(i)))))
        for i in allPosP:
            if abs(float(p(i))) > 1e14:
                pass
            else:
                densitiesHoles.append((abs(float(p(i)))))

    allDensities(xVal1, xVal2)

    avgConcN=0
    for i in densitiesElectrons:
        avgConcN+=i

    avgConcN/=len(densitiesElectrons)

    avgConcP=0

    for i in densitiesHoles:
        avgConcP+=i

    avgConcP/=len(densitiesHoles)

    # print(avgConcN, avgConcP)

    # print("Electron Lifetime")
    electronLifetime = bulkRecombinationLifetimeMaj(avgConcP, avgConcN, 'n')
    # print(electronLifetime)
    # print("Hole Lifetime")
    holeLifetime = bulkRecombinationLifetimeMaj(avgConcN, avgConcP, 'p')
    # print(holeLifetime)



    densityElectrons = abs(n(xVal1))
    densityHoles =p(xVal2)
    currentElectron = diffusionCurrent(300, 'n', nSide, densityElectrons)
    currentHole = -diffusionCurrent(300, 'p', pSide, densityHoles)
    totalCurrent = currentElectron+currentHole
                                       

    return ({"Electron density":densityElectrons, "Hole density": abs(densityHoles),"Emin":eMin,"Efield":abs(avgField),"xn":xn,'xp':xp,'W':W, "nSide":nSide, "pSide":pSide, "ratio":(eMin/abs(avgField)), "percent":percent, "Current":totalCurrent, "Current-Electron": currentElectron, "Current-Hole": currentHole, "Hole-lifetime":holeLifetime[0], "Electron-lifetime":electronLifetime[0],'Electron-Lifetime-Components':[electronLifetime[1], electronLifetime[2], electronLifetime[3]], "Hole-Lifetime-Components":[holeLifetime[1], holeLifetime[2], holeLifetime[3]], "Electron-density":densitiesElectrons, "Hole-density":densitiesHoles})


# xn = 0.09
# W = 8.8
# xp=0.18
# xn/=10000
# W/=10000
# xp/=10000

xn= 0.000257 
W= 0.002572 
xp=0.000514
estimateN= 9.19e-8
estimateP=2.54e-10
mobilityN= 1200
mobilityP=40 
nSide= 1.00e+17
pSide=1.00e+17
iSide=1.00e+15

def findConvergenceLifetime(estimateN, estimateP, xn, xp, W, nSide, pSide, iSide, percent, mobilityN, mobilityP, hv, photonFlux):
    estimateN= 9.19e-8
    estimateP=2.54e-10
    while True:
        results = JDR(-xn, xp+W, xn,xp,W,estimateN, 1265, estimateP, 40, nSide, pSide,iSide,percent, hv, photonFlux)
        lifetimeN = results['Electron-lifetime']
        lifetimeP = results['Hole-lifetime']
        # if abs(lifetimeN-estimateN) < 1e-9 and abs(lifetimeP-estimateP) < 1e-11:
        return [lifetimeN, lifetimeP, results]
        # else:
        #     estimateN = lifetimeN
        #     estimateP = lifetimeP
        #     continue


# print(findConvergenceLifetime(9e-8, 25e-11, xn, xp, W, nSide, pSide, iSide, 0.5, 1000, 40))


def genDevice(x, width, hv, photonFlux, E, a):
    def Q(x):
        distance = x  # Now using absolute position directly
        gen = (((hv*photonFlux)/E))*mp.exp(-(a*distance))
        return gen
    
    # Integrate generation over the entire device width
    i, err = quad(Q, x, width)

    return [i, Q(x)]

print("Section 1:"+str((genDevice(0, 0.0001, hv, photonFlux, E, a)[0])/(1*0.0001)))

print("Section 2:"+str(genDevice(0.001, 0.002, hv, photonFlux, E, a)[0]))

print("Section 3:" + str(genDevice(0.002, 0.003, hv, photonFlux, E, a)[0]))



def eMin(depWidth):
    importantVal = {
        'mobilityN':1000,
        'lifetimeN': 9e-8,
        'mobilityP':40,
        'lifetimeP': 25e-11
    }

    eMinN = depWidth/(importantVal['lifetimeN']*importantVal['mobilityN'])
    eMinP = depWidth/(importantVal['lifetimeP']*importantVal['mobilityP'])

    return [eMinN, eMinP]

# output = JDR(0,10e-9,1000,1e-9,30,11e15, 1e15,iSide,0.95)
# print(output)


results = []
percentValues = [round(x, 2) for x in np.arange(0.05, 1.01, 0.02)]
pDopes = [10**x for x in np.arange(15, 23.01, 0.5)]
nDopes = [10**x for x in np.arange(15, 23.01, 0.5)]
n = 1
# for percent in percentValues:
#     for pDope in pDopes:
#         for nDope in nDopes:
#             output = JDR(0,10e-9,1000,1e-9, 30, nDope, pDope,iSide,percent)
#             results.append(output)

#             print(n)
#             n+=1



            
            

def print_top_10_results(results):
    # Sort top 10 by electron density (descending)
        # Convert mpf and other values to float-safe copies for printing/saving
    def safe_float(val):
        try:
            return float(val)
        except:
            return val

    # Format entries into fully-float dictionaries for export
    formatted_results = []
    for entry in results:
        formatted_results.append({
            'Electron density': safe_float(entry['Electron density']),
            'Efield': safe_float(entry['Efield']),
            'Emin': safe_float(entry['Emin']),
            'ratio': safe_float(entry['ratio']),
            'nSide': safe_float(entry['nSide']),
            'pSide': safe_float(entry['pSide']),
            'percent': safe_float(entry['percent']),
            'W': safe_float(entry['W']),
            'xp': safe_float(entry['xp']),
            'xn': safe_float(entry['xn']),
            'Total Current' : float(entry['Current']),
            'Current-Electron' : float(entry['Current-Electron']),
            'Current-Hole' : float(entry['Current-Hole']),
        })

    top_by_density = sorted(results, key=lambda x: x['Current'], reverse=True)[:10]

    # Sort top 10 by ratio closest to 1
    top_by_ratio = sorted(results, key=lambda x: abs(x['ratio'] - 1))[:10]


    # Helper function to print each table
    def print_table(title, entries, mode='current'):
        print(f"\n🔟 {title}")
        print("-" * 130)
        if mode == 'current':
            header = f"{'Rank':<5} {'Electron Density':<20} {'Ratio':<10} {'Efield':<12} {'Emin':<12} {'nSide':<12} {'pSide':<12} {'%':<6} {'W':<10} {'xp':<10} {'xn':<10} {'Current':<10} {'Current-Electron':<10} {'Current-Hole':<10}"
        else:
            header = f"{'Rank':<5} {'Ratio':<10} {'Δ from 1':<12} {'Electron Density':<20} {'Efield':<12} {'Emin':<12} {'nSide':<12} {'pSide':<12} {'%':<6} {'W':<10} {'xp':<10} {'xn':<10} {'Current':<10} {'Current-Electron':<10} {'Current-Hole':<10}"
        print(header)
        print("-" * 130)

        for idx, entry in enumerate(entries, 1):
            ratio = float(entry['ratio'])
            ed = float(entry['Electron density'])
            eField = float(entry['Efield'])
            eMin = float(entry['Emin'])
            delta = (abs(float(ratio) - 1))
            n = float(entry['nSide'])
            p = float(entry['pSide'])
            percent = float(entry['percent'])
            W = float(entry['W'])
            xp = float(entry['xp'])
            xn = float(entry['xn'])
            current = float(entry['Current'])
            currentElectron = float(entry['Current-Electron'])
            currentHole = float(entry['Current-Hole'])

            if mode == 'current':
                print(f"{idx:<5} {ed:<20.3e} {ratio:<10.4f} {eField:<12} {eMin:<12} {n:<12.1e} {p:<12.1e} {percent:<6.2f} {W:<10} {xp:<10} {xn:<10} {current:<10} {currentElectron:<10} {currentHole:<10}")
            else:
                print(f"{idx:<5} {ratio:<10.4f} {delta:<12.2e} {ed:<20.3e} {eField:<12} {eMin:<12} {n:<12.1e} {p:<12.1e} {percent:<6.2f} {W:<10} {xp:<10} {xn:<10} {current:<10} {currentElectron:<10} {currentHole:<10}")

    # Print both tables
    print_table("Top Configurations by Total Current", top_by_density, mode='current')
    print_table("Top Configurations by E/Emin Ratio ≈ 1", top_by_ratio, mode='ratio')

    # 📁 Save full dataset to CSV
    with open('largeResults.csv', 'w', newline='') as csvfile:
        fieldnames = ['Electron density', 'Efield', 'Emin', 'ratio', 'nSide', 'pSide', 'percent', 'W', 'xp', 'xn', 'Total Current', 'Current-Electron', 'Current-Hole']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in formatted_results:
            writer.writerow(row)

    print(f"\n✅ All {len(results)} results saved to 'largeResults.csv'!")




# 🚨 Make sure 'results' is already populated
print_top_10_results(results)









# %%

# plt.plot(pos, electricVal)
# plt.xlabel('Device')
# plt.ylabel("Electron density")
# plt.title('Current through device')
# plt.show()

# %%
# plt.plot(posCurrent, current)
# plt.xlabel('Device')
# plt.ylabel("Current")
# plt.title('Current through depletion region')
# plt.show()





# %%
