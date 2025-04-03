import math
import numpy as np
import mpmath as mp
import time
#Function for carrier mobility

#Ref values for carrier mobility



def mobility(type,conc, temp):
    refValues = {
        "mobiltyMaxElectrons":1265,
        'mobilityMaxHoles':40,
        'mobilityMinElectrons': 55,
        'mobilityMinHoles': 3,
        'electronDopingRef':(2e17),
        'holeDopingRef':(3e17),
        'scalingElectrons':1,
        'scalingHoles':2,
        'paramAlphaElectrons':-2,
        'paramAlphaHoles':-3,
        'paramBetaElectrons':-3.8,
        'paramBetaHoles':-3.7
    }
    if type == 'n':
        mobilityMax = refValues['mobiltyMaxElectrons']
        mobilityMin = refValues['mobilityMinElectrons']
        dopingRef = refValues['electronDopingRef']
        scale = refValues['scalingElectrons']
        paramAlpha = refValues['paramAlphaElectrons']
        paramBeta = refValues['paramBetaElectrons']

    if type == 'p':
        mobilityMax = refValues['mobilityMaxHoles']
        mobilityMin = refValues['mobilityMinHoles']
        dopingRef = refValues['holeDopingRef']
        scale = refValues['scalingHoles']
        paramAlpha = refValues['paramAlphaHoles']
        paramBeta = refValues['paramBetaHoles']

    # val1 = (mobilityMax*((temp/300)**paramAlpha)) - (mobilityMin)
    # val2 = val1 / (1+(((temp/300)**paramBeta))*((conc/dopingRef)**scale))
    # result = mobilityMin+ val2

    val1 = mobilityMax-mobilityMin
    val2 = val1/(1+((conc/dopingRef)**scale))

    result = mobilityMin+val2

    return result
   

print("mobility:" + str(mobility('p',1e15,300)))


#Getting diffusion constants (Einstein relation)

def diffusionConstants(temp, mobility):
    boltzmann = 1.38e-23
    q = 1.6e-19
    mobilityVal = mobility
    result = ((temp*boltzmann)/(q))*mobilityVal
    return result

print(str(diffusionConstants(300, mobility('p',(1e-10**2/1e17),300)))+" cm^2/s")



refValuesDepletion = {
    'intrinsicCarrier':1.6e-10,
    'permittivity':8.9, #Also 9.7 for wurtzite structure
    'vacumnPermittivity': 8.85418782e-12
}

#Built in potential

#use math.log()


def builtInPotential(temp, concElectron, concHole):
    intrinsicConc = refValuesDepletion['intrinsicCarrier']
    boltzmann = 1.38e-23
    q=1.6e-19


    val1 = (boltzmann*temp)/q
    r=(concElectron*concHole)/(intrinsicConc**2)
    val2 = math.log(r)
    # print(r)
    result = val1*val2
    return result


#Depletion width is in meters
def depletionWidth(builtInPotential, concElectron, concHole, appliedVoltage):
    permittivity = refValuesDepletion['permittivity']*refValuesDepletion['vacumnPermittivity']
    vacumnPermittivity = refValuesDepletion['vacumnPermittivity']
    q=1.6e-19

    # concElectron*=1e6
    # concHole*=1e6


    widthNegative = math.sqrt(((2*builtInPotential*permittivity)/q)*(concHole/(concElectron*(concHole+concElectron))))
    widthPositive = math.sqrt(((2*builtInPotential*permittivity)/q)*(concElectron/(concHole*(concHole+concElectron))))
    width = [widthNegative+widthPositive, widthNegative, widthPositive]

    # width = math.sqrt(((2*permittivity)/q)*builtInPotential*((1/concHole)+(1/concElectron)))


    # width = ((2*permittivity*(builtInPotential)*(concElectron+concHole))/(q*concHole*concElectron))**0.5

    return width

# print(str(builtInPotential(300,1e1, 1e17))+" volts")

Val = 17
# for i in range(8):
#     test = 1*(10**Val)
#     print(str(test)+ " : "+str(depletionWidth(builtInPotential(300,test, test),test,test, -1)))
#     Val+=1


#Calculating recombintion:

recombinationRefValues = {
    'radiativeRecombinationCoefficient': 1.1e-8, #cm3s-1
}

def recombination(concElectron, concHole):
    radiativeRecombinationCoefficient = recombinationRefValues['radiativeRecombinationCoefficient']
    lifetimeBandElectrons = 1/(radiativeRecombinationCoefficient*concElectron)
    totalRecombinationElectron = lifetimeBandElectrons

    return totalRecombinationElectron

# print(recombintion(1e9,0))

#Rediative recombination
recombinationRefValues = {
    'radiativeRecombinationCoefficient': 1.1e-8, #cm3s-1
}

def radiativeRecombintion(concMinority, concMajority):
    radiativeRecombinationCoefficient = recombinationRefValues['radiativeRecombinationCoefficient']
    lifetimeMinority = 1/(radiativeRecombinationCoefficient*concMajority)

    return lifetimeMinority


#Auger recombination

#Auger lifetimes
refAuger = {
    'simData':1e-30,
    'direct':3.5e-34,
    'indirect':2e-30
}

def augerLifetime(concMinority, concMajority, type): #Majority or minority
    augerCoefficient = refAuger['simData']
    if type =='minority':

        augerMinority = (concMinority/(augerCoefficient*(((concMajority**2)*concMinority)+((concMajority**2)*concMinority))))
        return augerMinority
    if type =='majority':

        augerMinority = (concMajority/(augerCoefficient*(((concMajority**2)*concMinority)+((concMajority**2)*concMinority))))
        return augerMinority

print("Auger Lifetime:" + str(augerLifetime(1.6e-10, 1e17, "minority")))
#Srh recombination

def srhLifetime():

    #Defect paramaters = [crossSection, concentration]
    defectsElectron = [[8e-15,1.6e13],[1e-14, 3e13], [1.65e-17,1.19e15]]#cm
    defectsHole = [[5e-15,5e14],[3e-14,2e15],[2e-13,5e14],[3e-14,1e14]]#cm

    #Calculating srh lifetimes
    vthElectron =2.43e7#cm/s
    vthHoles = 2.38e7#cm/s
    tE = 0
    for i in defectsElectron:
        tE += ((i[0]*i[1]*vthElectron))


    finalLifetimeElectron = 1/tE

    tP = 0
    for i in defectsHole:
        tP += ((i[0]*i[1]*vthHoles))

    finalLifetimeHole = 1/tP

    return [finalLifetimeElectron, finalLifetimeHole]

print('srhLifetime:'+str(srhLifetime()[0]))

def bulkRecombinationLifetime(concMinority,concMajority, type):
    radiative = radiativeRecombintion(concMinority, concMajority)
    auger = augerLifetime(concMinority, concMajority, 'minority')
    srh = srhLifetime()


    if type == 'n':
        #For electron
        # minorityTotalBulkLifetime = 1/((1/radiative)+(1/auger)+(1/srh[0]))
        minorityTotalBulkLifetime = 1/((1/radiative)+(1/auger)+(1/srh[0]))
    
    if type == 'p':
        #For hole
        minorityTotalBulkLifetime = 1/((1/radiative)+(1/auger)+(1/srh[1]))


    return minorityTotalBulkLifetime

print("Minority Lifetime: " + str(bulkRecombinationLifetime((1.6e-10/(1e16**2)),1e16, 'n')))

refValuesSurface = {
    'r0':3e-6,#cm
    'Ndd': 1e9, #cm-2

}
def surfaceRecombinationLifetimeInterface(minorityDiffusivity):
    r0 = refValuesSurface['r0']
    Ndd=refValuesSurface['Ndd']
    pi = math.pi
    lifetime = (1/(2*pi*minorityDiffusivity*Ndd))*(math.log(1/(r0*math.sqrt(pi*Ndd)))-0.57)
    return lifetime

print("interface: "+str(surfaceRecombinationLifetimeInterface(1.03)))

def surfaceVelocityInterface(minorityDiffusivity,width):
    lifetime = surfaceRecombinationLifetimeInterface(minorityDiffusivity)
    velocity = math.sqrt(minorityDiffusivity*(1/lifetime))*math.tan((width/2)*math.sqrt((1/minorityDiffusivity)*(1/lifetime)))
    return velocity

print("interfaceVelocity: "+str(surfaceVelocityInterface(1.03, 1e-5)))


def surfaceRecombinationLifetimeBare(minorityDiffusivity,width):
    pi=math.pi
    lifetime = width/5e4+((1/minorityDiffusivity)*((width/pi)**2))
    return lifetime

print("bare: "+str(surfaceRecombinationLifetimeBare(1.03,1e-5)))

def surfaceVelocity(minorityDiffusivity,width,lifetime):
    velocity = math.sqrt(minorityDiffusivity*(1/lifetime))*math.tan((width/2)*math.sqrt((1/minorityDiffusivity)*(1/lifetime)))
    return velocity




def totalMinorityLifetime(concMinority,concMajority,type,minorityDiffusivity,width): #In cm
    # totalLifetimeSurface =surfaceRecombinationLifetimeInterface(minorityDiffusivity)+surfaceRecombinationLifetimeBare(minorityDiffusivity,width)
    totalLifetimeBulk = bulkRecombinationLifetime(concMinority,concMajority,type)

    total = (1/totalLifetimeBulk)

    return total**-1

majority = 1e17
minority = (1.6e-10**2)/majority
print('totalMinorityLifetime:'+str(totalMinorityLifetime(minority,majority,'n',0.12,0.3)))

diffusion = diffusionConstants(300, mobility('n',(minority),300))

#Lifetime
# print(math.sqrt(diffusion*totalMinorityLifetime(minority,majority,'n',diffusion,0.3)))



#Final simulation
def totalMinorityLength(temp, typeOfMinority, majorityConc, width):
    majority = majorityConc
    minority = (1.6e-10**2)/majority
    diffusion = diffusionConstants(temp, mobility(typeOfMinority,(minority),temp))
    totalMinorityLifetimeVal = totalMinorityLifetime(minority,majority,typeOfMinority,diffusion,width)
    totalLength = math.sqrt(diffusion*totalMinorityLifetimeVal)

    return totalLength

# print(totalMinorityLength(300,'n',1e17,0.3))

# a = 5.2e-2
a=2e-1
q = 1.6e-19
E = (2.67*3.39)+0.87
E = 1.58956743543344983e-18
# hv = 2e-13
hv=2.4e-14
#IIF = incident illumination flux
def Jl(IIF, Sr):
    q=1.6e-19

    return q*IIF*Sr

def Je(temp, typeOfMinority, majorityConc, width,a):
    q = 1.6e-19
    permittivity = 8.9

    #For an average wavelength of 1.25 mev (Co60):
    a = a #cm2/g
    lp = totalMinorityLength(temp, typeOfMinority, majorityConc, width)

    minority = (1.6e-10**2)/majorityConc
    dp= diffusionConstants(temp, mobility(typeOfMinority,(majorityConc),temp))
    sp = 1e4


    allVal = {
        'lp': lp,
        'dp': dp,
        'sp': sp,
        'minority': minority,
        'width': width
    }
    # print(allVal)
    # print(width/lp)
    
    val1 = (q*hv*a*lp)/((E*((a**2)*(lp**2)) - 1))
    # print("val1:" + str(val1))

    if width > 1e-3:
        val2num = ((sp*lp)/dp)-((mp.exp(-a*width))*((((sp*lp)/dp)*(mp.exp(width/lp))/2)+mp.exp((width/lp))/2))
        val2den = ((sp*lp)/dp)*(mp.exp((width/lp))/2)+(mp.exp(width/lp)/2)
    else:
        val2num = ((sp*lp)/dp)-((np.exp(-a*width))*((((sp*lp)/dp)*np.cosh(width/lp))+np.sinh(width/lp)))
        val2den = ((sp*lp)/dp)*(np.sinh(width/lp))+(np.cosh(width/lp))

    # print("val2num:" + str(val2num))
    # print("val2den:" + str(val2den))

    # print(f"val2num = {val2num}, val2den = {val2den}")
    # print(f"sp = {sp}, lp = {lp}, dp = {dp}")
    # print(f"sp*lp/dp = {sp*lp/dp}")
    # print(f"sp*lp = {sp*lp}")
    # print(f"sinh(width/lp) = {np.sinh(width/lp)}, cosh(width/lp) = {np.cosh(width/lp)}")


    # print(f"sinh(width/lp) * cosh(width/lp) = {np.sinh(width/lp) * np.cosh(width/lp)}")

    # print(f"(-a*width) = {-a*width}")


    val2 = val2num/val2den

    val3 = a*lp*np.exp(-a*width)

    finalVal = val1 * (val2-val3)

    return finalVal

def Jdr(temp, concElectron, concHole, width, a):
    q = 1.6e-19
    a = a #cm2/g
    a*=6.15
    builtInPotentialVal = builtInPotential(temp,concElectron,concHole)
    # print("builtInPotential:"+str(builtInPotentialVal))
    depletionWidthVal = depletionWidth(builtInPotentialVal,concElectron,concHole,0)[0]

    # depletionWidthVal = 3
    # print('depletionWidthVal:'+str(depletionWidthVal))
    finalVal = (((q*hv)/E)*np.exp(-(a*width)))*(1-(np.exp(-(a*depletionWidthVal))))
    # finalVal = q*(np.exp(-(a*depletionWidthVal)))

    return [finalVal,depletionWidthVal,builtInPotentialVal]

def Jb(temp, typeOfMinority, majorityConc, width1, width2, depletionWidth, a):
    
    q = 1.6e-19
    permittivity = 8.9

    #For an average wavelength of 1.25 mev (Co60):
    a = a#cm2/g
    a*=6.15
    ln = totalMinorityLength(temp, typeOfMinority, majorityConc, width2)

    minority = (1.6e-10**2)/majorityConc
    dn= diffusionConstants(temp, mobility(typeOfMinority,(majorityConc),temp))
    sn = surfaceVelocityInterface(dn, width2)


    allVal = {
        'ln': ln,
        'dn': dn,
        'sn': sn,
        'minority': minority,
        'width': width2,
    }
    # print(allVal)

    val1 = ((q*a*hv*ln)/(E*((a**2)*(ln**2))-1))*mp.exp(-a*(width1+depletionWidth))

    val2num = ((sn*ln)/dn)*(mp.cosh(width2/ln)-mp.exp(-a*width2))+mp.sinh(width2/ln)+(a*ln*mp.exp(-a*width2))

    val2den = ((sn*ln)/dn)*(mp.sinh(width2/ln))+(mp.cosh(width2/ln))

    finalVal = val1*((a*ln)-(val2num/val2den))

    return finalVal

def reverseSaturationCurrent(temp, donorConc, acceptorConc, emitterWidth, baseWidth):
    q = q = 1.6e-19
    intrinsicConc = 1.6e-10
    nA = acceptorConc
    nD = donorConc

    #For p-type region:
    ln = totalMinorityLength(temp, 'n', nA, emitterWidth)
    minority = (1.6e-10**2)/nA
    dn= diffusionConstants(temp, mobility('n',(nA),temp))
    sn = surfaceVelocityInterface(dn, emitterWidth)

    #For n-type region:
    lp = totalMinorityLength(temp, 'p', nD, baseWidth)
    minority = (1.6e-10**2)/nD
    dp= diffusionConstants(temp, mobility('p',(nD),temp))
    sp = surfaceVelocityInterface(dn, baseWidth)

    #Calculations
    val1 = (q*dn*(intrinsicConc**2))/(ln*nA)
    val2 = ((((sn*ln)/dn)*mp.cosh(emitterSize/ln))+mp.sinh(emitterSize/ln))/((((sn*ln)/dn)*mp.sinh(emitterSize/ln))+mp.cosh(emitterSize/ln))

    val3 = (q*dp*(intrinsicConc**2))/(lp*nD)
    val4 = ((((sp*lp)/dp)*mp.cosh(baseSize/lp))+mp.sinh(baseSize/lp))/((((sp*lp)/dp)*mp.sinh(baseSize/lp))+mp.cosh(baseSize/lp))

    finalVal = (val1*val2)+(val3*val4)

    return finalVal

def Voc(totalCurrent, reverseSaturationCurrent, temp):
    boltzmann = 1.38e-23

    val1 = ((boltzmann*temp)/q) * float(np.log((totalCurrent/reverseSaturationCurrent)+1))
    return val1

def radPerHour(photonFlux, size, a):
    radPerHour = ((photonFlux*(1-mp.exp(-a*(size)))*hv)/(6.15*(size)/1000))*36
    return radPerHour

size=1

emitterSize = 1e-8
baseSize = 1

pDope = 1e15
nDope = 1e15

flux = 2.97e7
def photonFlux(energy, powerDensity):
    wavelength = 4.1357e-15*(3.8e8)/(energy)
    wavelength *= 10e6
    return powerDensity*wavelength/(q*1.24)

# photonFlux = 1e17
photonFlux=2e11

# print('photonFlux: ' + str(photonFlux))

hv = 2.3999964e-14
jdr = Jdr(300,nDope,pDope,emitterSize,a)
je = Je(300, 'p', nDope, 0.041667625,a)
jb = Jb(300, 'n', pDope,emitterSize,baseSize,jdr[1],a)
rC = reverseSaturationCurrent(300, nDope, pDope, emitterSize, baseSize)
# print('Jdr:'+str(jdr[0]))
# print('je:'+str(je))
# print('jb:'+str(jb))
finalVal=photonFlux*(je+jdr[0]+jb)*(size**2)
# print("Built in potential val: "+str(jdr[2]))
# print('FinalVal:'+str(finalVal))
print(jdr)
print(rC)

# # Define the ranges
# emitter_sizes = np.linspace(1e-6, 1, num=25)  # From 1e-6 to 1 in 100 steps
# base_sizes = np.linspace(1e-6, 1, num=25)  # From 1e-6 to 1 in 100 steps

# # Generate doping values (1e14 to 1e22 with increments on both base and exponent)
# pDopes = [1 * 10**exp for exp in range(15, 23)]
# nDopes = [1 * 10**exp for exp in range(15, 23)]

# # Store results
# import heapq
# class Top10Lists:
#     def __init__(self):
#         self.heap = []  # Min-heap to store top 10 lists

#     def add_list(self, new_list):
#         """
#         Adds a new list while keeping only the top 10 largest first-element lists.
#         :param new_list: A list where the first element is the key.
#         """
#         if len(self.heap) < 10:
#             # Add if we have fewer than 10 elements
#             heapq.heappush(self.heap, (new_list[0], new_list))
#         else:
#             # If the new list's first element is larger than the smallest in the heap, replace it
#             if new_list[0] > self.heap[0][0]:  # Compare first elements
#                 heapq.heappushpop(self.heap, (new_list[0], new_list))

#     def get_top_10(self):
#         """Returns the top 10 lists sorted in descending order."""
#         return sorted([lst for _, lst in self.heap], key=lambda x: x[0], reverse=True)

# # Initialize the Top10Lists manager
# top_10_manager = Top10Lists()

# val=0
# # Nested loop to iterate through all combinations
# for emitterSize in emitter_sizes:
#     for baseSize in base_sizes:
#         for pDope in pDopes:
#             for nDope in nDopes:
#                 # Append the combination to the list
#                 jdr = Jdr(300,nDope,pDope,emitterSize,a)
#                 if (jdr[1]/2) > emitterSize or (jdr[1]/2) > baseSize:
#                     continue
#                 else:
#                     je = Je(300, 'p', nDope, (emitterSize-(jdr[1]/2)),a)
#                     jb = Jb(300, 'n', pDope,(emitterSize-(jdr[1]/2)),(baseSize-(jdr[1]/2)),jdr[1],a)
#                     finalVal=photonFlux*(je+jdr[0]+jb)*(size**2)
#                     rC = reverseSaturationCurrent(300, nDope, pDope, (emitterSize-(jdr[1]/2)), (baseSize-(jdr[1]/2)))
#                     voltage = Voc(finalVal, rC, 300)
#                 #Finding degradation
#                     radPerHour = ((photonFlux*(1-mp.exp(-a*(emitterSize+baseSize)))*hv*(size**2))/(6.15*(emitterSize+baseSize)/1000))*36
               
#                     top_10_manager.add_list([finalVal, rC,voltage, jdr[1], emitterSize,baseSize,pDope,nDope,je,jdr,jb,radPerHour])
            
#                 # Print the values at each iteration
#                 # print(f"FinalVal:{mp.nstr(finalVal)}, Emitter Size: {emitterSize:.1e}, Base Size: {baseSize:.1e}, pDope: {pDope:.1e}, nDope: {nDope:.1e}")

# # Total combinations count



# # Get the final top 10 lists
# final_top_10 = top_10_manager.get_top_10()

# # Print results
# print("\nFinal Top 10 Lists (Sorted by Current):")
# for lst in final_top_10:
#     print('Current: '+str(lst[0]))
#     print('Reverse saturation current: '+ str(lst[1]))
#     print('Voltage: ' + str(lst[2]))
#     print('Depletion-Width Size: ' + str(lst[3]))
#     print('Emitter Size: ' + str(lst[4]))
#     print('Base Size: ' + str(lst[5]))
#     print('Doner Doping: ' + str(lst[6]))
#     print('Acceptor Doping: ' + str((lst[7])))
#     print('Je:'+str(lst[8]))
#     print('Jdr:'+str(lst[9]))
#     print('Jb:'+str(lst[10]))
#     print('Rad Per Hour: '+ str(lst[11]))

