import math

#Ref values for carrier mobility
refValues = {
    "mobiltyMaxElectrons":1000,
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

def mobility(type,conc, temp):
    if type == 'n':
        mobilityMax = refValues['mobiltyMaxElectrons']
        mobilityMin = refValues['mobilityMinElectrons']
        dopingRef = refValues['holeDopingRef']
        scale = refValues['scalingElectrons']
        paramAlpha = refValues['paramAlphaElectrons']
        paramBeta = refValues['paramBetaElectrons']

    if type == 'p':
        mobilityMax = refValues['mobilityMaxHoles']
        mobilityMin = refValues['mobilityMinHoles']
        dopingRef = refValues['electronDopingRef']
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
   

print("mobility:" + str(mobility('n',1e18,300)))


#Getting diffusion constants (Einstein relation)

def diffusionConstants(temp, mobility):
    boltzmann = 1.38e-23
    q = 1.6e-19
    mobilityVal = mobility
    result = ((temp*boltzmann)/(q))*mobilityVal
    return result

print('diffusion:'+str(diffusionConstants(300,mobility('n',1e18,300))))

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

def augerLifetime(concMinority, concMajority):
    augerCoefficient = refAuger['simData']
    augerMinority = (1/(augerCoefficient*(concMajority**2)))
    return augerMinority

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


def bulkRecombinationLifetime(concMinority,concMajority, type):
    radiative = radiativeRecombintion(concMinority, concMajority)
    auger = augerLifetime(concMinority, concMajority)
    srh = srhLifetime()

    if type == 'n':

    #For electron
        minorityTotalBulkLifetime = 1/((1/radiative)+(1/auger)+(1/srh[0]))
    
    if type == 'p':
        minorityTotalBulkLifetime = 1/((1/radiative)+(1/auger)+(1/srh[1]))
    #For hole

    return minorityTotalBulkLifetime

# print(bulkRecombinationLifetime(1.6e-10,1.6e17, 'electron'))

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



def surfaceRecombinationLifetimeBare(minorityDiffusivity,width):
    pi=math.pi
    lifetime = width/5e4+((1/minorityDiffusivity)*((width/pi)**2))
    return lifetime


def totalMinorityLifetime(concMinority,concMajority,type,minorityDiffusivity,width):
    totalLifetimeSurface =surfaceRecombinationLifetimeInterface(minorityDiffusivity)+surfaceRecombinationLifetimeBare(minorityDiffusivity,width)
    totalLifetimeBulk = bulkRecombinationLifetime(concMinority,concMajority,type)

    total = (1/totalLifetimeSurface)+(1/totalLifetimeBulk)

    return total**-1

majority = 1.6e-10
minority = (1.6e-10**2)/majority

lifetime = totalMinorityLifetime(minority,majority,'n',diffusionConstants(300,mobility('n',1.6e-10,300)),0.3)
print("Lifetime:"+str(lifetime))

print(math.sqrt(diffusionConstants(300,mobility('n',1e-10,300)*lifetime)))