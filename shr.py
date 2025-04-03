import math

refValues = {
    'intrinsicCarrier':1.6e-10,
    'KbT':25.7, #meV
    'Ei': 1700, #mev
}
#Fermi levels

nD = 1e18

eF = (refValues['KbT']*(math.log((nD/refValues['intrinsicCarrier']))))+refValues['Ei']
# print(eF)

# p = (refValues())

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
print(finalLifetimeElectron)

tP = 0
for i in defectsHole:
    tP += ((i[0]*i[1]*vthHoles))

finalLifetimeHole = 1/tP
print(finalLifetimeHole)

#Auger lifetimes
refAuger = {
    'simData':1e-30,
    'direct':3.5e-34,
    'indirect':2e-30
}

def augerLifetime(electronConc, holeConc):
    augerCoefficient = refAuger['simData']
    augerElectron = (1/(augerCoefficient*(electronConc**2)))
    augerHole = (1/(augerCoefficient*(holeConc**2)))
    return augerElectron, augerHole

print(augerLifetime(1e17,1e17))