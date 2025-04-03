refValues = {
    "mobiltyMaxElectrons":1000,
    'mobilityMaxHoles':170,
    'mobilityMinElectrons': 55,
    'mobilityMinHoles': 3,
    'electronDopingRef':(2*(1e17)),
    'holeDopingRef':(3*(1e17)),
    'scalingElectrons':1,
    'scalingHoles':2,
    'paramAlphaElectrons':2,
    'paramAlphaHoles':5,
    'paramBetaElectrons':0.7,
    'paramBetaHoles':0.7
}

def funcConcentration(type,conc):
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

    val1 = mobilityMin+(mobilityMax*((dopingRef/conc)**scale))
    val2 = mobilityMax-mobilityMin
    result = val1/val2
    return result

def mobilityTotal(temp, type,conc):
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

    funcConcResult = funcConcentration(type,conc)

    val1 = funcConcResult*((temp/300)**paramBeta)
    val2 = 1+(funcConcResult*((temp/300)**(paramAlpha+paramBeta)))
    result = mobilityMax*(val1/val2)

    return result

print(mobilityTotal(300, 'n', 1e16))


