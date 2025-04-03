from phase1 import totalMinorityLength, diffusionConstants, mobility, surfaceVelocityInterface
import numpy as np
import mpmath as mp

def Je(temp, typeOfMinority, majorityConc, width):
    q = 1.6e-19
    permittivity = 8.9

    #For an average wavelength of 1.25 mev (Co60):
    a = 7e-1 #cm2/g
    a*=6.15
    lp = totalMinorityLength(temp, typeOfMinority, majorityConc, width)

    minority = (1.6e-10**2)/majorityConc
    dp= diffusionConstants(temp, mobility(typeOfMinority,(minority),temp))
    sp1 = 5e4 + surfaceVelocityInterface(dp, width)
    sp = sp1**-1


    allVal = {
        'lp': lp,
        'dp': dp,
        'sp': sp,
        'minority': minority,
        'width': width
    }
    print(allVal)
    print(width/lp)
    
    val1 = (q*a*lp)/((((a**2)*(lp**2)) - 1))
    print("val1:" + str(val1))


    val2num = ((sp*lp)/dp)-((mp.exp(-a*width))*((((sp*lp)/dp)*(mp.exp(width/lp))/2)+mp.exp((width/lp))/2))
    val2den = ((sp*lp)/dp)*(mp.exp((width/lp))/2)+(mp.exp(width/lp)/2)

    print("val2num:" + str(val2num))
    print("val2den:" + str(val2den))

    # print(f"val2num = {val2num}, val2den = {val2den}")
    # print(f"sp = {sp}, lp = {lp}, dp = {dp}")
    # print(f"sp*lp/dp = {sp*lp/dp}")
    # print(f"sp*lp = {sp*lp}")
    # print(f"sinh(width/lp) = {np.sinh(width/lp)}, cosh(width/lp) = {np.cosh(width/lp)}")


    # print(f"sinh(width/lp) * cosh(width/lp) = {np.sinh(width/lp) * np.cosh(width/lp)}")

    print(f"(-a*width) = {-a*width}")


    val2 = val2num/val2den

    val3 = a*lp*np.exp(-a*width)

    finalVal = val1 * (val2-val3)

    return finalVal


size=100
je = Je(300, 'p', 1e18, 1e-3)
print('je:'+str(je))
finalVal=2.97e7*(je)*(size**2)
print(finalVal)