def builtInPotential(temp, concElectron, concHole):
    intrinsicConc = refValuesDepletion['intrinsicCarrier']
    boltzmann = 1.38e-23
    q = 1.6e-19

    print(f"Calculating built-in potential with:")
    print(f"Temperature: {temp} K")
    print(f"Electron concentration: {concElectron} cm^-3")
    print(f"Hole concentration: {concHole} cm^-3")
    print(f"Intrinsic carrier concentration: {intrinsicConc} cm^-3")

    val1 = (boltzmann * temp) / q
    print(f"val1 (kT/q): {val1} V")

    r = (concElectron * concHole) / (intrinsicConc ** 2)
    print(f"r (n*p/ni^2): {r}")

    try:
        val2 = math.log(r)
        print(f"val2 (ln(r)): {val2}")
    except Exception as e:
        print(f"Error calculating ln(r): {str(e)}")
        print(f"r value that caused error: {r}")
        raise

    result = val1 * val2
    print(f"Final built-in potential: {result} V")

    return result 