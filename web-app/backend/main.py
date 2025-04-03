import sys
import os
from pathlib import Path
import asyncio
import logging
import json
import time
import traceback
import math
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
import csv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Tuple, Literal, Optional, Union, Any
import numpy as np

# Define fundamental constants
q = 1.6e-19  # Elementary charge in Coulombs

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get absolute path to the root directory and add it to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger.info(f"Root directory: {root_dir}")

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
    logger.info(f"Added {root_dir} to Python path")

logger.info("Current Python path:")
for path in sys.path:
    logger.info(f"  {path}")

try:
    # Import phase1 first as it's the most fundamental
    logger.info("Attempting to import phase1...")
    sys.path.insert(0, root_dir)  # Ensure root_dir is first in path
    from phase1 import (
        builtInPotential as phase1_builtInPotential,
        depletionWidth,
        mobility,
        diffusionConstants,
        srhLifetime,
        radiativeRecombintion as radiativeRecombination,
        augerLifetime,
        refAuger,
        recombinationRefValues,
        bulkRecombinationLifetime,
        surfaceVelocityInterface,
        surfaceRecombinationLifetimeBare,
        surfaceRecombinationLifetimeInterface,
        surfaceVelocity,
        totalMinorityLength,
        Je,
        Jb,
        reverseSaturationCurrent,
        Voc,
        radPerHour
    )
    logger.info("Successfully imported phase1")

    # Import jdr2 which depends on phase1
    logger.info("Attempting to import jdr2...")
    from jdr2 import ElecFieldAsX
    logger.info("Successfully imported jdr2")

    # Import jdr which depends on both phase1 and jdr2
    logger.info("Attempting to import jdr...")
    from jdr import eMin, totalElectricField, genDevice, JDR, findConvergenceLifetime
    logger.info("Successfully imported jdr")

except ImportError as e:
    logger.error(f"Import Error: {str(e)}")
    logger.error(f"Current directory: {os.getcwd()}")
    logger.error(f"Python path: {sys.path}")
    logger.error(f"Files in root directory: {os.listdir(root_dir)}")
    raise
except Exception as e:
    logger.error(f"Unexpected error during imports: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise

matplotlib.use('Agg')

app = FastAPI()

# Configure CORS - more permissive during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulationResult(BaseModel):
    position: float
    electricField: float

class SimulationParams(BaseModel):
    donorConcentration: Any
    acceptorConcentration: Any
    temperature: float
    nRegionWidth: float = 15.0  # Default to 15 micrometers
    pRegionWidth: float = 15.0  # Default to 15 micrometers
    junctionType: str
    intrinsicConcentration: Optional[float] = None
    percent: Optional[float] = None
    photonFlux: float
    photonEnergy: float

    @validator('donorConcentration', 'acceptorConcentration')
    def validate_concentration(cls, v):
        try:
            return float(v)  # Convert any numeric value to float
        except (ValueError, TypeError):
            raise ValueError(f"Invalid concentration value: {v}")

    class Config:
        json_schema_extra = {
            "example": {
                "temperature": 300,
                "donorConcentration": 1e15,
                "acceptorConcentration": 1e15,
                "nRegionWidth": 15.0,
                "pRegionWidth": 15.0,
                "junctionType": "PN",
                "intrinsicConcentration": None,
                "percent": None,
                "photonFlux": 1e13,
                "photonEnergy": 0.0
            }
        }

class SimulationResponse(BaseModel):
    builtInPotential: float
    depletionWidth: float
    electricField: List[Tuple[float, float]]
    position: List[float]
    temperature: float
    electronConcentration: float
    holeConcentration: float
    message: str
    pSideWidth: float
    intrinsicWidth: float
    nSideWidth: float
    eMinElectron: float
    eMinHole: float
    eMin: float
    totalElectricField: float
    photonFlux: float
    photonEnergy: float
    # Device Parameters
    emitterSize: float  # N-region width (nRegionWidth)
    baseSize: float  # P-region width (pRegionWidth)
    totalDeviceWidth: float  # Sum of all region widths
    sizeWarning: Optional[str] = None  # Warning message for small device sizes
    # GaN Material Properties
    ganDensity: float = 6.15  # g/cm³
    massAttenuation: float  # cm²/g (calculated from CSV)
    linearAttenuation: float  # cm⁻¹ (calculated from density * mass attenuation)
    mobilityMaxElectrons: float = 1000.0  # cm²/V·s
    mobilityMaxHoles: float = 40.0  # cm²/V·s
    mobilityMinElectrons: float = 55.0  # cm²/V·s
    mobilityMinHoles: float = 3.0  # cm²/V·s
    intrinsicCarrierConcentration: float = 1.6e-10  # cm⁻³
    dielectricConstant: float = 8.9  # dimensionless
    radiativeRecombinationCoefficient: float = 1.1e-8  # cm²/s
    augerCoefficient: float = 1e-30  # cm⁶/s
    electronThermalVelocity: float = 2.43e7  # cm/s
    holeThermalVelocity: float = 2.38e7  # cm/s
    # Surface Recombination Velocities
    surfaceRecombinationVelocities: dict = {
        "bareEmitter": 0.0,  # cm/s
        "substrateBase": 0.0  # cm/s
    }
    # Minority Recombination Rates
    minorityRecombinationRates: List[dict] = [
        {
            "region": "Emitter (N-Doped)",
            "auger": 0.0,
            "srh": 0.0,
            "radiative": 0.0,
            "bulk": 0.0
        }
    ]
    # Generation Rate
    generationRateData: List[Tuple[float, float]] = []
    # Generation per Region
    generationPerRegion: dict = {
        "emitter": 0.0,
        "depletion": 0.0,
        "base": 0.0
    }
    # Generation Rate Profile
    generationRateProfile: dict = {
        "positions": [],
        "values": []
    }
    # Current Generation Data
    currentGenerationData: dict = {
        "emitter": {
            "generated_carriers": 0.0,
            "surviving_carriers": 0.0,
            "current": 0.0
        },
        "depletion": {
            "generated_carriers": 0.0,
            "surviving_carriers": 0.0,
            "current": 0.0
        },
        "base": {
            "generated_carriers": 0.0,
            "surviving_carriers": 0.0,
            "current": 0.0
        },
        "jdr": {
            "current": 0.0,
            "electron_current": 0.0,
            "hole_current": 0.0,
            "Electron density": 0.0,
            "Hole density": 0.0
        },
        "solar_cell_parameters": {
            "total_current": 0.0,
            "reverse_saturation_current": 0.0,
            "voc": 0.0,
            "rad_per_hour": 0.0
        }
    }
    # Electron and Hole Density Data
    electron_density_data: dict = {
        "positions": [],
        "values": []
    }
    hole_density_data: dict = {
        "positions": [],
        "values": []
    }

# Global variables for simulation results
current_simulation_results: List[SimulationResult] = []
current_simulation_params: Optional[SimulationResponse] = None

# Global variables for radiation and generation parameters
E = 1.58956743543344983e-18  # Minimum Energy per EHP in Joules
hv = 0.0  # Photon energy in Joules

def find_closest_mass_attenuation(photon_energy_ev: float) -> float:
    """Find the closest matching mass attenuation value from the CSV file."""
    # Convert eV to MeV (1 eV = 1e-6 MeV)
    photon_energy_mev = photon_energy_ev * 1e-6
    
    # Read the CSV file
    csv_path = os.path.join(root_dir, 'attentuationData.csv')
    energies = []
    mass_attenuations = []
    
    with open(csv_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row
        for row in csv_reader:
            if len(row) >= 2:
                energy = float(row[0])
                mass_attenuation = float(row[1])
                energies.append(energy)
                mass_attenuations.append(mass_attenuation)
    
    # Find the closest energy value
    energies_array = np.array(energies)
    closest_idx = np.argmin(np.abs(energies_array - photon_energy_mev))
    
    return mass_attenuations[closest_idx]

def generate_electric_field_data(params: SimulationParams, p_side_width: float, intrinsic_width: float, n_side_width: float) -> Tuple[List[Tuple[float, float]], List[float]]:
    """Generate electric field data using ElecFieldAsX function."""
    # Set up parameters for the ElecFieldAsX function
    nSide = float(params.donorConcentration)  # Ensure float conversion
    pSide = float(params.acceptorConcentration)
    iSide = float(params.intrinsicConcentration if params.intrinsicConcentration is not None else 0)
    
    # Use calculated widths instead of input region widths
    xn = n_side_width
    xp = p_side_width
    W = intrinsic_width

    # Create position array with different ranges based on junction type
    num_points = 1000  # Number of points for smooth plotting
    if params.junctionType == 'PIN' and params.intrinsicConcentration is not None:
        # For PIN junction, go from -xn to xp+W
        x_positions = np.linspace(-xn, xp + W, num_points)
    else:
        # For PN junction, go from -xn to xp
        x_positions = np.linspace(-xn, xp, num_points)
    
    # Calculate electric field for each position
    results = []
    positions = []
    for x in x_positions:
        try:
            # Call ElecFieldAsX with all required parameters
            e_field = ElecFieldAsX(
                x=x,
                nSide=nSide,
                pSide=pSide,
                iSide=iSide,
                xp=xp,
                W=W,
                xn=xn
            )
            results.append((float(x), float(e_field)))  # Ensure float conversion
            positions.append(float(x))  # Store position separately
        except Exception as e:
            # print(f"Error calculating electric field at x={x}: {str(e)}")
            results.append((float(x), 0.0))  # Fallback value
            positions.append(float(x))
    
    return results, positions

@app.post("/api/simulate")
async def run_simulation(params: SimulationParams) -> SimulationResponse:
    logger.info("Starting simulation...")
    logger.info(f"Received parameters: {params}")
    
    try:
        # Convert input parameters to float without validation
        logger.info("Converting input parameters...")
        donor_conc = float(str(params.donorConcentration))
        acceptor_conc = float(str(params.acceptorConcentration))
        temp = float(params.temperature)
        n_width = float(params.nRegionWidth)
        p_width = float(params.pRegionWidth)
        intrinsic_conc = float(params.intrinsicConcentration) if params.intrinsicConcentration is not None else 1e14
        
        logger.info(f"Converted parameters: donor_conc={donor_conc}, acceptor_conc={acceptor_conc}, temp={temp}")
        
        # Calculate photon energy in Joules
        global hv
        hv = float(params.photonEnergy) * 1.60218e-19
        logger.info(f"Calculated photon energy: {hv} Joules")
        
        # Find the closest mass attenuation value
        logger.info("Finding mass attenuation...")
        mass_attenuation = find_closest_mass_attenuation(float(params.photonEnergy))
        linear_attenuation = 6.15 * mass_attenuation
        logger.info(f"Mass attenuation: {mass_attenuation}, Linear attenuation: {linear_attenuation}")

        # Calculate built-in potential and depletion width
        logger.info("Calculating built-in potential...")
        built_in_potential = phase1_builtInPotential(
            float(params.temperature),
            float(params.donorConcentration),
            float(params.acceptorConcentration)
        )
        logger.info(f"Built-in potential: {built_in_potential} V")

        # Calculate depletion width based on junction type
        if params.junctionType == 'PIN' and params.intrinsicConcentration is not None:
            # For PIN junction, calculate depletion widths using acceptor and intrinsic concentrations
            dep_width_result = depletionWidth(
                builtInPotential=built_in_potential,
                concElectron=float(params.intrinsicConcentration),   # Use intrinsic concentration
                concHole=float(params.acceptorConcentration),        # Use acceptor concentration
                appliedVoltage=0.0
            )
            # depletionWidth returns [total, intrinsic, p_side] in cm
            intrinsic_width = dep_width_result[1] * ((params.percent / 100) if params.percent is not None else 1)  # Scale by user percentage
            p_side_width = dep_width_result[2]  # Third value is p-side width (cm)
            
            # Calculate n-side width using charge neutrality
            n_side_width = ((float(params.acceptorConcentration) * p_side_width) - 
                          (float(params.intrinsicConcentration) * intrinsic_width)) / float(params.donorConcentration)
            
            # Ensure all widths are positive
            if n_side_width < 0:
                n_side_width = abs(n_side_width)
            if p_side_width < 0:
                p_side_width = abs(p_side_width)
            if intrinsic_width < 0:
                intrinsic_width = abs(intrinsic_width)
        else:
            # For PN junction, use n-side and p-side concentrations
            dep_width_result = depletionWidth(
                builtInPotential=built_in_potential,
                concElectron=float(params.donorConcentration),
                concHole=float(params.acceptorConcentration),
                appliedVoltage=0.0
            )
            # For PN junction, depletionWidth returns [total, n_side, p_side] in cm
            n_side_width = dep_width_result[1]  # Second value is n-side width (cm)
            p_side_width = dep_width_result[2]  # Third value is p-side width (cm)
            intrinsic_width = 0.0  # No intrinsic region for PN junction

        total_depletion_width = p_side_width + intrinsic_width + n_side_width

        # print(f"PIN Junction Calculations:")
        # print(f"Built-in potential: {built_in_potential} V")
        # print(f"P-side width: {p_side_width} cm")
        # print(f"N-side width: {n_side_width} cm")
        # print(f"Intrinsic width: {intrinsic_width} cm")
        # print(f"Total depletion width: {total_depletion_width} cm")

        # Calculate electric field data
        electric_field_data, position_data = generate_electric_field_data(
            params, p_side_width, intrinsic_width, n_side_width
        )

        # Calculate eMin values
        e_min_values = eMin(total_depletion_width)
        e_min_electron = e_min_values[0]
        e_min_hole = e_min_values[1]
        e_min = max(e_min_electron, e_min_hole)

        # Calculate total electric field
        total_electric_field = abs(totalElectricField(
            nSide=donor_conc,
            pSide=acceptor_conc,
            iSide=0.0 if params.junctionType == 'PN' else intrinsic_conc,  # Use 0 for PN junction
            xp=p_side_width,
            W=intrinsic_width,
            xn=n_side_width,
            start=-n_side_width,
            end=p_side_width + intrinsic_width
        )) * 100  # Convert from V/cm to V/m and take absolute value

        # print(f"Total electric field: {total_electric_field} V/m")
        # print(f"Depletion widths - p-side: {p_side_width}, intrinsic: {intrinsic_width}, n-side: {n_side_width}")

        # Calculate device sizes (all in micrometers)
        # For emitter: input width - n-side depletion width
        emitter_size = n_width - (n_side_width * 10000)  # Subtract depletion width from input width
        # For base: input width - p-side depletion width
        base_size = p_width - (p_side_width * 10000)  # Subtract depletion width from input width
        
        # Check for small device sizes
        size_warning = None
        if emitter_size < 1 or base_size < 1:
            warnings = []
            if emitter_size < 1:
                warnings.append(f"Emitter Size ({emitter_size:.2f} μm)")
            if base_size < 1:
                warnings.append(f"Base Size ({base_size:.2f} μm)")
            size_warning = f"Warning: The following device regions are too small (< 1 μm): {', '.join(warnings)}"

        # print(f"\nDevice Size Calculations:")
        # print(f"Input n_width: {n_width} μm")
        # print(f"Input p_width: {p_width} μm")
        # print(f"n_side_width (depletion): {n_side_width} cm = {n_side_width * 10000} μm")
        # print(f"p_side_width (depletion): {p_side_width} cm = {p_side_width * 10000} μm")
        # print(f"Calculated emitter_size: {emitter_size} μm (input - n-side depletion)")
        # print(f"Calculated base_size: {base_size} μm (input - p-side depletion)")
        # print(f"Total device width: {emitter_size + (intrinsic_width * 10000) + base_size} μm")

        # Calculate minority recombination rates for Emitter (N-Doped) region
        n_side = float(params.donorConcentration)
        minority_conc = (1.6e-10)**2 / n_side
        
        minority_rates_emitter = {
            "region": "Emitter (N-Doped)",
            "auger": augerLifetime(minority_conc, n_side, 'minority'),
            "srh": srhLifetime()[1],
            "radiative": radiativeRecombination(minority_conc, n_side),
            "bulk": bulkRecombinationLifetime(minority_conc, n_side, 'p')
        }

        # Calculate minority recombination rates for Base (P-Doped) region
        p_side = float(params.acceptorConcentration)
        minority_conc_p = (1.6e-10)**2 / p_side
        
        minority_rates_base = {
            "region": "Base (P-Doped)",
            "auger": augerLifetime(minority_conc_p, p_side, 'minority'),
            "srh": srhLifetime()[0],
            "radiative": radiativeRecombination(minority_conc_p, p_side),
            "bulk": bulkRecombinationLifetime(minority_conc_p, p_side, 'n')
        }
        
        # Calculate surface recombination velocities
        logger.info("Calculating surface recombination velocities...")
        
        # Calculate diffusion constants for both regions
        diffusion_emitter = diffusionConstants(temp, mobility('p', minority_conc, temp))
        diffusion_base = diffusionConstants(temp, mobility('n', minority_conc_p, temp))
        
        # print("\nSurface Recombination Velocity Calculations:")
        # print(f"Emitter diffusion constant: {diffusion_emitter:.2e} cm²/s")
        # print(f"Base diffusion constant: {diffusion_base:.2e} cm²/s")
        
        # Calculate surface recombination velocities
        # Emitter (N-doped region) calculations
        minorityMobility = mobility('p', (1.6e-10**2)/params.donorConcentration, temp)
        minorityDiffusivity = diffusionConstants(temp, minorityMobility)
        lifetime = surfaceRecombinationLifetimeBare(minorityDiffusivity, emitter_size)
        bare_emitter_velocity = surfaceVelocity(minorityDiffusivity, emitter_size, lifetime)
        # print(f"Bare emitter velocity: {bare_emitter_velocity:.2e} cm/s")

        # Base (P-doped region) calculations
        minorityMobilityBase = mobility('n', (1.6e-10**2)/params.acceptorConcentration, temp)
        minorityDiffusivityBase = diffusionConstants(temp, minorityMobilityBase)
        lifetimeBase = surfaceRecombinationLifetimeInterface(minorityDiffusivityBase)
        substrate_base_velocity = surfaceVelocityInterface(minorityDiffusivityBase, base_size)
        # print(f"Substrate base velocity: {substrate_base_velocity:.2e} cm/s")

        # Update surface recombination velocities in response
        surface_recombination_velocities = {
            "bareEmitter": float(bare_emitter_velocity),
            "substrateBase": float(substrate_base_velocity)
        }
        
        logger.info(f"Surface recombination velocities: {surface_recombination_velocities}")
        
        # Calculate JDR and density profiles
        # print("\nCalculating JDR and density profiles...")
        
        # Initial lifetime estimates
        estimateN = 10e-9  # Initial estimate for electron lifetime
        estimateP = 1e-9   # Initial estimate for hole lifetime
        
        # Get the percent value, default to 100 if not provided
        percent = params.percent if params.percent is not None else 100
        
        # Mobility values
        mobilityN = 1000.0  # cm²/V·s for electrons
        mobilityP = 40.0    # cm²/V·s for holes
        
        # Convert concentrations to float
        nSide = float(params.donorConcentration)
        pSide = float(params.acceptorConcentration)
        iSide = float(params.intrinsicConcentration) if params.intrinsicConcentration is not None else 0
        
        # Use the previously calculated depletion widths
        xn = n_side_width
        xp = p_side_width
        W = intrinsic_width
        
        # Calculate lifetimes using JDR
        # print("\nCalculating lifetimes using JDR...")
        # print(f"Initial estimates:")
        # print(f"  Electron lifetime: {estimateN:.2e} s")
        # print(f"  Hole lifetime: {estimateP:.2e} s")
        
        # Convert photon energy to eV and calculate hv
        photon_energy_ev = float(params.photonEnergy)
        hv = photon_energy_ev * 1.60218e-19  # Convert eV to Joules
        
        # Get user inputted photon flux
        photon_flux = float(params.photonFlux)
        
        # Call findConvergenceLifetime with hv and photonFlux
        converged_lifetimes = findConvergenceLifetime(
            estimateN, estimateP, xn, xp, W, 
            donor_conc, acceptor_conc, intrinsic_conc, 
            0.5, mobilityN, mobilityP, hv, photon_flux
        )
        
        electron_lifetime = converged_lifetimes[0]
        hole_lifetime = converged_lifetimes[1]
        jdr_result = converged_lifetimes[2]
        
        # print(f"Converged lifetimes:")
        # print(f"  Electron lifetime: {electron_lifetime:.2e} s")
        # print(f"  Hole lifetime: {hole_lifetime:.2e} s")

        # Extract and log raw lifetime data
        # print("\nRaw Lifetime Data from JDR:")
        # print(f"Raw Electron-lifetime: {jdr_result.get('Electron-lifetime')}")
        # print(f"Raw Electron-Lifetime-Components: {jdr_result.get('Electron-Lifetime-Components')}")
        # print(f"Raw Hole-lifetime: {jdr_result.get('Hole-lifetime')}")
        # print(f"Raw Hole-Lifetime-Components: {jdr_result.get('Hole-Lifetime-Components')}")

        # Extract lifetimes from JDR results with detailed logging
        electron_lifetime_components = []
        for i, comp in enumerate(['Radiative', 'Auger', 'SRH']):
            try:
                val = float(jdr_result.get(f"{comp}-lifetime", 0.0))
                # print(f"  {comp}: {val:.2e}")
                electron_lifetime_components.append(val)
            except (ValueError, TypeError) as e:
                # print(f"  Error converting {comp} component: {e}")
                electron_lifetime_components.append(0.0)

        hole_lifetime_components = []
        for i, comp in enumerate(['Radiative', 'Auger', 'SRH']):
            try:
                val = float(jdr_result.get(f"{comp}-lifetime-components", [0.0, 0.0, 0.0])[i])
                # print(f"  {comp}: {val:.2e}")
                hole_lifetime_components.append(val)
            except (ValueError, TypeError) as e:
                # print(f"  Error converting {comp} component: {e}")
                hole_lifetime_components.append(0.0)

        # print("\nFinal Processed Lifetime Components:")
        # print(f"Electron: {electron_lifetime_components}")
        # print(f"Hole: {hole_lifetime_components}")

        # Calculate total width for generation rate (including emitter and base)
        if params.junctionType == 'PIN' and params.intrinsicConcentration is not None:
            total_width_cm = (n_width/10000) + (p_width/10000) + intrinsic_width  # Convert μm to cm and include intrinsic width
        else:
            total_width_cm = (n_width/10000) + (p_width/10000)  # Convert μm to cm
        # print(f"Total width for generation rate: {total_width_cm:.2e} cm")

        # Calculate generation rate data points
        num_points = 100  # Number of points for smooth plotting
        x_values = np.linspace(0, total_width_cm, num_points)  # Create evenly spaced points
        generation_rate_data = []
        generation_rate_profile = {"positions": [], "values": []}
        
        # Initialize lists to store generation rates for each region
        emitter_rates = []
        depletion_rates = []
        base_rates = []
        
        # Define region boundaries
        emitter_end = emitter_size/10000  # Convert μm to cm
        depletion_end = emitter_end + total_depletion_width
        
        for x in x_values:
            gen_rate = genDevice(x, total_width_cm, hv, float(params.photonFlux), E, linear_attenuation)
            generation_rate_data.append((float(x), float(gen_rate[1])))  # Store position (cm) and generation rate
            generation_rate_profile["positions"].append(float(x))
            generation_rate_profile["values"].append(float(gen_rate[1]))
            
            # Categorize generation rates by region
            if x <= emitter_end:
                emitter_rates.append(float(gen_rate[1]))
            elif x <= depletion_end:
                depletion_rates.append(float(gen_rate[1]))
            else:
                base_rates.append(float(gen_rate[1]))
        
        # Calculate average generation rates per region
        avg_emitter_rate = sum(emitter_rates) / len(emitter_rates) if emitter_rates else 0
        avg_depletion_rate = sum(depletion_rates) / len(depletion_rates) if depletion_rates else 0
        avg_base_rate = sum(base_rates) / len(base_rates) if base_rates else 0
        
        # print("\nAverage Generation Rates:")
        # print(f"Emitter: {avg_emitter_rate:.2e} cm⁻³s⁻¹")
        # print(f"Depletion: {avg_depletion_rate:.2e} cm⁻³s⁻¹")
        # print(f"Base: {avg_base_rate:.2e} cm⁻³s⁻¹")
        
        generation_per_region = {
            "emitter": float(avg_emitter_rate),
            "depletion": float(avg_depletion_rate),
            "base": float(avg_base_rate)
        }

        # Calculate current in emitter region
        # print("\nCalculating current in emitter region...")
        # print(f"Je parameters:")
        # print(f"  temp: {temp}")
        # print(f"  typeOfMinority: 'p'")
        # print(f"  majorityConc (donor_conc): {donor_conc:.2e} cm⁻³")
        # print(f"  width (emitter_size/10000): {emitter_size/10000:.2e} cm")
        # print(f"  linear_attenuation: {linear_attenuation:.2e} cm⁻¹")
        emitter_current = Je(temp, 'p', donor_conc, emitter_size/10000, linear_attenuation)
        emitter_current_float = float(emitter_current) * photon_flux
        # print(f"Emitter current: {emitter_current_float:.2e} A/cm²")

        # Calculate current in base region
        # print("\nCalculating current in base region...")
        # print(f"Jb parameters:")
        # print(f"  temp: {temp}")
        # print(f"  typeOfMinority: 'n'")
        # print(f"  majorityConc (acceptor_conc): {acceptor_conc:.2e} cm⁻³")
        # print(f"  width1 (emitter_size/10000): {emitter_size/10000:.2e} cm")
        # print(f"  width2 (base_size/10000): {base_size/10000:.2e} cm")
        # print(f"  depletionWidth: {total_depletion_width:.2e} cm")
        # print(f"  linear_attenuation: {linear_attenuation:.2e} cm⁻¹")
        base_current = Jb(temp, 'n', acceptor_conc, emitter_size/10000, base_size/10000, total_depletion_width, linear_attenuation)
        base_current_float = float(base_current) * photon_flux
        # print(f"Base current: {base_current_float:.2e} A/cm²")

        # Calculate reverse saturation current
        # print("\nCalculating reverse saturation current...")
        # print(f"Parameters:")
        # print(f"  temp: {temp}")
        # print(f"  donorConc: {donor_conc:.2e} cm⁻³")
        # print(f"  acceptorConc: {acceptor_conc:.2e} cm⁻³")
        # print(f"  emitterWidth: {emitter_size/10000:.2e} cm")
        # print(f"  baseWidth: {base_size/10000:.2e} cm")
        reverse_sat_current = reverseSaturationCurrent(temp, donor_conc, acceptor_conc, emitter_size/10000, base_size/10000)
        reverse_sat_current_float = abs(float(reverse_sat_current))
        # print(f"Reverse saturation current: {reverse_sat_current_float:.2e} A/cm²")

        # Calculate total current (sum of emitter, base, and JDR currents)
        jdr_current = float(jdr_result.get("Current", 0.0) or 0.0)
        total_current = emitter_current_float + base_current_float + jdr_current
        # print(f"\nTotal current: {total_current:.2e} A/cm²")
        # print(f"  Emitter current: {emitter_current_float:.2e} A/cm²")
        # print(f"  Base current: {base_current_float:.2e} A/cm²")
        # print(f"  JDR current: {jdr_current:.2e} A/cm²")

        # Calculate Voc
        # print("\nCalculating Voc...")
        # print(f"Parameters:")
        # print(f"  totalCurrent: {total_current:.2e} A/cm²")
        # print(f"  reverseSaturationCurrent: {reverse_sat_current_float:.2e} A/cm²")
        # print(f"  temp: {temp}")
        voc = Voc(total_current, reverse_sat_current_float, temp)
        voc_float = float(voc)
        # print(f"Voc: {voc_float:.2e} V")
        
        # Calculate radiation per hour
        # print("\nCalculating radiation per hour...")
        # print(f"Parameters:")
        # print(f"  photonFlux: {photon_flux:.2e} photons/cm²·s")
        total_device_width = (emitter_size + base_size)/10000 + total_depletion_width  # Convert μm to cm
        # print(f"  totalDeviceWidth: {total_device_width:.2e} cm")
        # print(f"  linearAttenuation: {linear_attenuation:.2e} cm⁻¹")
        rad_per_hour = radPerHour(photon_flux, total_device_width, linear_attenuation)
        rad_per_hour_float = float(rad_per_hour)
        # print(f"Radiation per hour: {rad_per_hour_float:.2e} rad/hour")

        # Calculate surviving carriers for both emitter and base
        emitter_surviving_carriers = emitter_current_float / q if emitter_current_float != 0 else 0
        base_surviving_carriers = base_current_float / q if base_current_float != 0 else 0

        # Create current generation data
        current_generation_data = {
            "emitter": {
                "generated_carriers": float(avg_emitter_rate),
                "surviving_carriers": float(emitter_surviving_carriers),
                "current": float(emitter_current_float)
            },
            "depletion": {
                "generated_carriers": float(avg_depletion_rate),
                "surviving_carriers": 0.0,
                "current": 0.0
            },
            "base": {
                "generated_carriers": float(avg_base_rate),
                "surviving_carriers": float(base_surviving_carriers),
                "current": float(base_current_float)
            },
            "jdr": {
                "current": float(jdr_result.get("Current", 0.0) or 0.0),
                "electron_current": float(jdr_result.get("Current-Electron", 0.0) or 0.0),
                "hole_current": float(jdr_result.get("Current-Hole", 0.0) or 0.0),
                "Electron density": float(jdr_result.get("Electron-density", [0.0])[-1] or 0.0),
                "Hole density": float(jdr_result.get("Hole-density", [0.0])[-1] or 0.0),
                "Electron-lifetime": electron_lifetime,
                "Electron-Lifetime-Components": electron_lifetime_components,
                "Hole-lifetime": hole_lifetime,
                "Hole-Lifetime-Components": hole_lifetime_components
            },
            "solar_cell_parameters": {
                "total_current": float(total_current),
                "reverse_saturation_current": float(reverse_sat_current_float),
                "voc": float(voc_float),
                "rad_per_hour": float(rad_per_hour_float)
            }
        }

        # Create density data with equal increments from 0 to total depletion width
        num_points = 100  # Number of points for the density profiles
        total_width = total_depletion_width  # Total depletion width in cm
        x_positions = np.linspace(0, total_width, num_points)  # Generate positions from 0 to total width
        
        # Round positions to 2 decimal places
        x_positions = np.round(x_positions, decimals=2)
        
        # Create electron density data
        electron_density_data = {
            "values": [float(val) if val is not None else 0.0 for val in jdr_result["Electron-density"]],
            "positions": [float(pos) for pos in x_positions]  # Use the rounded positions
        }
        
        # Create hole density data
        hole_density_data = {
            "values": [float(val) if val is not None else 0.0 for val in jdr_result["Hole-density"]],
            "positions": [float(pos) for pos in x_positions]  # Use the same rounded positions
        }

        # print("\nDensity Profile Data:")
        # print(f"Total depletion width: {total_width:.2e} cm")
        # print(f"Number of points: {num_points}")
        # print(f"X-axis range: 0 to {total_width:.2e} cm")

        # Calculate carrier densities
        emitter_carrier_density = avg_emitter_rate  # Use average generation rate
        depletion_carrier_density = avg_depletion_rate  # Use average generation rate
        base_carrier_density = avg_base_rate  # Use average generation rate

        # print("\nCarrier Density Calculations:")
        # print(f"Emitter carrier density: {emitter_carrier_density:.2e} cm⁻³")
        # print(f"Base carrier density: {base_carrier_density:.2e} cm⁻³")
        # if params.junctionType == "PIN":
        #     print(f"Depletion carrier density: {depletion_carrier_density:.2e} cm⁻³")

        # Print detailed generation rate information
        # print("\nDetailed Generation Rate Information:")
        # print(f"Number of points in emitter region: {len(emitter_rates)}")
        # print(f"Number of points in depletion region: {len(depletion_rates)}")
        # print(f"Number of points in base region: {len(base_rates)}")
        # print(f"\nGeneration Rate Ranges:")
        # if emitter_rates:
        #     print(f"Emitter: {min(emitter_rates):.2e} to {max(emitter_rates):.2e} cm⁻³s⁻¹")
        # if depletion_rates:
        #     print(f"Depletion: {min(depletion_rates):.2e} to {max(depletion_rates):.2e} cm⁻³s⁻¹")
        # if base_rates:
        #     print(f"Base: {min(base_rates):.2e} to {max(base_rates):.2e} cm⁻³s⁻¹")

        # Create simulation response
        simulation_response = SimulationResponse(
            builtInPotential=built_in_potential,
            depletionWidth=total_depletion_width,
            electricField=electric_field_data,
            position=position_data,
            temperature=temp,
            electronConcentration=float(params.donorConcentration),
            holeConcentration=float(params.acceptorConcentration),
            message="Simulation completed successfully",
            pSideWidth=p_side_width,
            intrinsicWidth=intrinsic_width,
            nSideWidth=n_side_width,
            eMinElectron=e_min_electron,
            eMinHole=e_min_hole,
            eMin=e_min,
            totalElectricField=total_electric_field,
            photonFlux=params.photonFlux,
            photonEnergy=params.photonEnergy,
            emitterSize=emitter_size,
            baseSize=base_size,
            totalDeviceWidth=emitter_size + (intrinsic_width * 10000) + base_size,
            sizeWarning=size_warning,
            ganDensity=6.15,
            massAttenuation=mass_attenuation,
            linearAttenuation=linear_attenuation,
            mobilityMaxElectrons=1000.0,
            mobilityMaxHoles=40.0,
            mobilityMinElectrons=55.0,
            mobilityMinHoles=3.0,
            intrinsicCarrierConcentration=1.6e-10,
            dielectricConstant=8.9,
            radiativeRecombinationCoefficient=1.1e-8,
            augerCoefficient=1e-30,
            electronThermalVelocity=2.43e7,
            holeThermalVelocity=2.38e7,
            surfaceRecombinationVelocities={
                "bareEmitter": float(bare_emitter_velocity),
                "substrateBase": float(substrate_base_velocity)
            },
            generationRateData=generation_rate_data,
            minorityRecombinationRates=[minority_rates_emitter, minority_rates_base],
            electron_density_data=electron_density_data,
            hole_density_data=hole_density_data,
            currentGenerationData=current_generation_data,
            generationRateProfile=generation_rate_profile
        )

        # Store results in global variables
        global current_simulation_results, current_simulation_params
        current_simulation_results = [
            SimulationResult(position=x, electricField=e_field)
            for x, e_field in electric_field_data
        ]
        current_simulation_params = simulation_response

        logger.info("Simulation completed successfully")
        return simulation_response
    
    except Exception as e:
        logger.error(f"Error in simulation: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error args: {e.args}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/simulate", response_model=SimulationResponse)
async def get_simulation_params():
    if current_simulation_params is None:
        raise HTTPException(status_code=404, detail="No simulation has been run yet")
    return current_simulation_params

@app.get("/api/results", response_model=List[SimulationResult])
async def get_results():
    print("Received request for results")
    print(f"Number of results available: {len(current_simulation_results)}")
    return current_simulation_results

@app.post("/api/save-sweep-results")
async def save_sweep_results(results: List[dict]):
    """Save parameter sweep simulation results to a CSV file with detailed data."""
    logger.info(f"Saving {len(results)} detailed parameter sweep results to results.csv")
    
    try:
        # Define the path to the results.csv file in the root directory
        results_file = Path(root_dir) / "results.csv"
        
        # Check if file exists to determine if we need to write headers
        file_exists = results_file.exists()
        
        # Open the file in append mode
        with open(results_file, 'a', newline='') as f:
            # Get field names from the first result if available
            if results and len(results) > 0:
                fieldnames = results[0].keys()
            else:
                fieldnames = ['donorConcentration', 'acceptorConcentration', 'percent', 
                            'totalCurrent', 'voc', 'radPerHour', 'deviceLifetime']
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Write header only if file doesn't exist
            if not file_exists:
                writer.writeheader()
            
            # Write all results
            writer.writerows(results)
        
        logger.info(f"Successfully saved results to {results_file}")
        return {"success": True, "message": f"Successfully saved {len(results)} results to {results_file}"}
    
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to save results: {str(e)}")

@app.post("/api/save-detailed-results")
async def save_detailed_results(simulations: List[dict]):
    """Save comprehensive simulation results with all data to a detailed CSV file."""
    logger.info(f"Saving {len(simulations)} detailed simulation results")
    
    try:
        # Define the path to the detailed results file in the root directory
        results_file = Path(root_dir) / "detailed_results.csv"
        
        # Extract all possible fields from the simulation results
        all_fields = set()
        for sim in simulations:
            # Flatten nested dictionaries with dot notation
            flat_dict = {}
            
            def flatten_dict(d, parent_key=''):
                for k, v in d.items():
                    new_key = f"{parent_key}.{k}" if parent_key else k
                    
                    if isinstance(v, dict):
                        flatten_dict(v, new_key)
                    elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                        # Handle list of dictionaries (extract only the first item)
                        flatten_dict(v[0], f"{new_key}[0]")
                    else:
                        # Handle scalar values, ensuring they're converted to strings
                        if isinstance(v, (int, float, str, bool)) or v is None:
                            flat_dict[new_key] = v
                        else:
                            # For complex objects like arrays, convert to string representation
                            flat_dict[new_key] = str(v)
            
            flatten_dict(sim)
            all_fields.update(flat_dict.keys())
        
        # Sort fields for consistent CSV columns
        fieldnames = sorted(list(all_fields))
        
        # Check if file exists to determine if we need to write headers
        file_exists = results_file.exists()
        
        # Prepare rows for writing
        rows_to_write = []
        for sim in simulations:
            flat_dict = {}
            
            def flatten_dict(d, parent_key=''):
                for k, v in d.items():
                    new_key = f"{parent_key}.{k}" if parent_key else k
                    
                    if isinstance(v, dict):
                        flatten_dict(v, new_key)
                    elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                        flatten_dict(v[0], f"{new_key}[0]")
                    else:
                        if isinstance(v, (int, float, str, bool)) or v is None:
                            flat_dict[new_key] = v
                        else:
                            flat_dict[new_key] = str(v)
            
            flatten_dict(sim)
            rows_to_write.append(flat_dict)
        
        # Write to CSV
        with open(results_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Write header only if file doesn't exist
            if not file_exists:
                writer.writeheader()
            
            # Write all flattened results
            writer.writerows(rows_to_write)
        
        logger.info(f"Successfully saved detailed results to {results_file}")
        return {
            "success": True, 
            "message": f"Successfully saved {len(simulations)} detailed simulation results to {results_file}",
            "filename": str(results_file)
        }
    
    except Exception as e:
        logger.error(f"Error saving detailed results: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to save detailed results: {str(e)}")

# Add a test endpoint to verify the server is running
@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info") 