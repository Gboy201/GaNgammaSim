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
import uuid
from mpmath import mp
from scipy.integrate import quad

# Set up mpmath precision
mp.dps = 25  # Set precision to 25 digits

# Define fundamental constants
q = 1.6e-19  # Elementary charge in Coulombs

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def safe_float(value):
    """Convert a value to a JSON-compliant float, handling edge cases."""
    try:
        if isinstance(value, (int, float)):
            # Check for infinity, NaN, or extremely large values
            if math.isinf(value) or math.isnan(value) or abs(value) > 1e308:
                return 0.0
            return float(value)
        elif hasattr(value, '__float__'):
            # Handle mpf and similar types
            result = float(value)
            if math.isinf(result) or math.isnan(result) or abs(result) > 1e308:
                return 0.0
            return result
        return 0.0
    except (ValueError, TypeError, OverflowError):
        return 0.0

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
    expose_headers=["*"],
    max_age=3600,
)

# Global variables for simulation tracking
current_simulation_results = []
current_simulation_params = None
simulation_counter = 0  # Counter for number of simulations performed
current_sweep_counter = 0  # Counter for current parameter sweep

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
    donorConcentration: float
    acceptorConcentration: float
    intrinsicConcentration: float
    nSideWidth: float
    pSideWidth: float
    intrinsicWidth: float
    totalDepletionWidth: float
    temperature: float
    junctionType: str
    builtInPotential: float
    depletionWidth: float
    electricField: float
    maxElectricField: float
    diffusionCurrent: float
    driftCurrent: float
    totalCurrent: float
    reverseSaturationCurrent: float
    voc: float
    radPerHour: float
    generationRateData: List[Tuple[float, float]]
    generationPerRegion: dict
    generationRateProfile: dict
    electronDensity: float
    holeDensity: float
    electronDensityData: dict
    holeDensityData: dict
    currentGenerationData: dict
    electricFieldData: dict
    baseSize: float  # Added for N-doped region width
    emitterSize: float  # Added for P-doped region width
    ganDensity: float
    massAttenuation: float
    linearAttenuation: float
    mobilityMaxElectrons: float
    mobilityMaxHoles: float
    mobilityMinElectrons: float
    mobilityMinHoles: float
    intrinsicCarrierConcentration: float
    dielectricConstant: float
    radiativeRecombinationCoefficient: float
    augerCoefficient: float
    electronThermalVelocity: float
    holeThermalVelocity: float
    electronDensityProfile: Optional[List[float]] = None
    holeDensityProfile: Optional[List[float]] = None

# Global variables for radiation and generation parameters
E = 1.58956743543344983e-18  # Minimum Energy per EHP in Joules
hv = 0.0  # Photon energy in Joules

# Define a helper function to flatten nested dictionaries for CSV export
def flatten_dict(d, result, parent_key=''):
    """
    Recursively flatten a nested dictionary using dot notation for keys.
    Args:
        d: The dictionary to flatten
        result: The dictionary to store the flattened result
        parent_key: The parent key prefix to use
    """
    if d is None:
        return
        
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            
            if isinstance(v, dict):
                flatten_dict(v, result, new_key)
            elif isinstance(v, list):
                # Handle lists of different types
                if len(v) > 0:
                    if isinstance(v[0], dict):
                        # For list of dictionaries, process at least the first item
                        flatten_dict(v[0], result, f"{new_key}[0]")
                    elif isinstance(v[0], (int, float, str, bool)):
                        # For primitive type lists, store the entire list as a string
                        result[new_key] = str(v)
                    else:
                        # For complex objects, store as string
                        result[new_key] = str(v)
                else:
                    # Empty list
                    result[new_key] = "[]"
            else:
                # Handle scalar values, ensuring they're converted appropriately
                if isinstance(v, (int, float, str, bool)) or v is None:
                    result[new_key] = v
                else:
                    # For complex objects like arrays, convert to string representation
                    result[new_key] = str(v)
    elif isinstance(d, list) and len(d) > 0 and isinstance(d[0], dict):
        # Handle list of dictionaries at the top level
        flatten_dict(d[0], result, parent_key)

# Custom JSON encoder to handle complex types for API responses
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__float__'):  # Handle mpf and similar types
            return float(obj)
        elif isinstance(obj, np.ndarray):  # Handle numpy arrays
            return obj.tolist()
        elif isinstance(obj, np.integer):  # Handle numpy integers
            return int(obj)
        elif isinstance(obj, np.floating):  # Handle numpy floats
            return float(obj)
        return super().default(obj)

def safe_json_dumps(obj, **kwargs):
    """Safely convert any object to JSON string, handling mpf and numpy types"""
    return json.dumps(obj, cls=CustomJSONEncoder, **kwargs)

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
async def run_simulation(sim_params: SimulationParams):
    try:
        # Increment simulation counters
        global simulation_counter, current_sweep_counter, current_simulation_results, current_simulation_params
        simulation_counter += 1
        current_sweep_counter += 1
        
        # Generate a unique simulation ID for logging
        simulation_id = f"sim-{datetime.now().strftime('%Y%m%d%H%M%S')}-{simulation_counter}"
        logging.info(f"[{simulation_id}] Starting simulation #{simulation_counter}")
        
        # Log all parameters in standardized format
        logging.info(f"[{simulation_id}] Received simulation request with parameters:")
        param_log = {
            "temperature": safe_float(sim_params.temperature),
            "donor_concentration": safe_float(sim_params.donorConcentration),
            "acceptor_concentration": safe_float(sim_params.acceptorConcentration),
            "n_region_width": safe_float(sim_params.nRegionWidth),
            "p_region_width": safe_float(sim_params.pRegionWidth),
            "photon_energy": safe_float(sim_params.photonEnergy),
            "photon_flux": safe_float(sim_params.photonFlux),
            "junction_type": sim_params.junctionType,
            "percent": safe_float(sim_params.percent) if sim_params.percent is not None else None,
            "intrinsic_concentration": safe_float(sim_params.intrinsicConcentration) if sim_params.intrinsicConcentration is not None else None
        }
        logging.info(f"[{simulation_id}] PARAMETERS: {safe_json_dumps(param_log, indent=2)}")
        
        # Extract parameters in the format expected by the simulation code
        temperature = float(sim_params.temperature)
        donorConcentration = float(sim_params.donorConcentration)
        acceptorConcentration = float(sim_params.acceptorConcentration)
        nRegionWidth = float(sim_params.nRegionWidth)
        pRegionWidth = float(sim_params.pRegionWidth)
        photonEnergy = float(sim_params.photonEnergy)
        photonFlux = float(sim_params.photonFlux)
        junctionType = sim_params.junctionType
        percent = float(sim_params.percent) if sim_params.percent is not None else 0.0

        # Log information about the current simulation
        logging.info(f"[{simulation_id}] Starting simulation with T={temperature}K, donor={donorConcentration}cm^-3, acceptor={acceptorConcentration}cm^-3")
        
        # Calculate photon energy in Joules
        global hv
        hv = photonEnergy * 1.60218e-19
        logging.info(f"[{simulation_id}] Calculated photon energy: {hv} Joules")
        
        # Find the closest mass attenuation value
        logging.info(f"[{simulation_id}] Finding mass attenuation...")
        mass_attenuation = find_closest_mass_attenuation(photonEnergy)
        linear_attenuation = 6.15 * mass_attenuation
        logging.info(f"[{simulation_id}] Mass attenuation: {mass_attenuation}, Linear attenuation: {linear_attenuation}")

        # Calculate built-in potential and depletion width
        logging.info(f"[{simulation_id}] Calculating built-in potential...")
        built_in_potential = phase1_builtInPotential(
            temperature, donorConcentration, acceptorConcentration
        )
        logging.info(f"[{simulation_id}] Built-in potential: {built_in_potential} V")

        # Calculate depletion width based on junction type
        if junctionType == 'PIN' and sim_params.intrinsicConcentration is not None:
            # For PIN junction, calculate depletion widths using acceptor and intrinsic concentrations
            dep_width_result = depletionWidth(
                builtInPotential=built_in_potential,
                concElectron=float(sim_params.intrinsicConcentration),  # Use intrinsic concentration
                concHole=float(acceptorConcentration),  # Use acceptor concentration
                appliedVoltage=0.0
            )
            # depletionWidth returns [total, intrinsic, p_side] in cm
            depletion_width = dep_width_result[0]  # Total depletion width
            intrinsic_width = dep_width_result[1] * ((percent / 100) if percent is not None else 1)  # Scale by user percentage
            p_side_width = dep_width_result[2]  # Third value is p-side width (cm)
            
            # Calculate n-side width using charge neutrality
            n_side_width = ((float(acceptorConcentration) * p_side_width) - 
                          (float(sim_params.intrinsicConcentration) * intrinsic_width)) / float(donorConcentration)
            
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
                concElectron=float(donorConcentration),
                concHole=float(acceptorConcentration),
                appliedVoltage=0.0
            )
            # For PN junction, depletionWidth returns [total, n_side, p_side] in cm
            depletion_width = dep_width_result[0]  # Total depletion width
            n_side_width = dep_width_result[1]  # Second value is n-side width (cm)
            p_side_width = dep_width_result[2]  # Third value is p-side width (cm)
            intrinsic_width = 0.0  # No intrinsic region for PN junction

        total_depletion_width = p_side_width + intrinsic_width + n_side_width

        # Calculate electric field profile using ElecFieldAsX
        positions = np.linspace(-n_side_width, p_side_width + intrinsic_width, 1000)  # Create 1000 points
        electric_field_values = []
        for x in positions:
            field = ElecFieldAsX(
                x=x,
                nSide=donorConcentration,
                pSide=acceptorConcentration,
                iSide=0.0 if junctionType == 'PN' else sim_params.intrinsicConcentration,
                xp=p_side_width,
                W=intrinsic_width,
                xn=n_side_width
            )
            electric_field_values.append(float(field))

        # Calculate electric field data
        electric_field_data = {
            "positions": [float(x) for x in positions],
            "values": electric_field_values
        }

        # Calculate JDR and density profiles
        # Initial lifetime estimates
        estimateN = 10e-9  # Initial estimate for electron lifetime
        estimateP = 1e-9   # Initial estimate for hole lifetime
        
        # Calculate JDR and density profiles using findConvergenceLifetime
        convergence_results = findConvergenceLifetime(
            estimateN=estimateN,
            estimateP=estimateP,
            xn=n_side_width,
            xp=p_side_width,
            W=intrinsic_width,
            nSide=donorConcentration,
            pSide=acceptorConcentration,
            iSide=0.0 if junctionType == 'PN' else sim_params.intrinsicConcentration,
            percent=1.0,  # Using full width
            mobilityN=1000.0,  # Using default mobility for electrons
            mobilityP=40.0,    # Using default mobility for holes
            hv=photonEnergy * 1.60218e-19,  # Convert eV to Joules
            photonFlux=photonFlux
        )
        
        # Extract results from convergence calculation
        electron_lifetime, hole_lifetime, jdr_results = convergence_results
        
        # Extract electric field data from JDR results
        electric_field = float(jdr_results['Efield'])
        e_min = float(jdr_results['Emin'])

        # Calculate diffusion and drift currents using JDR results
        diffusion_current = float(jdr_results['Current-Electron'] + jdr_results['Current-Hole'])
        drift_current = float(electric_field * (float(jdr_results['Electron density']) + float(jdr_results['Hole density'])))

        # Extract JDR results and convert mpf to float
        electron_current = float(jdr_results['Current-Electron'])
        hole_current = float(jdr_results['Current-Hole'])
        electron_density = float(jdr_results['Electron density'])
        hole_density = float(jdr_results['Hole density'])
        electron_density_profile = [float(x) for x in jdr_results['Electron-density']]
        hole_density_profile = [float(x) for x in jdr_results['Hole-density']]
        electron_lifetime_components = [float(x) for x in jdr_results['Electron-Lifetime-Components']]
        hole_lifetime_components = [float(x) for x in jdr_results['Hole-Lifetime-Components']]
        
        # Calculate total current
        total_current = electron_current + hole_current
        
        # Calculate reverse saturation current
        logging.info(f"[{simulation_id}] Calculating reverse saturation current with parameters:")
        logging.info(f"[{simulation_id}] temperature: {temperature}")
        logging.info(f"[{simulation_id}] donorConc: {donorConcentration}")
        logging.info(f"[{simulation_id}] acceptorConc: {acceptorConcentration}")
        logging.info(f"[{simulation_id}] emitterWidth: {n_side_width}")
        logging.info(f"[{simulation_id}] baseWidth: {p_side_width}")
        
        reverse_saturation_current = reverseSaturationCurrent(
            temp=temperature,
            donorConc=donorConcentration,
            acceptorConc=acceptorConcentration,
            emitterWidth=n_side_width,
            baseWidth=p_side_width
        )
        
        logging.info(f"[{simulation_id}] Calculated reverse saturation current: {reverse_saturation_current}")
        
        # Calculate Voc
        logging.info(f"[{simulation_id}] Calculating Voc with parameters:")
        logging.info(f"[{simulation_id}] totalCurrent: {total_current}")
        logging.info(f"[{simulation_id}] reverseSaturationCurrent: {reverse_saturation_current}")
        logging.info(f"[{simulation_id}] temperature: {temperature}")
        logging.info(f"[{simulation_id}] totalCurrent/reverseSaturationCurrent: {total_current/reverse_saturation_current}")
        
        voc = Voc(
            totalCurrent=float(total_current),
            reverseSaturationCurrent=float(reverse_saturation_current),
            temp=float(temperature)
        )
        
        # Calculate rad per hour
        rad_per_hour = radPerHour(
            photonFlux=photonFlux,
            size=total_depletion_width,
            a=linear_attenuation
        )
        
        # Calculate generation rate data
        logging.info(f"[{simulation_id}] Calculating generation rate with parameters:")
        logging.info(f"[{simulation_id}] total_depletion_width: {total_depletion_width}")
        logging.info(f"[{simulation_id}] photonEnergy (eV): {photonEnergy}")
        logging.info(f"[{simulation_id}] photonFlux: {photonFlux}")
        logging.info(f"[{simulation_id}] E: {E}")
        logging.info(f"[{simulation_id}] linear_attenuation: {linear_attenuation}")
        
        # Convert photon energy from eV to Joules
        hv_joules = photonEnergy * 1.60218e-19
        logging.info(f"[{simulation_id}] photonEnergy (Joules): {hv_joules}")
        
        generation_rate_values = []
        for x in positions:
            result = genDevice(
                x=x,
                width=total_depletion_width,
                hv=hv_joules,  # Use photon energy in Joules
                photonFlux=photonFlux,
                E=E,  # Use the global E constant
                a=linear_attenuation
            )
            generation_rate_values.append(result[1])  # Use Q(x) value
        
        # Calculate generation rate data
        generation_rate_data = [(float(x), float(y)) for x, y in zip(positions, generation_rate_values)]
        
        # Calculate generation per region
        region_size = len(generation_rate_values) // 3
        generation_per_region = {
            "emitter": float(sum(generation_rate_values[:region_size])),
            "depletion": float(sum(generation_rate_values[region_size:2*region_size])),
            "base": float(sum(generation_rate_values[2*region_size:]))
        }
        
        # Calculate generation rate profile
        generation_rate_profile = {
            "positions": [float(x) for x in positions],
            "values": [float(x) for x in generation_rate_values]
        }
        
        # Calculate current generation data
        current_generation_data = {
            "emitter": {
                "generated_carriers": float(generation_per_region["emitter"]),
                "current": float(electron_current * 0.3)  # Approximate 30% of current in emitter
            },
            "depletion": {
                "generated_carriers": float(generation_per_region["depletion"]),
                "current": float(electron_current * 0.4)  # Approximate 40% of current in depletion
            },
            "base": {
                "generated_carriers": float(generation_per_region["base"]),
                "current": float(electron_current * 0.3)  # Approximate 30% of current in base
            },
            "jdr": {
                "current": float(total_current),
                "Current-Electron": float(electron_current),  # Changed to match frontend expectation
                "Current-Hole": float(hole_current),  # Changed to match frontend expectation
                "Electron density": float(electron_density),
                "Hole density": float(hole_density),
                "Electron-lifetime": float(electron_lifetime),
                "Hole-lifetime": float(hole_lifetime),
                "Electron-Lifetime-Components": [float(x) for x in electron_lifetime_components],
                "Hole-Lifetime-Components": [float(x) for x in hole_lifetime_components],
                "Electric field": float(electric_field),
                "E-min": float(e_min)
            },
            "solar_cell_parameters": {
                "total_current": float(total_current),
                "reverse_saturation_current": float(reverse_saturation_current),
                "voc": float(voc),
                "rad_per_hour": float(rad_per_hour)
            }
        }
        
        # Log the carrier densities from findConvergenceLifetime results
        # print("\n*** CARRIER DENSITY CALCULATIONS ***")
        # print(f"Electron Density: {electron_density:.2e} cm^-3")
        # print(f"Hole Density: {hole_density:.2e} cm^-3")
        # print(f"Intrinsic Concentration: {sim_params.intrinsicConcentration:.2e} cm^-3")
        # print(f"Donor Concentration: {donorConcentration:.2e} cm^-3")
        # print(f"Acceptor Concentration: {acceptorConcentration:.2e} cm^-3")
        
        # print("\n*** ELECTRIC FIELD DATA ***")
        # print(f"Electric field: {electric_field:.2e} V/cm")
        # print(f"E-min: {e_min:.2e} V/cm")
        # print(f"Electric field profile points: {len(electric_field_values)}")
        # print(f"First few points: {electric_field_values[:5]}")
        # print(f"Last few points: {electric_field_values[-5:]}")

        # Create response with both single point and profile data
        response = SimulationResponse(
            donorConcentration=safe_float(donorConcentration),
            acceptorConcentration=safe_float(acceptorConcentration),
            intrinsicConcentration=safe_float(sim_params.intrinsicConcentration),
            nSideWidth=safe_float(n_side_width),
            pSideWidth=safe_float(p_side_width),
            intrinsicWidth=safe_float(intrinsic_width),
            totalDepletionWidth=safe_float(total_depletion_width),
            temperature=safe_float(temperature),
            junctionType=sim_params.junctionType,
            builtInPotential=safe_float(built_in_potential),
            depletionWidth=safe_float(depletion_width),
            electricField=safe_float(electric_field),
            maxElectricField=safe_float(max(electric_field_values)),  # Use max from profile
            diffusionCurrent=safe_float(diffusion_current),
            driftCurrent=safe_float(drift_current),
            totalCurrent=safe_float(total_current),
            reverseSaturationCurrent=safe_float(reverse_saturation_current),
            voc=safe_float(voc),
            radPerHour=safe_float(rad_per_hour),
            generationRateData=generation_rate_data,
            generationPerRegion=generation_per_region,
            generationRateProfile=generation_rate_profile,
            electronDensity=electron_density,
            holeDensity=hole_density,
            electronDensityData={
                "positions": [float(x) for x in positions],
                "values": electron_density_profile
            },
            holeDensityData={
                "positions": [float(x) for x in positions],
                "values": hole_density_profile
            },
            currentGenerationData=current_generation_data,
            electricFieldData={
                "positions": [float(x) for x in positions],
                "values": electric_field_values
            },
            baseSize=safe_float(nRegionWidth),  # Use nRegionWidth for base size
            emitterSize=safe_float(pRegionWidth),  # Use pRegionWidth for emitter size
            # Add constants from phase1.py
            ganDensity=6.15,  # g/cm³
            massAttenuation=safe_float(mass_attenuation),
            linearAttenuation=safe_float(linear_attenuation),
            mobilityMaxElectrons=1265,  # cm²/V·s
            mobilityMaxHoles=40,  # cm²/V·s
            mobilityMinElectrons=55,  # cm²/V·s
            mobilityMinHoles=3,  # cm²/V·s
            intrinsicCarrierConcentration=1.6e-10,  # cm⁻³
            dielectricConstant=8.9,  # Also 9.7 for wurtzite structure
            radiativeRecombinationCoefficient=1.1e-8,  # cm³/s
            augerCoefficient=1e-30,  # cm⁶/s
            electronThermalVelocity=2.43e7,  # cm/s
            holeThermalVelocity=2.38e7  # cm/s
        )

        # Store results in the global dictionary
        current_simulation_results = [
            SimulationResult(position=float(x), electricField=float(field))
            for x, field in zip(positions, electric_field_values)
        ]
        current_simulation_params = response

        # Log the response data for debugging
        logging.info(f"[{simulation_id}] Response data structure:")
        logging.info(f"[{simulation_id}] JDR data: {current_generation_data['jdr']}")
        logging.info(f"[{simulation_id}] Electric field data: {electric_field_data}")
        logging.info(f"[{simulation_id}] Electron density data: {response.electronDensityData}")
        logging.info(f"[{simulation_id}] Hole density data: {response.holeDensityData}")

        # Return the response
        return response
    
    except Exception as e:
        logging.error(f"Error in simulation: {str(e)}")
        logging.error(f"Error type: {type(e)}")
        logging.error(f"Error args: {e.args}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise

@app.get("/api/results", response_model=List[SimulationResult])
async def get_results():
    """Get the current simulation results."""
    if not current_simulation_results:
        raise HTTPException(status_code=404, detail="No simulation results available")
    return current_simulation_results

@app.get("/api/simulate", response_model=SimulationResponse)
async def get_simulation_params():
    """Get the current simulation parameters and results."""
    if current_simulation_params is None:
        raise HTTPException(status_code=404, detail="No simulation has been run yet")
    return current_simulation_params

@app.post("/api/save-sweep-results")
async def save_sweep_results(sweep_results: List[dict]):
    """
    Save simplified sweep results to a CSV file.
    Handles chunked data for large parameter sweeps.
    """
    try:
        # Generate a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sweep_results_{timestamp}.csv"
        current_dir = os.getcwd()
        filepath = os.path.join(current_dir, filename)
        
        # Log the file path information
        print(f"Saving sweep results to: {filepath}")
        logging.info(f"Saving sweep results to: {filepath}")
        
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.isfile(filepath)
        
        # Track successful and failed rows
        successful_rows = 0
        failed_rows = 0
        
        # Open file in append mode
        with open(filepath, 'a', newline='') as csvfile:
            if not file_exists and len(sweep_results) > 0:
                # Get fieldnames from the first item
                fieldnames = list(sweep_results[0].keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            elif file_exists and len(sweep_results) > 0:
                with open(filepath, 'r', newline='') as readfile:
                    reader = csv.reader(readfile)
                    fieldnames = next(reader)  # Get the headers from the existing file
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write each result row
            for idx, result in enumerate(sweep_results):
                try:
                    # Convert all values to native Python types for CSV compatibility
                    row_data = {k: float(v) if isinstance(v, (int, float)) else v for k, v in result.items()}
                    writer.writerow(row_data)
                    successful_rows += 1
                except Exception as e:
                    failed_rows += 1
                    logging.error(f"Error writing sweep result row {idx}: {str(e)}")
                    # Try to save the raw data for debugging
                    logging.error(f"Failed row data: {result}")
        
        # Log summary
        logging.info(f"Save summary: {successful_rows} rows saved successfully, {failed_rows} rows failed")
        if failed_rows > 0:
            logging.warning(f"Some rows failed to save. Check the logs for details.")
        
        return {
            "message": f"Sweep results saved to {filepath}",
            "filename": filepath,
            "successful_rows": successful_rows,
            "failed_rows": failed_rows
        }
    
    except Exception as e:
        logging.error(f"Error saving sweep results: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error saving sweep results: {str(e)}")

@app.post("/api/save-detailed-results")
async def save_detailed_results(detailed_results: List[dict]):
    """
    Save detailed simulation results to a CSV file.
    Handles chunked data for large parameter sweeps.
    """
    try:
        # Generate a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detailed_results_{timestamp}.csv"
        current_dir = os.getcwd()
        filepath = os.path.join(current_dir, filename)
        
        # Log the file path information
        print(f"Saving detailed results to: {filepath}")
        logging.info(f"Saving detailed results to: {filepath}")
        
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.isfile(filepath)
        
        # Open file in append mode so we can add chunks of data
        with open(filepath, 'a', newline='') as csvfile:
            # Process the first item to get all possible keys for headers
            if not file_exists and len(detailed_results) > 0:
                # Flatten the nested structure for the first item
                flattened_first_item = {}
                flatten_dict(detailed_results[0], flattened_first_item)
                
                # Write the CSV header using all keys from the flattened dict
                fieldnames = list(flattened_first_item.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            
            # If file exists, we need to get existing headers
            elif file_exists and len(detailed_results) > 0:
                with open(filepath, 'r', newline='') as readfile:
                    reader = csv.reader(readfile)
                    fieldnames = next(reader)  # Get the headers from the existing file
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Process each result and write to CSV
            for result in detailed_results:
                # Flatten the nested structure into a single-level dictionary
                flattened_item = {}
                flatten_dict(result, flattened_item)
                
                # Write the flattened data to CSV
                try:
                    writer.writerow(flattened_item)
                except ValueError as e:
                    # If we're missing some keys (might happen with chunked data), add them with null values
                    missing_keys = set(fieldnames) - set(flattened_item.keys())
                    for key in missing_keys:
                        flattened_item[key] = None
                    writer.writerow(flattened_item)
                except Exception as e:
                    logging.error(f"Error writing row to CSV: {str(e)}")
        
        print(f"Successfully saved {len(detailed_results)} detailed results to {filepath}")
        return {"message": f"Detailed results saved to {filepath}", "filename": filepath}
    
    except Exception as e:
        logging.error(f"Error saving detailed results: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error saving detailed results: {str(e)}")

@app.get("/api/debug/compare-simulations")
async def compare_simulations(sim1_id: str, sim2_id: str):
    """
    Debug endpoint to compare parameters between two simulations.
    Returns the differences in parameters and results to help diagnose inconsistencies.
    
    Args:
        sim1_id: First simulation ID (from logs)
        sim2_id: Second simulation ID (from logs)
    
    Returns:
        Comparison of parameters and results
    """
    try:
        # This is a placeholder implementation that would need to be connected to a database
        # or log parser in a production environment
        logging.info(f"Comparing simulations {sim1_id} and {sim2_id}")
        
        # In a real implementation, you would:
        # 1. Retrieve simulation parameters and results from a database or parsed logs
        # 2. Compare the parameters and results to find differences
        # 3. Return a structured comparison
        
        return {
            "message": "This is a placeholder endpoint. To use it properly, you would need to implement log parsing or database storage of simulation parameters and results.",
            "instructions": "Check the simulation_server.log file for entries with these simulation IDs to manually compare parameters",
            "simulation_ids": {
                "sim1": sim1_id,
                "sim2": sim2_id
            },
            "log_file_location": "web-app/backend/simulation_server.log"
        }
    except Exception as e:
        logging.error(f"Error comparing simulations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper function to extract lifetimes from logs for debugging
@app.get("/api/debug/extract-lifetimes")
async def extract_lifetimes():
    """
    Debug endpoint to extract and compare electron and hole lifetimes from simulation logs.
    This helps identify patterns in lifetime values across different simulation runs.
    
    Returns:
        A collection of lifetime values extracted from logs
    """
    try:
        import re
        import os
        
        # Path to the log file
        log_file = "simulation_server.log"
        
        if not os.path.exists(log_file):
            return {"error": f"Log file {log_file} not found"}
        
        # Patterns to match in logs
        electron_pattern = r'\[([^\]]+)\].*Electron lifetime: ([0-9.e+-]+)'
        hole_pattern = r'\[([^\]]+)\].*Hole lifetime: ([0-9.e+-]+)'
        
        # Store extracted data
        lifetimes = []
        
        # Open and read the log file
        with open(log_file, 'r') as f:
            content = f.read()
            
            # Find all matches
            electron_matches = re.findall(electron_pattern, content)
            hole_matches = re.findall(hole_pattern, content)
            
            # Create a dictionary of simulation IDs mapped to lifetimes
            sim_lifetimes = {}
            
            # Process electron lifetimes
            for sim_id, lifetime in electron_matches:
                if sim_id not in sim_lifetimes:
                    sim_lifetimes[sim_id] = {}
                sim_lifetimes[sim_id]['electron'] = float(lifetime)
            
            # Process hole lifetimes
            for sim_id, lifetime in hole_matches:
                if sim_id not in sim_lifetimes:
                    sim_lifetimes[sim_id] = {}
                sim_lifetimes[sim_id]['hole'] = float(lifetime)
            
            # Convert to a list of records with simulation ID and both lifetimes
            for sim_id, data in sim_lifetimes.items():
                if 'electron' in data and 'hole' in data:
                    lifetimes.append({
                        'simulation_id': sim_id,
                        'electron_lifetime': data['electron'],
                        'hole_lifetime': data['hole']
                    })
        
        return {
            "lifetimes": lifetimes,
            "count": len(lifetimes),
            "note": "These are the extracted electron and hole lifetimes from simulation logs"
        }
    except Exception as e:
        logging.error(f"Error extracting lifetimes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add a test endpoint to verify the server is running
@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

# Add a new endpoint to get the simulation count
@app.get("/api/simulation-count")
async def get_simulation_count():
    """Returns the current simulation count."""
    return {"total_simulations": current_sweep_counter}

# Add a new endpoint to reset the simulation counter
@app.post("/api/reset-simulation-count")
async def reset_simulation_count():
    """Resets the simulation counter for a new parameter sweep."""
    global simulation_counter, current_sweep_counter
    simulation_counter = 0
    current_sweep_counter = 0
    return {"message": "Simulation counter reset"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info") 