import React from "react";
import {
  Box,
  Heading,
  Spinner,
  Text,
  VStack,
  Center,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  TableContainer,
  SimpleGrid,
  Grid,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
} from "@chakra-ui/react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  LabelList,
} from "recharts";
import { useQuery } from "@tanstack/react-query";
import axios from "axios";

interface ElectricFieldData {
  position: number;
  electricField: number;
}

interface DensityData {
  position: number;
  density: number;
}

interface CurrentGenerationData {
  region: string;
  value: number;
}

interface SimulationData {
  electricField: [number, number][];
  position: number[];
  generationRateData: [number, number][];
  electron_density_data: {
    positions: number[];
    values: number[];
  };
  hole_density_data: {
    positions: number[];
    values: number[];
  };
  generationRateProfile: {
    positions: number[];
    values: number[];
  };
  currentGenerationData: {
    emitter: {
      generated_carriers: number;
      surviving_carriers: number;
      current: number;
    };
    depletion: {
      generated_carriers: number;
      surviving_carriers: number;
      current: number;
    };
    base: {
      generated_carriers: number;
      surviving_carriers: number;
      current: number;
    };
    jdr: {
      current: number;
      electron_current: number;
      hole_current: number;
      "Electron density": number;
      "Hole density": number;
      "Electron-lifetime": number;
      "Electron-Lifetime-Components": number[];
      "Hole-lifetime": number;
      "Hole-Lifetime-Components": number[];
    };
    solar_cell_parameters: {
      total_current: number;
      reverse_saturation_current: number;
      voc: number;
      rad_per_hour: number;
    };
  };
  surfaceRecombinationVelocities: {
    bareEmitter: number;
    substrateBase: number;
  };
  depletionWidth: number;
  sizeWarning?: string;
  builtInPotential: number;
  totalElectricField: number;
  photonFlux: number;
  photonEnergy: number;
  emitterSize: number;
  baseSize: number;
  totalDeviceWidth: number;
  ganDensity: number;
  massAttenuation: number;
  linearAttenuation: number;
  mobilityMaxElectrons: number;
  mobilityMaxHoles: number;
  mobilityMinElectrons: number;
  mobilityMinHoles: number;
  intrinsicCarrierConcentration: number;
  dielectricConstant: number;
  radiativeRecombinationCoefficient: number;
  augerCoefficient: number;
  electronThermalVelocity: number;
  holeThermalVelocity: number;
  pSideWidth: number;
  intrinsicWidth: number;
  nSideWidth: number;
  eMinElectron: number;
  eMinHole: number;
  eMin: number;
  temperature: number;
  electronConcentration: number;
  holeConcentration: number;
  message: string;
}

interface SimulationResponse {
  builtInPotential: number;
  depletionWidth: number;
  electricField: Array<[number, number]>;
  position: Array<number>;
  temperature: number;
  electronConcentration: number;
  holeConcentration: number;
  message: string;
  pSideWidth: number;
  intrinsicWidth: number;
  nSideWidth: number;
  eMinElectron: number;
  eMinHole: number;
  eMin: number;
  totalElectricField: number;
  photonFlux: number;
  photonEnergy: number;
  emitterSize: number;
  baseSize: number;
  totalDeviceWidth: number;
  sizeWarning?: string;
  injectionWarning?: string;
  ganDensity: number;
  massAttenuation: number;
  linearAttenuation: number;
  mobilityMaxElectrons: number;
  mobilityMaxHoles: number;
  mobilityMinElectrons: number;
  mobilityMinHoles: number;
  intrinsicCarrierConcentration: number;
  dielectricConstant: number;
  radiativeRecombinationCoefficient: number;
  augerCoefficient: number;
  electronThermalVelocity: number;
  holeThermalVelocity: number;
  surfaceRecombinationVelocities: {
    bareEmitter: number;
    substrateBase: number;
  };
  generationRateData: Array<[number, number]>;
  electron_density_data: {
    positions: number[];
    values: number[];
  };
  hole_density_data: {
    positions: number[];
    values: number[];
  };
  generationRateProfile: {
    positions: number[];
    values: number[];
  };
  currentGenerationData: {
    emitter: {
      generated_carriers: number;
      surviving_carriers: number;
      current: number;
    };
    depletion: {
      generated_carriers: number;
      surviving_carriers: number;
      current: number;
    };
    base: {
      generated_carriers: number;
      surviving_carriers: number;
      current: number;
    };
    jdr: {
      current: number;
      electron_current: number;
      hole_current: number;
      "Electron density": number;
      "Hole density": number;
      "Electron-lifetime": number;
      "Electron-Lifetime-Components": number[];
      "Hole-lifetime": number;
      "Hole-Lifetime-Components": number[];
    };
    solar_cell_parameters: {
      total_current: number;
      reverse_saturation_current: number;
      voc: number;
      rad_per_hour: number;
    };
  };
  minorityRecombinationRates: Array<{
    region: string;
    auger: number;
    srh: number;
    radiative: number;
    bulk: number;
  }>;
}

interface SimulationResultsProps {
  simulationResponse?: SimulationResponse;
}

interface DefectShapeProps {
  x: number;
  y: number;
  width: number;
  height: number;
  fill: string;
  payload: {
    type: "defect" | "band";
  };
}

const SimulationResults: React.FC<SimulationResultsProps> = ({
  simulationResponse,
}) => {
  const [lastValidResponse, setLastValidResponse] =
    React.useState<SimulationResponse | null>(null);
  const [lastValidData, setLastValidData] = React.useState<SimulationData[]>(
    []
  );

  // Update lastValidResponse when simulationResponse changes
  React.useEffect(() => {
    console.log("SimulationResponse changed:", simulationResponse);
    if (simulationResponse?.builtInPotential !== undefined) {
      console.log("Setting lastValidResponse to:", simulationResponse);
      setLastValidResponse(simulationResponse);
    }
  }, [simulationResponse]);

  // Log the current values for debugging
  React.useEffect(() => {
    console.log("Current simulationResponse:", simulationResponse);
    console.log("Current lastValidResponse:", lastValidResponse);
    console.log("eMin values:", {
      electron:
        simulationResponse?.eMinElectron || lastValidResponse?.eMinElectron,
      hole: simulationResponse?.eMinHole || lastValidResponse?.eMinHole,
      max: simulationResponse?.eMin || lastValidResponse?.eMin,
    });
  }, [simulationResponse, lastValidResponse]);

  const { data, isLoading, error } = useQuery<SimulationResponse>({
    queryKey: ["simulationResults"],
    queryFn: async () => {
      console.log("Fetching simulation data from /api/simulate");
      const response = await axios.get<SimulationResponse>(
        "http://localhost:8000/api/simulate"
      );
      const newData = response.data;
      console.log("Received simulation data:", newData);
      if (newData) {
        setLastValidResponse(newData);
        console.log("Updated lastValidResponse with:", newData);
      }
      return newData;
    },
    retry: 1,
    staleTime: 0,
    refetchOnWindowFocus: false,
    placeholderData: lastValidResponse || undefined,
    enabled: !!simulationResponse,
  });

  // Transform data for the chart
  const transformedData = React.useMemo(() => {
    const currentData = (simulationResponse ||
      lastValidResponse) as SimulationData;
    console.log("Current simulation response:", currentData);
    console.log("Electric field data:", currentData?.electricField);
    console.log("Position data:", currentData?.position);

    if (!currentData?.electricField || !currentData?.position) {
      console.log("No valid data available");
      return [];
    }

    try {
      return currentData.position.map((position, index) => {
        const electricFieldValue = currentData.electricField[index];
        return {
          position: parseFloat(position.toString()) * 10000, // Convert to micrometers
          electricField: parseFloat(electricFieldValue[1].toString()) * 100, // Convert from V/cm to V/m
        };
      });
    } catch (error) {
      console.log("Error transforming data:", error);
      return [];
    }
  }, [simulationResponse, lastValidResponse]) as ElectricFieldData[];

  // Transform generation rate data for the chart
  const transformedGenerationData = React.useMemo(() => {
    const currentData = simulationResponse || lastValidResponse;
    console.log("Generation rate data:", currentData?.generationRateData);

    if (!currentData?.generationRateData) {
      console.log("No generation rate data available");
      return [];
    }

    try {
      return currentData.generationRateData.map(([position, genRate]) => ({
        position: parseFloat(position.toString()) * 10000, // Convert from cm to micrometers
        generationRate: parseFloat(genRate.toString()) * 2, // Multiply by 2 as requested
      }));
    } catch (error) {
      console.log("Error transforming generation rate data:", error);
      return [];
    }
  }, [simulationResponse, lastValidResponse]);

  // Transform density data for the charts
  const transformedElectronDensityData = React.useMemo(() => {
    const currentData = simulationResponse || lastValidResponse;
    console.log("Electron density data:", currentData?.electron_density_data);

    if (
      !currentData?.electron_density_data?.values ||
      !currentData?.electron_density_data?.positions ||
      !currentData?.depletionWidth
    ) {
      console.log("No electron density data available");
      return [];
    }

    try {
      const totalWidth = currentData.depletionWidth * 10000; // Convert to μm
      const numPoints = currentData.electron_density_data.values.length;
      const dx = totalWidth / (numPoints - 1);

      return currentData.electron_density_data.values.map((density, index) => ({
        position: Number((index * dx).toFixed(2)), // Position from 0 to total width
        density: parseFloat(density.toString()),
      }));
    } catch (error) {
      console.log("Error transforming electron density data:", error);
      return [];
    }
  }, [simulationResponse, lastValidResponse]);

  const transformedHoleDensityData = React.useMemo(() => {
    const currentData = simulationResponse || lastValidResponse;
    console.log("Hole density data:", currentData?.hole_density_data);

    if (
      !currentData?.hole_density_data?.values ||
      !currentData?.hole_density_data?.positions ||
      !currentData?.depletionWidth
    ) {
      console.log("No hole density data available");
      return [];
    }

    try {
      const totalWidth = currentData.depletionWidth * 10000; // Convert to μm
      const numPoints = currentData.hole_density_data.values.length;
      const dx = totalWidth / (numPoints - 1);

      return currentData.hole_density_data.values.map((density, index) => ({
        position: Number((index * dx).toFixed(2)), // Position from 0 to total width
        density: parseFloat(density.toString()),
      }));
    } catch (error) {
      console.log("Error transforming hole density data:", error);
      return [];
    }
  }, [simulationResponse, lastValidResponse]);

  // Transform current generation data for the chart
  const transformedCurrentGenerationData = React.useMemo(() => {
    const currentData = simulationResponse || lastValidResponse;
    if (!currentData?.currentGenerationData) {
      return [];
    }

    try {
      return [
        {
          region: "Emitter",
          value: parseFloat(
            (
              currentData.currentGenerationData.emitter.generated_carriers * 2
            ).toString()
          ),
        },
        {
          region: "JDR",
          value: parseFloat(
            currentData.currentGenerationData.jdr.current.toString()
          ),
        },
      ];
    } catch (error) {
      console.log("Error transforming current generation data:", error);
      return [];
    }
  }, [simulationResponse, lastValidResponse]);

  // Add effect to log when responses change
  React.useEffect(() => {
    console.log("Simulation response changed:", simulationResponse);
    console.log("Last valid response:", lastValidResponse);
  }, [simulationResponse, lastValidResponse]);

  // Early return for loading state
  if (isLoading && !data) {
    return (
      <Center p={8}>
        <Spinner size="xl" />
        <Text ml={4}>Loading simulation results...</Text>
      </Center>
    );
  }

  // Early return for error state
  if (error) {
    console.error("Error fetching results:", error);
    return (
      <Center p={8}>
        <Text color="red.500">
          Error loading simulation results. Please try again.
        </Text>
      </Center>
    );
  }

  // Early return for no data state
  if (!data) {
    return (
      <Center p={8}>
        <Text color="gray.500">
          No simulation data available. Start a simulation to see results.
        </Text>
      </Center>
    );
  }

  return (
    <VStack spacing={8} align="stretch" w="100%" py={4}>
      {/* Built-in Potential and Depletion Width Display */}
      <Box bg="white" p={6} borderRadius="lg" boxShadow="md">
        <Heading size="md" mb={4} color="gray.800">
          Simulation Results
        </Heading>
        <Grid templateColumns="repeat(3, 1fr)" gap={4}>
          {/* First Row */}
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              Depletion Width
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                (simulationResponse?.depletionWidth ||
                  lastValidResponse?.depletionWidth ||
                  0) * 10000
              ).toFixed(6)}{" "}
              μm
            </Text>
          </Box>
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              Built-in Potential
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                simulationResponse?.builtInPotential ||
                lastValidResponse?.builtInPotential
              )?.toFixed(6) || "N/A"}{" "}
              V
            </Text>
          </Box>
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              Total Electric Field
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                simulationResponse?.totalElectricField ||
                lastValidResponse?.totalElectricField
              )?.toExponential(2) || "N/A"}{" "}
              V/m
            </Text>
          </Box>
          {/* Second Row */}
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              P-Side Width
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                (simulationResponse?.pSideWidth ||
                  lastValidResponse?.pSideWidth ||
                  0) * 10000
              ).toFixed(6)}{" "}
              μm
            </Text>
          </Box>
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              Intrinsic Width
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                (simulationResponse?.intrinsicWidth ||
                  lastValidResponse?.intrinsicWidth ||
                  0) * 10000
              ).toFixed(6)}{" "}
              μm
            </Text>
          </Box>
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              N-Side Width
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                (simulationResponse?.nSideWidth ||
                  lastValidResponse?.nSideWidth ||
                  0) * 10000
              ).toFixed(6)}{" "}
              μm
            </Text>
          </Box>
          {/* Third Row */}
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              Emin Electron
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                (simulationResponse?.eMinElectron ||
                  lastValidResponse?.eMinElectron ||
                  0) * 100
              ).toExponential(2)}{" "}
              V/m
            </Text>
          </Box>
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              Emin Hole
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                (simulationResponse?.eMinHole ||
                  lastValidResponse?.eMinHole ||
                  0) * 100
              ).toExponential(2)}{" "}
              V/m
            </Text>
          </Box>
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              Emin
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                (simulationResponse?.eMin || lastValidResponse?.eMin || 0) * 100
              ).toExponential(2)}{" "}
              V/m
            </Text>
          </Box>
        </Grid>
      </Box>

      {/* Electric Field Graph */}
      <Box bg="white" p={6} borderRadius="lg" boxShadow="md">
        <Heading size="md" mb={4} color="gray.800">
          Electric Field vs. Position
        </Heading>
        <Box h="500px" w="100%">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={transformedData}
              margin={{ top: 60, right: 80, left: 80, bottom: 60 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                dataKey="position"
                label={{
                  value: "Position (μm)",
                  position: "bottom",
                  offset: 30,
                  style: {
                    fontSize: "14px",
                    fill: "#4b5563",
                    fontWeight: "500",
                  },
                }}
                tick={{ fontSize: 12, dy: 10 }}
                tickFormatter={(value) => value.toFixed(2)}
                tickCount={6}
              />
              <YAxis
                label={{
                  value: "Electric Field (V/m)",
                  angle: -90,
                  position: "outside",
                  offset: 20,
                  style: {
                    fontSize: "14px",
                    fill: "#4b5563",
                    fontWeight: "500",
                  },
                }}
                tick={{ fontSize: 12, dx: -10 }}
                tickFormatter={(value) => value.toExponential(2)}
                tickCount={6}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "white",
                  border: "1px solid #e5e7eb",
                  borderRadius: "0.375rem",
                  boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
                  padding: "12px",
                }}
                formatter={(value: number) => [
                  `${value.toExponential(2)} V/m`,
                  "Electric Field",
                ]}
                labelFormatter={(label) => `Position: ${label.toFixed(2)} μm`}
                cursor={{ stroke: "#e5e7eb", strokeWidth: 1 }}
              />
              <Line
                type="monotone"
                dataKey="electricField"
                stroke="#3b82f6"
                strokeWidth={2.5}
                dot={false}
                activeDot={{
                  r: 4,
                  fill: "#3b82f6",
                  stroke: "#3b82f6",
                  strokeWidth: 2,
                }}
              />
            </LineChart>
          </ResponsiveContainer>
        </Box>
      </Box>

      {/* Radiation and Generation Parameters */}
      <Box bg="white" p={6} borderRadius="lg" boxShadow="md">
        <Heading size="md" mb={4} color="gray.800">
          Radiation and Generation Parameters
        </Heading>
        <Grid templateColumns="repeat(3, 1fr)" gap={4}>
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              Minimum Energy per EHP
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(1.58956743543344983e-18).toExponential(3)} J
            </Text>
          </Box>
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              Photon Flux
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                simulationResponse?.photonFlux || lastValidResponse?.photonFlux
              )?.toExponential(2) || "N/A"}{" "}
              photons/cm²·s
            </Text>
          </Box>
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              Photon Energy
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                simulationResponse?.photonEnergy ||
                lastValidResponse?.photonEnergy
              )?.toExponential(2) || "N/A"}{" "}
              J
            </Text>
          </Box>
        </Grid>
      </Box>

      {/* Device Parameters */}
      <Box bg="white" p={6} borderRadius="lg" boxShadow="md">
        <Heading size="md" mb={4} color="gray.800">
          Device Parameters
        </Heading>
        {simulationResponse?.sizeWarning && (
          <Alert status="warning" mb={4} borderRadius="md">
            <AlertIcon />
            <AlertTitle>Device Size Warning</AlertTitle>
            <AlertDescription>
              {simulationResponse.sizeWarning}
            </AlertDescription>
          </Alert>
        )}
        <Grid templateColumns="repeat(5, 1fr)" gap={4}>
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              Emitter Size (N doped)
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                simulationResponse?.emitterSize ||
                lastValidResponse?.emitterSize ||
                0
              ).toFixed(2)}{" "}
              μm
            </Text>
          </Box>
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              Intrinsic Region Width
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                (simulationResponse?.intrinsicWidth ||
                  lastValidResponse?.intrinsicWidth ||
                  0) * 10000
              ).toFixed(2)}{" "}
              μm
            </Text>
          </Box>
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              Base Size (P doped)
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                simulationResponse?.baseSize ||
                lastValidResponse?.baseSize ||
                0
              ).toFixed(2)}{" "}
              μm
            </Text>
          </Box>
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              Total Depletion Width
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                (simulationResponse?.depletionWidth ||
                  lastValidResponse?.depletionWidth ||
                  0) * 10000
              ).toFixed(2)}{" "}
              μm
            </Text>
          </Box>
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              Total Device Width
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                (simulationResponse?.emitterSize ||
                  lastValidResponse?.emitterSize ||
                  0) +
                (simulationResponse?.baseSize ||
                  lastValidResponse?.baseSize ||
                  0) +
                (simulationResponse?.depletionWidth ||
                  lastValidResponse?.depletionWidth ||
                  0) *
                  10000
              ).toFixed(2)}{" "}
              μm
            </Text>
          </Box>
        </Grid>
      </Box>

      {/* GaN Material Properties */}
      <Box bg="white" p={6} borderRadius="lg" boxShadow="md">
        <Heading size="md" mb={4} color="gray.800">
          Gallium Nitride Material Properties
        </Heading>
        <Grid templateColumns="repeat(3, 1fr)" gap={4}>
          {/* First Row */}
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              GaN Density
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                simulationResponse?.ganDensity ||
                lastValidResponse?.ganDensity ||
                6.15
              ).toFixed(2)}{" "}
              g/cm³
            </Text>
          </Box>
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              Mass Attenuation
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                simulationResponse?.massAttenuation ||
                lastValidResponse?.massAttenuation
              )?.toExponential(2) || "N/A"}{" "}
              cm²/g
            </Text>
          </Box>
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              Linear Attenuation
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                simulationResponse?.linearAttenuation ||
                lastValidResponse?.linearAttenuation
              )?.toExponential(2) || "N/A"}{" "}
              cm⁻¹
            </Text>
          </Box>
          {/* Second Row */}
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              Mobility Max Electrons
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                simulationResponse?.mobilityMaxElectrons ||
                lastValidResponse?.mobilityMaxElectrons ||
                1000.0
              ).toFixed(0)}{" "}
              cm²/V·s
            </Text>
          </Box>
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              Mobility Max Holes
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                simulationResponse?.mobilityMaxHoles ||
                lastValidResponse?.mobilityMaxHoles ||
                40.0
              ).toFixed(0)}{" "}
              cm²/V·s
            </Text>
          </Box>
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              Mobility Min Electrons
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                simulationResponse?.mobilityMinElectrons ||
                lastValidResponse?.mobilityMinElectrons ||
                55.0
              ).toFixed(0)}{" "}
              cm²/V·s
            </Text>
          </Box>
          {/* Third Row */}
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              Mobility Min Holes
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                simulationResponse?.mobilityMinHoles ||
                lastValidResponse?.mobilityMinHoles ||
                3.0
              ).toFixed(0)}{" "}
              cm²/V·s
            </Text>
          </Box>
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              Intrinsic Carrier Concentration
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                simulationResponse?.intrinsicCarrierConcentration ||
                lastValidResponse?.intrinsicCarrierConcentration ||
                1.6e-10
              ).toExponential(2)}{" "}
              cm⁻³
            </Text>
          </Box>
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              Dielectric Constant
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                simulationResponse?.dielectricConstant ||
                lastValidResponse?.dielectricConstant ||
                8.9
              ).toFixed(1)}
            </Text>
          </Box>
          {/* Fourth Row */}
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              Radiative Recombination Coefficient
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                simulationResponse?.radiativeRecombinationCoefficient ||
                lastValidResponse?.radiativeRecombinationCoefficient ||
                1.1e-8
              ).toExponential(2)}{" "}
              cm²/s
            </Text>
          </Box>
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              Auger Coefficient
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                simulationResponse?.augerCoefficient ||
                lastValidResponse?.augerCoefficient ||
                1e-30
              ).toExponential(2)}{" "}
              cm⁶/s
            </Text>
          </Box>
          {/* Fifth Row */}
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              Electron Thermal Velocity
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                simulationResponse?.electronThermalVelocity ||
                lastValidResponse?.electronThermalVelocity ||
                2.43e7
              ).toExponential(2)}{" "}
              cm/s
            </Text>
          </Box>
          <Box>
            <Text fontSize="sm" color="gray.600" mb={2}>
              Hole Thermal Velocity
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="blue.500">
              {(
                simulationResponse?.holeThermalVelocity ||
                lastValidResponse?.holeThermalVelocity ||
                2.38e7
              ).toExponential(2)}{" "}
              cm/s
            </Text>
          </Box>
        </Grid>
      </Box>

      {/* GaN Defect Levels */}
      <Box bg="white" p={6} borderRadius="lg" boxShadow="md">
        <Heading size="md" mb={4} color="gray.800">
          GaN Defect Energy Levels
        </Heading>
        <Box h="400px" w="100%">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              layout="vertical"
              data={[
                {
                  name: "Conduction Band",
                  energy: 3.4, // GaN bandgap
                  type: "band",
                  fill: "#3b82f6",
                  crossSection: null,
                  density: null,
                },
                // Electron Traps
                {
                  name: "Electron Trap E1",
                  energy: 0.26,
                  type: "defect",
                  fill: "#ef4444",
                  crossSection: 8e-15,
                  density: 1.6e13,
                },
                {
                  name: "Electron Trap E2",
                  energy: 0.58,
                  type: "defect",
                  fill: "#ef4444",
                  crossSection: 1e-14,
                  density: 3e13,
                },
                {
                  name: "Electron Trap E3",
                  energy: 0.66,
                  type: "defect",
                  fill: "#ef4444",
                  crossSection: 1.65e-17,
                  density: 1.19e15,
                },
                // Hole Traps
                {
                  name: "Hole Trap H1",
                  energy: 0.86,
                  type: "defect",
                  fill: "#10b981",
                  crossSection: 5e-15,
                  density: 5e14,
                },
                {
                  name: "Hole Trap H2",
                  energy: 0.9,
                  type: "defect",
                  fill: "#10b981",
                  crossSection: 3e-14,
                  density: 2e15,
                },
                {
                  name: "Hole Trap H3",
                  energy: 1.1,
                  type: "defect",
                  fill: "#10b981",
                  crossSection: 2e-13,
                  density: 5e14,
                },
                {
                  name: "Hole Trap H4",
                  energy: 1.3,
                  type: "defect",
                  fill: "#10b981",
                  crossSection: 3e-14,
                  density: 1e14,
                },
                {
                  name: "Valence Band",
                  energy: 0,
                  type: "band",
                  fill: "#3b82f6",
                  crossSection: null,
                  density: null,
                },
              ]}
              margin={{ top: 20, right: 30, left: 150, bottom: 40 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                type="number"
                domain={[0, 3.5]}
                label={{
                  value: "Energy Level (eV)",
                  position: "bottom",
                  offset: 20,
                  style: {
                    fontSize: "14px",
                    fill: "#4b5563",
                    fontWeight: "500",
                  },
                }}
              />
              <YAxis
                dataKey="name"
                type="category"
                label={{
                  value: "Defect Type",
                  angle: -90,
                  position: "insideLeft",
                  offset: -140,
                  style: {
                    fontSize: "14px",
                    fill: "#4b5563",
                    fontWeight: "500",
                  },
                }}
              />
              <Tooltip
                formatter={(value: any, name: string, props: any) => {
                  const { payload } = props;
                  if (payload.type === "band") {
                    return [`${value.toFixed(2)} eV`, "Energy Level"];
                  }
                  return [
                    <div key={name}>
                      <div>Energy: {value.toFixed(2)} eV</div>
                      <div>
                        Cross-section: {payload.crossSection.toExponential(2)}{" "}
                        cm²
                      </div>
                      <div>
                        Density: {payload.density.toExponential(2)} cm⁻³
                      </div>
                    </div>,
                    payload.name,
                  ];
                }}
                cursor={{ fill: "transparent" }}
              />
              <Bar
                dataKey="energy"
                barSize={30}
                shape={(props: any) => {
                  const { x, y, width, height, fill, payload } = props;
                  const isDefect = payload.type === "defect";

                  return (
                    <rect
                      x={x}
                      y={isDefect ? y + height / 4 : y}
                      width={width}
                      height={isDefect ? height / 2 : height}
                      fill={fill}
                      rx={isDefect ? 2 : 0}
                      ry={isDefect ? 2 : 0}
                    />
                  );
                }}
              >
                <LabelList
                  dataKey="energy"
                  position="right"
                  formatter={(value: number) => `${value.toFixed(2)} eV`}
                  style={{ fontSize: "12px" }}
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Box>
        <Text fontSize="sm" color="gray.600" mt={2} textAlign="center">
          Energy levels measured from valence band edge. Electron traps (red)
          and hole traps (green) show energy level, cross-section area, and
          density. Band edges shown in blue.
        </Text>
      </Box>

      {/* Gallium Nitride Defects */}
      <Box bg="white" p={6} borderRadius="lg" boxShadow="md">
        <Heading size="md" mb={4} color="gray.800">
          Carrier Density Profiles
        </Heading>
        <VStack spacing={6} align="stretch">
          {/* Electron Density Profile */}
          <Box>
            <Heading size="sm" mb={4} color="gray.700">
              Electron Density Profile
            </Heading>
            <Box h="300px" w="100%">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={transformedElectronDensityData}
                  margin={{ top: 20, right: 30, left: 100, bottom: 40 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="position"
                    label={{
                      value: "Position (μm)",
                      position: "bottom",
                      offset: 14,
                    }}
                    domain={[
                      0,
                      (simulationResponse?.depletionWidth ||
                        lastValidResponse?.depletionWidth ||
                        0) * 10000,
                    ]}
                    tickFormatter={(value) => value.toFixed(2)}
                  />
                  <YAxis
                    label={{
                      value: "Electron Density (cm⁻³)",
                      angle: -90,
                      position: "insideLeft",
                      offset: -60,
                    }}
                    tickFormatter={(value) => value.toExponential(2)}
                  />
                  <Tooltip
                    formatter={(value: number) => [
                      value.toExponential(2),
                      "Electron Density",
                    ]}
                    labelFormatter={(label) =>
                      `Position: ${Number(label).toFixed(2)} μm`
                    }
                  />
                  <Line type="monotone" dataKey="density" stroke="#3b82f6" />
                </LineChart>
              </ResponsiveContainer>
                    </Box>
          </Box>

          {/* Hole Density Profile */}
                    <Box>
            <Heading size="sm" mb={4} color="gray.700">
              Hole Density Profile
            </Heading>
            <Box h="300px" w="100%">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={transformedHoleDensityData}
                  margin={{ top: 20, right: 30, left: 100, bottom: 40 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="position"
                    label={{
                      value: "Position (μm)",
                      position: "bottom",
                      offset: 14,
                    }}
                    domain={[
                      0,
                      (simulationResponse?.depletionWidth ||
                        lastValidResponse?.depletionWidth ||
                        0) * 10000,
                    ]}
                    tickFormatter={(value) => value.toFixed(2)}
                  />
                  <YAxis
                    label={{
                      value: "Hole Density (cm⁻³)",
                      angle: -90,
                      position: "insideLeft",
                      offset: -60,
                    }}
                    tickFormatter={(value) => value.toExponential(2)}
                  />
                  <Tooltip
                    formatter={(value: number) => [
                      value.toExponential(2),
                      "Hole Density",
                    ]}
                    labelFormatter={(label) =>
                      `Position: ${Number(label).toFixed(2)} μm`
                    }
                  />
                  <Line type="monotone" dataKey="density" stroke="#ef4444" />
                </LineChart>
              </ResponsiveContainer>
            </Box>
                    </Box>
                  </VStack>
      </Box>

      {/* Minority Recombination Rates */}
      <Box bg="white" p={6} borderRadius="lg" boxShadow="md">
        <Heading size="md" mb={4} color="gray.800">
          Minority Recombination Rates
        </Heading>
        {(
          simulationResponse?.minorityRecombinationRates ||
          lastValidResponse?.minorityRecombinationRates ||
          []
        ).map((rateData, index) => (
          <Box key={index} mb={4}>
            <Heading size="sm" mb={3} color="gray.700">
              {rateData.region}
            </Heading>
            <TableContainer>
              <Table variant="simple" size="sm">
                <Thead>
                  <Tr>
                    <Th>Recombination Type</Th>
                    <Th isNumeric>Lifetime (s)</Th>
                  </Tr>
                </Thead>
                <Tbody>
                  <Tr>
                    <Td>Auger</Td>
                    <Td isNumeric>
                      {rateData.auger ? rateData.auger.toExponential(2) : "N/A"}
                    </Td>
                  </Tr>
                  <Tr>
                    <Td>SRH</Td>
                    <Td isNumeric>
                      {rateData.srh ? rateData.srh.toExponential(2) : "N/A"}
                    </Td>
                  </Tr>
                  <Tr>
                    <Td>Radiative</Td>
                    <Td isNumeric>
                      {rateData.radiative
                        ? rateData.radiative.toExponential(2)
                        : "N/A"}
                    </Td>
                  </Tr>
                  <Tr>
                    <Td fontWeight="bold">Total Bulk</Td>
                    <Td isNumeric fontWeight="bold">
                      {rateData.bulk ? rateData.bulk.toExponential(2) : "N/A"}
                    </Td>
                  </Tr>
                </Tbody>
              </Table>
            </TableContainer>
                </Box>
              ))}
          </Box>

      {/* Recombination Rates in Depletion Region */}
      <Box bg="white" p={6} borderRadius="lg" boxShadow="md">
        <Heading size="md" mb={4} color="gray.800">
          Recombination Rates in Depletion Region
        </Heading>

        {/* Electron Lifetimes */}
        <Box mb={4}>
          <Heading size="sm" mb={3} color="gray.700">
            Electron Lifetimes
          </Heading>
          <TableContainer>
            <Table variant="simple" size="sm">
              <Thead>
                <Tr>
                  <Th>Recombination Type</Th>
                  <Th isNumeric>Lifetime (s)</Th>
                </Tr>
              </Thead>
              <Tbody>
                <Tr>
                  <Td>Radiative</Td>
                  <Td isNumeric>
                    {simulationResponse?.currentGenerationData?.jdr?.[
                      "Electron-Lifetime-Components"
                    ]?.[0]?.toExponential(2) || "N/A"}
                  </Td>
                </Tr>
                <Tr>
                  <Td>Auger</Td>
                  <Td isNumeric>
                    {simulationResponse?.currentGenerationData?.jdr?.[
                      "Electron-Lifetime-Components"
                    ]?.[1]?.toExponential(2) || "N/A"}
                  </Td>
                </Tr>
                <Tr>
                  <Td>SRH</Td>
                  <Td isNumeric>
                    {simulationResponse?.currentGenerationData?.jdr?.[
                      "Electron-Lifetime-Components"
                    ]?.[2]?.toExponential(2) || "N/A"}
                  </Td>
                </Tr>
                <Tr>
                  <Td fontWeight="bold">Total</Td>
                  <Td isNumeric fontWeight="bold">
                    {simulationResponse?.currentGenerationData?.jdr?.[
                      "Electron-lifetime"
                    ]?.toExponential(2) || "N/A"}
                  </Td>
                </Tr>
              </Tbody>
            </Table>
          </TableContainer>
        </Box>

        {/* Hole Lifetimes */}
        <Box>
          <Heading size="sm" mb={3} color="gray.700">
            Hole Lifetimes
          </Heading>
          <TableContainer>
            <Table variant="simple" size="sm">
              <Thead>
                <Tr>
                  <Th>Recombination Type</Th>
                  <Th isNumeric>Lifetime (s)</Th>
                </Tr>
              </Thead>
              <Tbody>
                <Tr>
                  <Td>Radiative</Td>
                  <Td isNumeric>
                    {simulationResponse?.currentGenerationData?.jdr?.[
                      "Hole-Lifetime-Components"
                    ]?.[0]?.toExponential(2) || "N/A"}
                  </Td>
                </Tr>
                <Tr>
                  <Td>Auger</Td>
                  <Td isNumeric>
                    {simulationResponse?.currentGenerationData?.jdr?.[
                      "Hole-Lifetime-Components"
                    ]?.[1]?.toExponential(2) || "N/A"}
                  </Td>
                </Tr>
                <Tr>
                  <Td>SRH</Td>
                  <Td isNumeric>
                    {simulationResponse?.currentGenerationData?.jdr?.[
                      "Hole-Lifetime-Components"
                    ]?.[2]?.toExponential(2) || "N/A"}
                  </Td>
                </Tr>
                <Tr>
                  <Td fontWeight="bold">Total</Td>
                  <Td isNumeric fontWeight="bold">
                    {simulationResponse?.currentGenerationData?.jdr?.[
                      "Hole-lifetime"
                    ]?.toExponential(2) || "N/A"}
                  </Td>
                </Tr>
              </Tbody>
            </Table>
          </TableContainer>
        </Box>
      </Box>

      {/* Generation Rate Profile Chart */}
      <Box bg="white" p={6} borderRadius="lg" boxShadow="md">
        <Heading size="md" mb={4} color="gray.800">
          Generation Rate Profile
        </Heading>
        <Box h="400px" w="100%">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={(
                simulationResponse?.generationRateProfile?.positions ||
                lastValidResponse?.generationRateProfile?.positions ||
                []
              ).map((pos: number, idx: number) => ({
                position: pos * 10000, // Convert to micrometers
                generationRate:
                  (simulationResponse?.generationRateProfile?.values ||
                    lastValidResponse?.generationRateProfile?.values ||
                    [])[idx] * 2, // Multiply by 2 as requested
              }))}
              margin={{ top: 20, right: 30, left: 100, bottom: 40 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="position"
                label={{
                  value: "Position (μm)",
                  position: "bottom",
                  offset: 20,
                }}
                tickFormatter={(value) => value.toFixed(2)}
              />
              <YAxis
                label={{
                  value: "Generation Rate (cm⁻³s⁻¹)",
                  angle: -90,
                  position: "insideLeft",
                  offset: -60,
                }}
                tickFormatter={(value) => value.toExponential(2)}
              />
              <Tooltip
                formatter={(value: any) => [
                  value.toExponential(2),
                  "Generation Rate",
                ]}
                labelFormatter={(label) =>
                  `Position: ${Number(label).toFixed(2)} μm`
                }
              />
              <Line
                type="monotone"
                dataKey="generationRate"
                stroke="#8884d8"
                dot={false}
                name="Generation Rate"
              />
            </LineChart>
          </ResponsiveContainer>
        </Box>
      </Box>

      {/* Generation Per Region Chart */}
      <Box bg="white" p={6} borderRadius="lg" boxShadow="md">
        <Heading size="md" mb={4} color="gray.800">
          Generation Rate and Current Density by Region
        </Heading>
        <Box h="400px" w="100%">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={[
                {
                  region: "Emitter",
                  generationRate:
                    (simulationResponse?.currentGenerationData?.emitter
                      ?.generated_carriers ||
                      lastValidResponse?.currentGenerationData?.emitter
                        ?.generated_carriers ||
                      0) * 2,
                  currentDensity:
                    simulationResponse?.currentGenerationData?.emitter
                      ?.current ||
                    lastValidResponse?.currentGenerationData?.emitter
                      ?.current ||
                    0,
                },
                {
                  region: "Depletion",
                  generationRate:
                    (simulationResponse?.currentGenerationData?.depletion
                      ?.generated_carriers ||
                      lastValidResponse?.currentGenerationData?.depletion
                        ?.generated_carriers ||
                      0) * 2,
                  currentDensity:
                    simulationResponse?.currentGenerationData?.jdr?.current ||
                    lastValidResponse?.currentGenerationData?.jdr?.current ||
                    0,
                },
                {
                  region: "Base",
                  generationRate:
                    (simulationResponse?.currentGenerationData?.base
                      ?.generated_carriers ||
                      lastValidResponse?.currentGenerationData?.base
                        ?.generated_carriers ||
                      0) * 2,
                  currentDensity:
                    simulationResponse?.currentGenerationData?.base?.current ||
                    lastValidResponse?.currentGenerationData?.base?.current ||
                    0,
                },
              ]}
              margin={{ top: 20, right: 30, left: 100, bottom: 40 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="region"
                label={{
                  value: "Device Region",
                  position: "bottom",
                  offset: 20,
                }}
              />
              <YAxis
                yAxisId="left"
                label={{
                  value: "Generation Rate (cm⁻³s⁻¹)",
                  angle: -90,
                  position: "insideLeft",
                  offset: -60,
                }}
                tickFormatter={(value) => value.toExponential(2)}
              />
              <YAxis
                yAxisId="right"
                orientation="right"
                label={{
                  value: "Current Density (A/cm²)",
                  angle: 90,
                  position: "insideRight",
                  offset: -40,
                }}
                tickFormatter={(value) => value.toExponential(2)}
              />
              <Tooltip
                formatter={(value: number, name: string) => {
                  const formattedValue = value.toExponential(2);
                  switch (name) {
                    case "generationRate":
                      return [`${formattedValue} cm⁻³s⁻¹`, "Generation Rate"];
                    case "currentDensity":
                      return [`${formattedValue} A/cm²`, "Current Density"];
                    default:
                      return [formattedValue, name];
                  }
                }}
              />
              <Bar
                yAxisId="left"
                dataKey="generationRate"
                fill="#3b82f6"
                name="Generation Rate"
              >
                <LabelList
                  dataKey="generationRate"
                  position="top"
                  formatter={(value: number) => value.toExponential(2)}
                />
              </Bar>
              <Bar
                yAxisId="right"
                dataKey="currentDensity"
                fill="#10b981"
                name="Current Density"
              >
                <LabelList
                  dataKey="currentDensity"
                  position="top"
                  formatter={(value: number) => value.toExponential(2)}
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Box>
      </Box>

      {/* Surface Recombination Velocities */}
      <Box bg="white" p={6} borderRadius="lg" boxShadow="md" mt={6}>
        <Heading size="md" mb={4} color="gray.800">
          Surface Recombination Velocities
        </Heading>
        <TableContainer>
          <Table variant="simple" size="sm">
            <Thead>
              <Tr>
                <Th>Region</Th>
                <Th isNumeric>Velocity (cm/s)</Th>
              </Tr>
            </Thead>
            <Tbody>
              <Tr>
                <Td>Bare Emitter</Td>
                <Td isNumeric>
                  {(
                    simulationResponse?.surfaceRecombinationVelocities
                      ?.bareEmitter ||
                    lastValidResponse?.surfaceRecombinationVelocities
                      ?.bareEmitter ||
                    0
                  ).toExponential(2)}
                </Td>
              </Tr>
              <Tr>
                <Td>Substrate Base</Td>
                <Td isNumeric>
                  {(
                    simulationResponse?.surfaceRecombinationVelocities
                      ?.substrateBase ||
                    lastValidResponse?.surfaceRecombinationVelocities
                      ?.substrateBase ||
                    0
                  ).toExponential(2)}
                </Td>
              </Tr>
            </Tbody>
          </Table>
        </TableContainer>
      </Box>

      {/* Current Generation Results */}
      <Box bg="white" p={6} borderRadius="lg" boxShadow="md" mt={6}>
        <Heading size="md" mb={4} color="gray.800">
          Current Generation Results
        </Heading>
        <SimpleGrid columns={3} spacing={6}>
          {/* Je (Emitter) Results */}
          <Box>
            <Heading size="sm" mb={4} color="gray.700">
              Je (Emitter) Results
            </Heading>
            <TableContainer>
              <Table variant="simple" size="sm">
                <Thead>
                  <Tr>
                    <Th>Parameter</Th>
                    <Th isNumeric>Value</Th>
                  </Tr>
                </Thead>
                <Tbody>
                  <Tr>
                    <Td>Generated Carriers</Td>
                    <Td isNumeric>
                      {(
                        (simulationResponse?.currentGenerationData?.emitter
                          ?.generated_carriers ||
                          lastValidResponse?.currentGenerationData?.emitter
                            ?.generated_carriers ||
                          0) * 2
                      ).toExponential(2)}{" "}
                      cm⁻³s⁻¹
                    </Td>
                  </Tr>
                  <Tr>
                    <Td>Surviving Carriers</Td>
                    <Td isNumeric>
                      {(
                        simulationResponse?.currentGenerationData?.emitter
                          ?.surviving_carriers ||
                        lastValidResponse?.currentGenerationData?.emitter
                          ?.surviving_carriers ||
                        0
                      ).toExponential(2)}{" "}
                      cm⁻³
                    </Td>
                  </Tr>
                  <Tr>
                    <Td fontWeight="bold">Current</Td>
                    <Td isNumeric fontWeight="bold">
                      {(
                        simulationResponse?.currentGenerationData?.emitter
                          ?.current ||
                        lastValidResponse?.currentGenerationData?.emitter
                          ?.current ||
                        0
                      ).toExponential(2)}{" "}
                      A/cm²
                    </Td>
                  </Tr>
                </Tbody>
              </Table>
            </TableContainer>
          </Box>

          {/* Jb (Base) Results */}
                    <Box>
            <Heading size="sm" mb={4} color="gray.700">
              Jb (Base) Results
            </Heading>
            <TableContainer>
              <Table variant="simple" size="sm">
                <Thead>
                  <Tr>
                    <Th>Parameter</Th>
                    <Th isNumeric>Value</Th>
                  </Tr>
                </Thead>
                <Tbody>
                  <Tr>
                    <Td>Generated Carriers</Td>
                    <Td isNumeric>
                      {(
                        (simulationResponse?.currentGenerationData?.base
                          ?.generated_carriers ||
                          lastValidResponse?.currentGenerationData?.base
                            ?.generated_carriers ||
                          0) * 2
                      ).toExponential(2)}{" "}
                      cm⁻³s⁻¹
                    </Td>
                  </Tr>
                  <Tr>
                    <Td>Surviving Carriers</Td>
                    <Td isNumeric>
                      {(
                        simulationResponse?.currentGenerationData?.base
                          ?.surviving_carriers ||
                        lastValidResponse?.currentGenerationData?.base
                          ?.surviving_carriers ||
                        0
                      ).toExponential(2)}{" "}
                      cm⁻³
                    </Td>
                  </Tr>
                  <Tr>
                    <Td fontWeight="bold">Current</Td>
                    <Td isNumeric fontWeight="bold">
                      {(
                        simulationResponse?.currentGenerationData?.base
                          ?.current ||
                        lastValidResponse?.currentGenerationData?.base
                          ?.current ||
                        0
                      ).toExponential(2)}{" "}
                      A/cm²
                    </Td>
                  </Tr>
                </Tbody>
              </Table>
            </TableContainer>
                    </Box>

          {/* JDR Results */}
                    <Box>
            <Heading size="sm" mb={4} color="gray.700">
              JDR Results
            </Heading>
            <TableContainer>
              <Table variant="simple" size="sm">
                <Thead>
                  <Tr>
                    <Th>Parameter</Th>
                    <Th isNumeric>Value</Th>
                  </Tr>
                </Thead>
                <Tbody>
                  <Tr>
                    <Td>Generated Carriers</Td>
                    <Td isNumeric>
                      {(
                        (simulationResponse?.currentGenerationData?.depletion
                          ?.generated_carriers ||
                          lastValidResponse?.currentGenerationData?.depletion
                            ?.generated_carriers ||
                          0) * 2
                      ).toExponential(2)}{" "}
                      cm⁻³s⁻¹
                    </Td>
                  </Tr>
                  <Tr>
                    <Td>Surviving Carriers</Td>
                    <Td isNumeric>
                      {(
                        (simulationResponse?.currentGenerationData?.jdr
                          ?.current ||
                          lastValidResponse?.currentGenerationData?.jdr
                            ?.current ||
                          0) / 1.6e-19
                      ).toExponential(2)}{" "}
                      cm⁻³
                    </Td>
                  </Tr>
                  <Tr>
                    <Td>Electron Current</Td>
                    <Td isNumeric>
                      {(
                        simulationResponse?.currentGenerationData?.jdr
                          ?.electron_current ||
                        lastValidResponse?.currentGenerationData?.jdr
                          ?.electron_current ||
                        0
                      ).toExponential(2)}{" "}
                      A/cm²
                    </Td>
                  </Tr>
                  <Tr>
                    <Td>Hole Current</Td>
                    <Td isNumeric>
                      {(
                        simulationResponse?.currentGenerationData?.jdr
                          ?.hole_current ||
                        lastValidResponse?.currentGenerationData?.jdr
                          ?.hole_current ||
                        0
                      ).toExponential(2)}{" "}
                      A/cm²
                    </Td>
                  </Tr>
                  <Tr>
                    <Td fontWeight="bold">Total Current</Td>
                    <Td isNumeric fontWeight="bold">
                      {(
                        simulationResponse?.currentGenerationData?.jdr
                          ?.current ||
                        lastValidResponse?.currentGenerationData?.jdr
                          ?.current ||
                        0
                      ).toExponential(2)}{" "}
                      A/cm²
                    </Td>
                  </Tr>
                </Tbody>
              </Table>
            </TableContainer>
                    </Box>
        </SimpleGrid>
                </Box>

      {/* Solar Cell Parameters */}
      <Box bg="white" p={6} borderRadius="lg" boxShadow="md" mt={6}>
        <Heading size="md" mb={4} color="gray.800">
          Gammavoltaic Cell Parameters
        </Heading>
        <TableContainer>
          <Table variant="simple" size="md">
            <Thead>
              <Tr>
                <Th>Parameter</Th>
                <Th isNumeric>Value</Th>
              </Tr>
            </Thead>
            <Tbody>
              <Tr>
                <Td fontWeight="bold">
                  Total Current (J<sub>sc</sub>)
                </Td>
                <Td isNumeric>
                  {(
                    simulationResponse?.currentGenerationData
                      ?.solar_cell_parameters?.total_current ||
                    lastValidResponse?.currentGenerationData
                      ?.solar_cell_parameters?.total_current ||
                    0
                  ).toExponential(2)}{" "}
                  A/cm²
                </Td>
              </Tr>
              <Tr>
                <Td fontWeight="bold">
                  Reverse Saturation Current (J<sub>0</sub>)
                </Td>
                <Td isNumeric>
                  {(
                    simulationResponse?.currentGenerationData
                      ?.solar_cell_parameters?.reverse_saturation_current ||
                    lastValidResponse?.currentGenerationData
                      ?.solar_cell_parameters?.reverse_saturation_current ||
                    0
                  ).toExponential(2)}{" "}
                  A/cm²
                </Td>
              </Tr>
              <Tr>
                <Td fontWeight="bold">
                  Open Circuit Voltage (V<sub>oc</sub>)
                </Td>
                <Td isNumeric>
                  {(
                    simulationResponse?.currentGenerationData
                      ?.solar_cell_parameters?.voc ||
                    lastValidResponse?.currentGenerationData
                      ?.solar_cell_parameters?.voc ||
                    0
                  ).toExponential(2)}{" "}
                  V
                </Td>
              </Tr>
            </Tbody>
          </Table>
        </TableContainer>
          </Box>

      {/* Radiation Exposure and Device Lifetime */}
      <Box bg="white" p={6} borderRadius="lg" boxShadow="md" mt={6}>
        <Heading size="md" mb={4} color="gray.800">
          Radiation Exposure and Device Lifetime
        </Heading>
        <TableContainer>
          <Table variant="simple" size="md">
            <Thead>
              <Tr>
                <Th>Parameter</Th>
                <Th isNumeric>Value</Th>
              </Tr>
            </Thead>
            <Tbody>
              <Tr>
                <Td fontWeight="bold">Radiation Exposure</Td>
                <Td isNumeric>
                  {(
                    simulationResponse?.currentGenerationData
                      ?.solar_cell_parameters?.rad_per_hour ||
                    lastValidResponse?.currentGenerationData
                      ?.solar_cell_parameters?.rad_per_hour ||
                    0
                  ).toExponential(2)}{" "}
                  rad/hour
                </Td>
              </Tr>
              <Tr>
                <Td fontWeight="bold">Device Lifetime</Td>
                <Td isNumeric>
                  {(() => {
                    const radPerHour =
                      simulationResponse?.currentGenerationData
                        ?.solar_cell_parameters?.rad_per_hour ||
                      lastValidResponse?.currentGenerationData
                        ?.solar_cell_parameters?.rad_per_hour ||
                      0;

                    const lifetime =
                      radPerHour > 0 ? 600000000 / radPerHour : 0;
                    // Convert to appropriate time unit (hours)
                    return `${lifetime.toExponential(2)} hours`;
                  })()}
                </Td>
              </Tr>
            </Tbody>
          </Table>
        </TableContainer>
      </Box>
    </VStack>
  );
};

export default SimulationResults;
