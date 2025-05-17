import React, { useState, useEffect, useCallback } from "react";
import {
  Box,
  Button,
  FormControl,
  FormLabel,
  VStack,
  useToast,
  FormErrorMessage,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Radio,
  RadioGroup,
  Stack,
  Input,
  Switch,
  Text,
  Flex,
  HStack,
  Checkbox,
  Heading,
  SimpleGrid,
  Select,
  ButtonGroup,
} from "@chakra-ui/react";
import { useForm, Controller } from "react-hook-form";
import { useMutation } from "@tanstack/react-query";
import axios from "axios";
import SimulationResults from "./SimulationResults";
import PlotComponent from "./PlotComponent";

interface ConcentrationRange {
  min: number;
  max: number;
  steps: number;
  useLogScale: boolean;
}

interface PercentRange {
  min: number;
  max: number;
  steps: number;
}

interface ParameterSweepParams {
  temperature: number;
  donorConcentrationRange: ConcentrationRange;
  acceptorConcentrationRange: ConcentrationRange;
  nRegionWidth: number;
  pRegionWidth: number;
  photonEnergy: number;
  photonFlux: number;
  junctionType: "PN" | "PIN";
  intrinsicConcentration?: number;
  percent?: number;
  percentRange?: PercentRange;
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
  minorityRecombinationRates: Array<{
    region: string;
    auger: number;
    srh: number;
    radiative: number;
    bulk: number;
  }>;
  generationRateData: Array<[number, number]>;
  generationPerRegion: {
    emitter: number;
    depletion: number;
    base: number;
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
      [key: string]: any;
    };
    solar_cell_parameters: {
      total_current: number;
      reverse_saturation_current: number;
      voc: number;
      rad_per_hour: number;
    };
  };
  electron_density_data: {
    positions: number[];
    values: number[];
  };
  hole_density_data: {
    positions: number[];
    values: number[];
  };
  [key: string]: any; // Allow other properties
}

interface SweepResult {
  donorConcentration: number;
  acceptorConcentration: number;
  totalCurrent: number;
  electronLifetime: number;
  holeLifetime: number;
  deviceLifetime: number;
  voc: number;
  radPerHour: number;
  percent?: number;
}

const ParameterSweepForm: React.FC = () => {
  const toast = useToast();
  const [sweepResults, setSweepResults] = useState<SweepResult[]>([]);
  const [detailedResults, setDetailedResults] = useState<any[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [lastSimulationResponse, setLastSimulationResponse] =
    useState<SimulationResponse | null>(null);
  const [simulationCount, setSimulationCount] = useState<number>(0);
  const [isCounterVisible, setIsCounterVisible] = useState<boolean>(false);
  const [currentPlotData, setCurrentPlotData] = useState<any>(null);
  const [pageSize, setPageSize] = useState<number>(20);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [isSavingData, setIsSavingData] = useState<boolean>(false);

  const {
    register,
    handleSubmit,
    control,
    watch,
    formState: { errors },
    getValues,
  } = useForm<ParameterSweepParams>({
    defaultValues: {
      temperature: 300,
      donorConcentrationRange: {
        min: 1e15,
        max: 1e17,
        steps: 5,
        useLogScale: true,
      },
      acceptorConcentrationRange: {
        min: 1e15,
        max: 1e17,
        steps: 5,
        useLogScale: true,
      },
      percentRange: {
        min: 50,
        max: 100,
        steps: 3,
      },
      nRegionWidth: 1,
      pRegionWidth: 1,
      photonEnergy: 149796.0,
      photonFlux: 1e13,
      junctionType: "PN",
      intrinsicConcentration: 1e14,
    },
  });

  const junctionType = watch("junctionType");
  const [saveToCSV, setSaveToCSV] = useState(false);

  // Calculate total simulations that will be run
  const calculateTotalSimulations = () => {
    try {
      // For log scale, count the number of powers of 10 in the range
      const donorMin = parseFloat(
        getValues("donorConcentrationRange.min").toString()
      );
      const donorMax = parseFloat(
        getValues("donorConcentrationRange.max").toString()
      );
      const donorLogScale = getValues("donorConcentrationRange.useLogScale");

      const acceptorMin = parseFloat(
        getValues("acceptorConcentrationRange.min").toString()
      );
      const acceptorMax = parseFloat(
        getValues("acceptorConcentrationRange.max").toString()
      );
      const acceptorLogScale = getValues(
        "acceptorConcentrationRange.useLogScale"
      );

      let donorCount;
      if (donorLogScale) {
        const logMin = Math.floor(Math.log10(donorMin));
        const logMax = Math.ceil(Math.log10(donorMax));
        donorCount = logMax - logMin + 1;
      } else {
        donorCount =
          parseInt(getValues("donorConcentrationRange.steps").toString()) || 1;
      }

      let acceptorCount;
      if (acceptorLogScale) {
        const logMin = Math.floor(Math.log10(acceptorMin));
        const logMax = Math.ceil(Math.log10(acceptorMax));
        acceptorCount = logMax - logMin + 1;
      } else {
        acceptorCount =
          parseInt(getValues("acceptorConcentrationRange.steps").toString()) ||
          1;
      }

      const isPin = getValues("junctionType") === "PIN";
      const percentSteps = isPin
        ? parseInt(getValues("percentRange.steps").toString()) || 1
        : 1;

      return donorCount * acceptorCount * percentSteps;
    } catch (e) {
      return "calculating...";
    }
  };

  // Add a state to track and display the total simulations
  const [simulationsToRun, setSimulationsToRun] = useState<number | string>(
    calculateTotalSimulations()
  );

  // Update the count whenever form values change
  useEffect(() => {
    setSimulationsToRun(calculateTotalSimulations());
  }, [
    watch("donorConcentrationRange.steps"),
    watch("acceptorConcentrationRange.steps"),
    watch("percentRange.steps"),
    watch("junctionType"),
  ]);

  useEffect(() => {
    if (!isCounterVisible) return;

    fetchSimulationCount();

    const intervalId = setInterval(() => {
      fetchSimulationCount();
    }, 2000);

    return () => clearInterval(intervalId);
  }, [isCounterVisible]);

  const fetchSimulationCount = async () => {
    try {
      const response = await axios.get(
        "http://localhost:8000/api/simulation-count"
      );
      setSimulationCount(response.data.total_simulations);
    } catch (error) {
      console.error("Error fetching simulation count:", error);
    }
  };

  // Function to save data in chunks to avoid timeout issues with large datasets
  const saveResultsInChunks = async (results: any[], endpoint: string) => {
    setIsSavingData(true);
    const chunkSize = 50; // Save 50 results at a time
    const totalChunks = Math.ceil(results.length / chunkSize);
    const maxRetries = 3; // Maximum number of retries for failed chunks

    try {
      for (let i = 0; i < totalChunks; i++) {
        const start = i * chunkSize;
        const end = Math.min(start + chunkSize, results.length);
        const chunk = results.slice(start, end);
        let retryCount = 0;
        let success = false;

        while (!success && retryCount < maxRetries) {
          try {
            await axios.post(`http://localhost:8000/api/${endpoint}`, chunk);
            success = true;
          } catch (error) {
            retryCount++;
            if (retryCount === maxRetries) {
              throw error; // Re-throw if all retries failed
            }
            // Wait before retrying (exponential backoff)
            await new Promise((resolve) =>
              setTimeout(resolve, 1000 * Math.pow(2, retryCount))
            );
            console.warn(
              `Retrying chunk ${i + 1}/${totalChunks} (attempt ${
                retryCount + 1
              }/${maxRetries})`
            );
          }
        }

        // Update progress toast
        toast({
          title: "Saving Results",
          description: `Progress: ${Math.round(
            ((i + 1) / totalChunks) * 100
          )}%`,
          status: "info",
          duration: 1000,
          isClosable: true,
        });
      }

      toast({
        title: "Results Saved",
        description: `All ${results.length} results saved successfully`,
        status: "success",
        duration: 3000,
        isClosable: true,
      });
    } catch (error) {
      console.error("Error saving results:", error);
      toast({
        title: "Save Error",
        description: `Error saving results: ${error}`,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsSavingData(false);
    }
  };

  const runSweep = async (data: ParameterSweepParams) => {
    if (isRunning) return;

    // Calculate the number of simulations that will be done based on the parameter ranges
    const donorSteps = parseInt(data.donorConcentrationRange.steps.toString());
    const acceptorSteps = parseInt(
      data.acceptorConcentrationRange.steps.toString()
    );
    const totalSimulations = donorSteps * acceptorSteps;

    setIsRunning(true);
    setSweepResults([]);
    setDetailedResults([]);
    setProgress(0);
    setIsCounterVisible(true);

    try {
      // Reset the simulation counter before starting a new sweep
      await axios.post("http://localhost:8000/api/reset-simulation-count");

      const donorMin = Number(data.donorConcentrationRange.min);
      const donorMax = Number(data.donorConcentrationRange.max);
      const donorLogScale = Boolean(data.donorConcentrationRange.useLogScale);

      const acceptorMin = Number(data.acceptorConcentrationRange.min);
      const acceptorMax = Number(data.acceptorConcentrationRange.max);
      const acceptorLogScale = Boolean(
        data.acceptorConcentrationRange.useLogScale
      );

      // Generate donor concentration values
      let donorValues: number[] = [];
      if (donorLogScale) {
        // For log scale, use powers of 10 directly
        const logMin = Math.floor(Math.log10(donorMin));
        const logMax = Math.ceil(Math.log10(donorMax));

        // Generate values at each power of 10 between min and max
        for (let i = logMin; i <= logMax; i++) {
          const value = Math.pow(10, i);
          // Only include if within the specified range
          if (value >= donorMin && value <= donorMax) {
            donorValues.push(value);
          }
        }

        // If no values were generated, include at least the min and max
        if (donorValues.length === 0) {
          donorValues = [donorMin, donorMax];
        }
      } else {
        const step = (donorMax - donorMin) / (donorSteps - 1);
        for (let i = 0; i < donorSteps; i++) {
          donorValues.push(donorMin + i * step);
        }
      }

      // Generate acceptor concentration values
      let acceptorValues: number[] = [];
      if (acceptorLogScale) {
        // For log scale, use powers of 10 directly
        const logMin = Math.floor(Math.log10(acceptorMin));
        const logMax = Math.ceil(Math.log10(acceptorMax));

        // Generate values at each power of 10 between min and max
        for (let i = logMin; i <= logMax; i++) {
          const value = Math.pow(10, i);
          // Only include if within the specified range
          if (value >= acceptorMin && value <= acceptorMax) {
            acceptorValues.push(value);
          }
        }

        // If no values were generated, include at least the min and max
        if (acceptorValues.length === 0) {
          acceptorValues = [acceptorMin, acceptorMax];
        }
      } else {
        const step = (acceptorMax - acceptorMin) / (acceptorSteps - 1);
        for (let i = 0; i < acceptorSteps; i++) {
          acceptorValues.push(acceptorMin + i * step);
        }
      }

      // Generate percent values if PIN junction and range is specified
      let percentValues: number[] = [];
      if (data.junctionType === "PIN" && data.percentRange) {
        const percentMin = Number(data.percentRange.min);
        const percentMax = Number(data.percentRange.max);
        const percentSteps = Number(data.percentRange.steps);

        if (percentSteps > 1) {
          const step = (percentMax - percentMin) / (percentSteps - 1);
          for (let i = 0; i < percentSteps; i++) {
            percentValues.push(percentMin + i * step);
          }
        } else {
          // If only one step, use the average
          percentValues.push((percentMin + percentMax) / 2);
        }
      } else if (data.junctionType === "PIN" && data.percent) {
        // If PIN junction but no range, use the single percent value
        percentValues.push(Number(data.percent));
      }

      let completedSimulations = 0;
      let allResults: SweepResult[] = [];
      let allDetailedResults: any[] = [];

      // Calculate the actual total number of simulations
      const actualTotalSimulations =
        donorValues.length *
        acceptorValues.length *
        (data.junctionType === "PIN" && percentValues.length > 0
          ? percentValues.length
          : 1);

      // Update toast with correct number of simulations
      toast({
        title: "Starting Parameter Sweep",
        description: `Running ${actualTotalSimulations} simulations...`,
        status: "info",
        duration: 5000,
        isClosable: true,
      });

      // Run simulations for all combinations
      for (const donorConc of donorValues) {
        for (const acceptorConc of acceptorValues) {
          // If PIN junction, iterate through percent values
          if (data.junctionType === "PIN" && percentValues.length > 0) {
            for (const percentValue of percentValues) {
              const simParams = {
                temperature: Number(data.temperature),
                donorConcentration: donorConc,
                acceptorConcentration: acceptorConc,
                nRegionWidth: Number(data.nRegionWidth),
                pRegionWidth: Number(data.pRegionWidth),
                photonEnergy: Number(data.photonEnergy),
                photonFlux: Number(data.photonFlux),
                junctionType: data.junctionType,
                intrinsicConcentration: data.intrinsicConcentration
                  ? Number(data.intrinsicConcentration)
                  : undefined,
                percent: percentValue,
              };

              try {
                const response = await axios.post<SimulationResponse>(
                  "http://localhost:8000/api/simulate",
                  simParams
                );

                const result: SweepResult = {
                  donorConcentration: donorConc,
                  acceptorConcentration: acceptorConc,
                  totalCurrent:
                    response.data.currentGenerationData.solar_cell_parameters
                      .total_current,
                  electronLifetime:
                    response.data.currentGenerationData.jdr[
                      "Electron-lifetime"
                    ],
                  holeLifetime:
                    response.data.currentGenerationData.jdr["Hole-lifetime"],
                  deviceLifetime: Math.min(
                    response.data.currentGenerationData.jdr[
                      "Electron-lifetime"
                    ],
                    response.data.currentGenerationData.jdr["Hole-lifetime"]
                  ),
                  voc: response.data.currentGenerationData.solar_cell_parameters
                    .voc,
                  radPerHour:
                    response.data.currentGenerationData.solar_cell_parameters
                      .rad_per_hour,
                  percent: percentValue,
                };

                allResults.push(result);

                // Also save detailed results
                allDetailedResults.push({
                  ...simParams,
                  ...response.data,
                });

                // Save the last simulation response for display
                setLastSimulationResponse(response.data);
              } catch (error) {
                console.error("Error in simulation:", error);
                toast({
                  title: "Simulation Error",
                  description: `Error in simulation with donor=${donorConc}, acceptor=${acceptorConc}, percent=${percentValue}: ${error}`,
                  status: "error",
                  duration: 5000,
                  isClosable: true,
                });
              }

              completedSimulations++;
              setProgress(
                Math.round(
                  (completedSimulations / actualTotalSimulations) * 100
                )
              );
            }
          } else {
            // For PN junction or no percent range
            const simParams = {
              temperature: Number(data.temperature),
              donorConcentration: donorConc,
              acceptorConcentration: acceptorConc,
              nRegionWidth: Number(data.nRegionWidth),
              pRegionWidth: Number(data.pRegionWidth),
              photonEnergy: Number(data.photonEnergy),
              photonFlux: Number(data.photonFlux),
              junctionType: data.junctionType,
              intrinsicConcentration: data.intrinsicConcentration
                ? Number(data.intrinsicConcentration)
                : undefined,
              percent:
                data.junctionType === "PIN"
                  ? data.percent
                    ? Number(data.percent)
                    : 50.0 // Default 50% for PIN if not specified
                  : 0.0, // Explicitly set to 0.0 for PN junctions
            };

            try {
              const response = await axios.post<SimulationResponse>(
                "http://localhost:8000/api/simulate",
                simParams
              );

              const result: SweepResult = {
                donorConcentration: donorConc,
                acceptorConcentration: acceptorConc,
                totalCurrent:
                  response.data.currentGenerationData.solar_cell_parameters
                    .total_current,
                electronLifetime:
                  response.data.currentGenerationData.jdr["Electron-lifetime"],
                holeLifetime:
                  response.data.currentGenerationData.jdr["Hole-lifetime"],
                deviceLifetime: Math.min(
                  response.data.currentGenerationData.jdr["Electron-lifetime"],
                  response.data.currentGenerationData.jdr["Hole-lifetime"]
                ),
                voc: response.data.currentGenerationData.solar_cell_parameters
                  .voc,
                radPerHour:
                  response.data.currentGenerationData.solar_cell_parameters
                    .rad_per_hour,
                percent: simParams.percent, // Use the consistent value from simParams
              };

              allResults.push(result);

              // Also save detailed results
              allDetailedResults.push({
                ...simParams,
                ...response.data,
              });

              // Save the last simulation response for display
              setLastSimulationResponse(response.data);
            } catch (error) {
              console.error("Error in simulation:", error);
              toast({
                title: "Simulation Error",
                description: `Error in simulation with donor=${donorConc}, acceptor=${acceptorConc}: ${error}`,
                status: "error",
                duration: 5000,
                isClosable: true,
              });
            }

            completedSimulations++;
            setProgress(
              Math.round((completedSimulations / actualTotalSimulations) * 100)
            );
          }
        }
      }

      setSweepResults(allResults);
      setDetailedResults(allDetailedResults);

      // Save results to CSV if requested
      if (saveToCSV && allResults.length > 0) {
        try {
          // Use the chunk saving method for large datasets
          await saveResultsInChunks(allResults, "save-sweep-results");
          await saveResultsInChunks(
            allDetailedResults,
            "save-detailed-results"
          );
        } catch (saveError) {
          console.error("Error saving results:", saveError);
          toast({
            title: "Save Error",
            description: `Error saving results to CSV: ${saveError}`,
            status: "error",
            duration: 5000,
            isClosable: true,
          });
        }
      }

      toast({
        title: "Parameter Sweep Complete",
        description: `Completed ${completedSimulations} simulations`,
        status: "success",
        duration: 5000,
        isClosable: true,
      });

      // Generate plot data
      const plotData = {
        x: allResults.map((r) => r.donorConcentration),
        y: allResults.map((r) => r.totalCurrent),
        type: "scatter",
        mode: "lines+markers",
        name: "Total Current",
      };

      setCurrentPlotData(plotData);
    } catch (error) {
      console.error("Error in parameter sweep:", error);
      toast({
        title: "Parameter Sweep Error",
        description: `Error running parameter sweep: ${error}`,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsRunning(false);
      setTimeout(() => setIsCounterVisible(false), 5000);
    }
  };

  const saveDetailedResultsToCSV = async () => {
    if (sweepResults.length === 0 || detailedResults.length === 0) {
      toast({
        title: "No Results",
        description: "No simulation results to save",
        status: "warning",
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    // Use the chunk saving method
    await saveResultsInChunks(detailedResults, "save-detailed-results");
  };

  // Calculate the current page of results to display
  const paginatedResults = useCallback(() => {
    const startIndex = (currentPage - 1) * pageSize;
    const endIndex = startIndex + pageSize;
    return sweepResults.slice(startIndex, endIndex);
  }, [sweepResults, currentPage, pageSize]);

  const totalPages = Math.ceil(sweepResults.length / pageSize);

  return (
    <VStack spacing={8} align="stretch" w="100%" p={4}>
      {isCounterVisible && (
        <Box
          position="fixed"
          top="20px"
          right="20px"
          bg="blue.600"
          color="white"
          px={4}
          py={2}
          borderRadius="md"
          boxShadow="md"
          zIndex={1000}
        >
          <Text fontWeight="bold">Simulations: {simulationCount}</Text>
        </Box>
      )}

      <form onSubmit={handleSubmit(runSweep)}>
        <VStack spacing={4} align="stretch">
          <FormControl isInvalid={!!errors.temperature}>
            <FormLabel>Temperature (K)</FormLabel>
            <NumberInput min={0} max={1000} step={1}>
              <NumberInputField
                {...register("temperature", {
                  required: "Temperature is required",
                  min: { value: 0, message: "Temperature must be positive" },
                })}
              />
              <NumberInputStepper>
                <NumberIncrementStepper />
                <NumberDecrementStepper />
              </NumberInputStepper>
            </NumberInput>
            {errors.temperature && errors.temperature.message}
          </FormControl>

          <Box borderWidth="1px" borderRadius="lg" p={4} mb={4}>
            <FormLabel>Donor Concentration Range (cm⁻³)</FormLabel>
            <VStack align="stretch" spacing={2}>
              <HStack>
                <FormControl>
                  <FormLabel fontSize="sm">Min</FormLabel>
                  <NumberInput
                    min={1e10}
                    max={1e20}
                    format={(val) => val}
                    precision={2}
                  >
                    <NumberInputField
                      {...register("donorConcentrationRange.min")}
                    />
                  </NumberInput>
                </FormControl>
                <FormControl>
                  <FormLabel fontSize="sm">Max</FormLabel>
                  <NumberInput
                    min={1e10}
                    max={1e20}
                    format={(val) => val}
                    precision={2}
                  >
                    <NumberInputField
                      {...register("donorConcentrationRange.max")}
                    />
                  </NumberInput>
                </FormControl>
              </HStack>
              <HStack>
                <FormControl>
                  <FormLabel fontSize="sm">Steps</FormLabel>
                  <NumberInput
                    min={2}
                    max={20}
                    isDisabled={watch("donorConcentrationRange.useLogScale")}
                  >
                    <NumberInputField
                      {...register("donorConcentrationRange.steps")}
                    />
                  </NumberInput>
                  {watch("donorConcentrationRange.useLogScale") && (
                    <Text fontSize="xs" color="gray.500" mt={1}>
                      With log scale, powers of 10 will be used
                    </Text>
                  )}
                </FormControl>
                <FormControl>
                  <FormLabel fontSize="sm">Log Scale</FormLabel>
                  <Switch
                    {...register("donorConcentrationRange.useLogScale")}
                    size="md"
                    defaultChecked
                  />
                </FormControl>
              </HStack>
            </VStack>
          </Box>

          <Box borderWidth="1px" borderRadius="lg" p={4} mb={4}>
            <FormLabel>Acceptor Concentration Range (cm⁻³)</FormLabel>
            <VStack align="stretch" spacing={2}>
              <HStack>
                <FormControl>
                  <FormLabel fontSize="sm">Min</FormLabel>
                  <NumberInput
                    min={1e10}
                    max={1e20}
                    format={(val) => val}
                    precision={2}
                  >
                    <NumberInputField
                      {...register("acceptorConcentrationRange.min")}
                    />
                  </NumberInput>
                </FormControl>
                <FormControl>
                  <FormLabel fontSize="sm">Max</FormLabel>
                  <NumberInput
                    min={1e10}
                    max={1e20}
                    format={(val) => val}
                    precision={2}
                  >
                    <NumberInputField
                      {...register("acceptorConcentrationRange.max")}
                    />
                  </NumberInput>
                </FormControl>
              </HStack>
              <HStack>
                <FormControl>
                  <FormLabel fontSize="sm">Steps</FormLabel>
                  <NumberInput
                    min={2}
                    max={20}
                    isDisabled={watch("acceptorConcentrationRange.useLogScale")}
                  >
                    <NumberInputField
                      {...register("acceptorConcentrationRange.steps")}
                    />
                  </NumberInput>
                  {watch("acceptorConcentrationRange.useLogScale") && (
                    <Text fontSize="xs" color="gray.500" mt={1}>
                      With log scale, powers of 10 will be used
                    </Text>
                  )}
                </FormControl>
                <FormControl>
                  <FormLabel fontSize="sm">Log Scale</FormLabel>
                  <Switch
                    {...register("acceptorConcentrationRange.useLogScale")}
                    size="md"
                    defaultChecked
                  />
                </FormControl>
              </HStack>
            </VStack>
          </Box>

          <FormControl isInvalid={!!errors.nRegionWidth}>
            <FormLabel>N-Region Width (μm)</FormLabel>
            <NumberInput min={0.1} max={1000} step={0.1}>
              <NumberInputField
                {...register("nRegionWidth", {
                  required: "N-Region Width is required",
                  min: { value: 0.1, message: "Width must be at least 0.1 μm" },
                })}
              />
              <NumberInputStepper>
                <NumberIncrementStepper />
                <NumberDecrementStepper />
              </NumberInputStepper>
            </NumberInput>
            {errors.nRegionWidth && errors.nRegionWidth.message}
          </FormControl>

          <FormControl isInvalid={!!errors.pRegionWidth}>
            <FormLabel>P-Region Width (μm)</FormLabel>
            <NumberInput min={0.1} max={1000} step={0.1}>
              <NumberInputField
                {...register("pRegionWidth", {
                  required: "P-Region Width is required",
                  min: { value: 0.1, message: "Width must be at least 0.1 μm" },
                })}
              />
              <NumberInputStepper>
                <NumberIncrementStepper />
                <NumberDecrementStepper />
              </NumberInputStepper>
            </NumberInput>
            {errors.pRegionWidth && errors.pRegionWidth.message}
          </FormControl>

          <FormControl>
            <FormLabel>Junction Type</FormLabel>
            <Controller
              name="junctionType"
              control={control}
              render={({ field }) => (
                <RadioGroup {...field}>
                  <Stack direction="row">
                    <Radio value="PN">PN Junction</Radio>
                    <Radio value="PIN">PIN Junction</Radio>
                  </Stack>
                </RadioGroup>
              )}
            />
          </FormControl>

          {junctionType === "PIN" && (
            <>
              <FormControl isInvalid={!!errors.intrinsicConcentration}>
                <FormLabel>Intrinsic Concentration (cm⁻³)</FormLabel>
                <NumberInput
                  min={1e10}
                  max={1e16}
                  format={(val) => val}
                  precision={2}
                >
                  <NumberInputField
                    {...register("intrinsicConcentration", {
                      required:
                        "Intrinsic Concentration is required for PIN junction",
                    })}
                  />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
                {errors.intrinsicConcentration &&
                  errors.intrinsicConcentration.message}
              </FormControl>

              <Box borderWidth="1px" borderRadius="lg" p={4} mb={4}>
                <FormLabel>Percentage of Intrinsic Region Range (%)</FormLabel>
                <VStack align="stretch" spacing={2}>
                  <HStack>
                    <FormControl>
                      <FormLabel fontSize="sm">Min (%)</FormLabel>
                      <NumberInput min={1} max={100}>
                        <NumberInputField
                          {...register("percentRange.min", {
                            min: {
                              value: 1,
                              message: "Min percentage must be at least 1%",
                            },
                            max: {
                              value: 100,
                              message: "Min percentage cannot exceed 100%",
                            },
                          })}
                        />
                      </NumberInput>
                    </FormControl>
                    <FormControl>
                      <FormLabel fontSize="sm">Max (%)</FormLabel>
                      <NumberInput min={1} max={100}>
                        <NumberInputField
                          {...register("percentRange.max", {
                            min: {
                              value: 1,
                              message: "Max percentage must be at least 1%",
                            },
                            max: {
                              value: 100,
                              message: "Max percentage cannot exceed 100%",
                            },
                          })}
                        />
                      </NumberInput>
                    </FormControl>
                  </HStack>
                  <FormControl>
                    <FormLabel fontSize="sm">Steps</FormLabel>
                    <NumberInput min={1} max={20}>
                      <NumberInputField {...register("percentRange.steps")} />
                    </NumberInput>
                  </FormControl>
                </VStack>
              </Box>
            </>
          )}

          <FormControl isInvalid={!!errors.photonEnergy}>
            <FormLabel>Photon Energy (eV)</FormLabel>
            <NumberInput min={0} format={(val) => val} precision={2}>
              <NumberInputField
                {...register("photonEnergy", {
                  required: "Photon Energy is required",
                  min: { value: 0, message: "Energy must be positive" },
                })}
              />
              <NumberInputStepper>
                <NumberIncrementStepper />
                <NumberDecrementStepper />
              </NumberInputStepper>
            </NumberInput>
            {errors.photonEnergy && errors.photonEnergy.message}
          </FormControl>

          <FormControl isInvalid={!!errors.photonFlux}>
            <FormLabel>Photon Flux (cm⁻² s⁻¹)</FormLabel>
            <NumberInput min={0} format={(val) => val} precision={2}>
              <NumberInputField
                {...register("photonFlux", {
                  required: "Photon Flux is required",
                  min: { value: 0, message: "Flux must be positive" },
                })}
              />
              <NumberInputStepper>
                <NumberIncrementStepper />
                <NumberDecrementStepper />
              </NumberInputStepper>
            </NumberInput>
            {errors.photonFlux && errors.photonFlux.message}
          </FormControl>

          <FormControl>
            <Checkbox
              isChecked={saveToCSV}
              onChange={(e) => setSaveToCSV(e.target.checked)}
            >
              Save results to CSV
            </Checkbox>
          </FormControl>

          {/* Display the number of simulations prominently */}
          <Box
            mt={4}
            p={3}
            bg="blue.50"
            borderRadius="md"
            borderWidth="1px"
            borderColor="blue.200"
          >
            <Text fontWeight="bold" textAlign="center">
              This parameter sweep will run {simulationsToRun} simulations
            </Text>
          </Box>

          <Button
            colorScheme="blue"
            type="submit"
            isLoading={isRunning}
            loadingText={`Running (${progress}%)`}
            isDisabled={isRunning}
          >
            Run Parameter Sweep
          </Button>
        </VStack>
      </form>

      {/* Parameter sweep results table with pagination */}
      {sweepResults.length > 0 && (
        <Box mt={6}>
          <Flex justifyContent="space-between" alignItems="center">
            <Text fontSize="xl" fontWeight="bold">
              Parameter Sweep Results ({sweepResults.length} simulations)
            </Text>
            <Flex alignItems="center">
              <Text mr={2}>Rows per page:</Text>
              <Select
                value={pageSize}
                onChange={(e) => {
                  setPageSize(Number(e.target.value));
                  setCurrentPage(1); // Reset to first page when changing page size
                }}
                width="100px"
              >
                <option value="10">10</option>
                <option value="20">20</option>
                <option value="50">50</option>
                <option value="100">100</option>
              </Select>
            </Flex>
          </Flex>

          <Box overflowX="auto" mt={4}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid gray" }}>
                  <th style={{ padding: "8px", textAlign: "left" }}>
                    Donor Conc (cm⁻³)
                  </th>
                  <th style={{ padding: "8px", textAlign: "left" }}>
                    Acceptor Conc (cm⁻³)
                  </th>
                  <th style={{ padding: "8px", textAlign: "left" }}>
                    Percent (%)
                  </th>
                  <th style={{ padding: "8px", textAlign: "left" }}>
                    Total Current (A/cm²)
                  </th>
                  <th style={{ padding: "8px", textAlign: "left" }}>
                    Electron Lifetime (s)
                  </th>
                  <th style={{ padding: "8px", textAlign: "left" }}>
                    Hole Lifetime (s)
                  </th>
                  <th style={{ padding: "8px", textAlign: "left" }}>
                    Device Lifetime (s)
                  </th>
                  <th style={{ padding: "8px", textAlign: "left" }}>Voc (V)</th>
                  <th style={{ padding: "8px", textAlign: "left" }}>
                    Rad/Hour
                  </th>
                </tr>
              </thead>
              <tbody>
                {paginatedResults().map((result, idx) => (
                  <tr key={idx} style={{ borderBottom: "1px solid #eee" }}>
                    <td style={{ padding: "8px" }}>
                      {result.donorConcentration.toExponential(2)}
                    </td>
                    <td style={{ padding: "8px" }}>
                      {result.acceptorConcentration.toExponential(2)}
                    </td>
                    <td style={{ padding: "8px" }}>
                      {result.percent !== undefined
                        ? result.percent.toFixed(1)
                        : "0.0"}
                    </td>
                    <td style={{ padding: "8px" }}>
                      {result.totalCurrent.toExponential(4)}
                    </td>
                    <td style={{ padding: "8px" }}>
                      {result.electronLifetime.toExponential(4)}
                    </td>
                    <td style={{ padding: "8px" }}>
                      {result.holeLifetime.toExponential(4)}
                    </td>
                    <td style={{ padding: "8px" }}>
                      {result.deviceLifetime.toExponential(4)}
                    </td>
                    <td style={{ padding: "8px" }}>{result.voc.toFixed(4)}</td>
                    <td style={{ padding: "8px" }}>
                      {result.radPerHour.toExponential(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Box>

          {/* Pagination controls */}
          {totalPages > 1 && (
            <Flex justifyContent="center" mt={4}>
              <ButtonGroup variant="outline" spacing="2">
                <Button
                  onClick={() => setCurrentPage(1)}
                  isDisabled={currentPage === 1}
                >
                  First
                </Button>
                <Button
                  onClick={() =>
                    setCurrentPage((prev) => Math.max(prev - 1, 1))
                  }
                  isDisabled={currentPage === 1}
                >
                  Previous
                </Button>
                <Button variant="solid" colorScheme="blue">
                  {currentPage} of {totalPages}
                </Button>
                <Button
                  onClick={() =>
                    setCurrentPage((prev) => Math.min(prev + 1, totalPages))
                  }
                  isDisabled={currentPage === totalPages}
                >
                  Next
                </Button>
                <Button
                  onClick={() => setCurrentPage(totalPages)}
                  isDisabled={currentPage === totalPages}
                >
                  Last
                </Button>
              </ButtonGroup>
            </Flex>
          )}

          {/* Additional button to save data after sweep is complete */}
          <Button
            mt={4}
            colorScheme="green"
            onClick={saveDetailedResultsToCSV}
            isLoading={isSavingData}
            loadingText="Saving data..."
            isDisabled={isSavingData || sweepResults.length === 0}
          >
            Save All Data to CSV
          </Button>
        </Box>
      )}
    </VStack>
  );
};

export default ParameterSweepForm;
