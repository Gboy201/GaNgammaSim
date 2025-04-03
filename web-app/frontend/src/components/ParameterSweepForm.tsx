import React, { useState, useEffect } from "react";
import {
  Box,
  Button,
  FormControl,
  FormLabel,
  VStack,
  useToast,
  FormErrorMessage,
  Input,
  Text,
  Heading,
  Flex,
  Grid,
  GridItem,
  Progress,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  TableContainer,
  Spinner,
  RadioGroup,
  Stack,
  Radio,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Select,
  Checkbox,
  Badge,
  Switch,
} from "@chakra-ui/react";
import { useForm, Controller } from "react-hook-form";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import axios from "axios";
import { DownloadIcon } from "@chakra-ui/icons";

interface ParameterRange {
  min: number;
  max: number;
  step: number;
}

interface ParameterSweepParams {
  temperature: number;
  donorConcentrationRange: {
    min: string | number;
    max: string | number;
    step: string | number;
    useLogScale: boolean;
  };
  acceptorConcentrationRange: {
    min: string | number;
    max: string | number;
    step: string | number;
    useLogScale: boolean;
  };
  percentRange: ParameterRange;
  nRegionWidth: number;
  pRegionWidth: number;
  photonEnergy: number;
  photonFlux: string | number;
  junctionType: "PN" | "PIN";
  intrinsicConcentration?: number;
}

interface SweepSimulationResult {
  donorConcentration: number;
  acceptorConcentration: number;
  percent: number;
  totalCurrent: number;
  voc: number;
  radPerHour: number;
  deviceLifetime: number;
}

const ParameterSweepForm: React.FC = () => {
  const toast = useToast();
  const queryClient = useQueryClient();
  const [simulationResults, setSimulationResults] = useState<
    SweepSimulationResult[]
  >([]);
  const [simulationProgress, setSimulationProgress] = useState(0);
  const [totalSimulations, setTotalSimulations] = useState(0);
  const [estimatedTime, setEstimatedTime] = useState<number | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [isCancelling, setIsCancelling] = useState(false);
  const cancelRef = React.useRef<boolean>(false);
  const [detailedResults, setDetailedResults] = useState<any[]>([]);

  const {
    register,
    handleSubmit,
    control,
    watch,
    formState: { errors },
  } = useForm<ParameterSweepParams>({
    defaultValues: {
      temperature: 300,
      donorConcentrationRange: {
        min: 1e15,
        max: 1e17,
        step: 1e16,
        useLogScale: false,
      },
      acceptorConcentrationRange: {
        min: 1e15,
        max: 1e17,
        step: 1e16,
        useLogScale: false,
      },
      percentRange: {
        min: 0.1,
        max: 0.9,
        step: 0.1,
      },
      nRegionWidth: 1,
      pRegionWidth: 1,
      photonEnergy: 149796.0,
      photonFlux: 1e13,
      junctionType: "PIN",
      intrinsicConcentration: 1e14,
    },
  });

  const junctionType = watch("junctionType");

  // Function to calculate number of simulations
  const calculateTotalSimulations = (data: ParameterSweepParams) => {
    let donorSteps: number;
    let acceptorSteps: number;

    // For logarithmic scale, calculate steps based on powers of 10
    if (data.donorConcentrationRange.useLogScale) {
      const donorMin = Math.log10(Number(data.donorConcentrationRange.min));
      const donorMax = Math.log10(Number(data.donorConcentrationRange.max));
      donorSteps = Math.floor(donorMax - donorMin) + 1;
    } else {
      donorSteps = Number(data.donorConcentrationRange.step);
    }

    if (data.acceptorConcentrationRange.useLogScale) {
      const acceptorMin = Math.log10(
        Number(data.acceptorConcentrationRange.min)
      );
      const acceptorMax = Math.log10(
        Number(data.acceptorConcentrationRange.max)
      );
      acceptorSteps = Math.floor(acceptorMax - acceptorMin) + 1;
    } else {
      acceptorSteps = Number(data.acceptorConcentrationRange.step);
    }

    const percentSteps = Number(data.percentRange.step);

    return donorSteps * acceptorSteps * percentSteps;
  };

  // Function to generate all parameter combinations
  const generateParameterCombinations = (data: ParameterSweepParams) => {
    const combinations = [];

    const donorMin = Number(data.donorConcentrationRange.min);
    const donorMax = Number(data.donorConcentrationRange.max);
    let donorValues: number[] = [];

    const acceptorMin = Number(data.acceptorConcentrationRange.min);
    const acceptorMax = Number(data.acceptorConcentrationRange.max);
    let acceptorValues: number[] = [];

    const percentMin = data.percentRange.min;
    const percentMax = data.percentRange.max;
    const percentSteps = data.percentRange.step;

    // Generate donor concentration values based on selected scale
    if (data.donorConcentrationRange.useLogScale) {
      // Logarithmic scale (powers of 10)
      const minPower = Math.floor(Math.log10(donorMin));
      const maxPower = Math.ceil(Math.log10(donorMax));

      for (let power = minPower; power <= maxPower; power++) {
        const value = Math.pow(10, power);
        if (value >= donorMin && value <= donorMax) {
          donorValues.push(value);
        }
      }
    } else {
      // Linear scale
      const donorSteps = Number(data.donorConcentrationRange.step);
      const donorStepSize =
        donorSteps > 1 ? (donorMax - donorMin) / (donorSteps - 1) : 0;

      for (let i = 0; i < donorSteps; i++) {
        donorValues.push(donorMin + i * donorStepSize);
      }
    }

    // Generate acceptor concentration values based on selected scale
    if (data.acceptorConcentrationRange.useLogScale) {
      // Logarithmic scale (powers of 10)
      const minPower = Math.floor(Math.log10(acceptorMin));
      const maxPower = Math.ceil(Math.log10(acceptorMax));

      for (let power = minPower; power <= maxPower; power++) {
        const value = Math.pow(10, power);
        if (value >= acceptorMin && value <= acceptorMax) {
          acceptorValues.push(value);
        }
      }
    } else {
      // Linear scale
      const acceptorSteps = Number(data.acceptorConcentrationRange.step);
      const acceptorStepSize =
        acceptorSteps > 1
          ? (acceptorMax - acceptorMin) / (acceptorSteps - 1)
          : 0;

      for (let i = 0; i < acceptorSteps; i++) {
        acceptorValues.push(acceptorMin + i * acceptorStepSize);
      }
    }

    // Generate percent values
    const percentStepSize =
      percentSteps > 1 ? (percentMax - percentMin) / (percentSteps - 1) : 0;
    const percentValues: number[] = [];

    for (let i = 0; i < percentSteps; i++) {
      percentValues.push(percentMin + i * percentStepSize);
    }

    // Create all combinations
    for (const donor of donorValues) {
      for (const acceptor of acceptorValues) {
        for (const percent of percentValues) {
          combinations.push({
            temperature: data.temperature,
            donorConcentration: donor,
            acceptorConcentration: acceptor,
            percent: percent,
            nRegionWidth: data.nRegionWidth,
            pRegionWidth: data.pRegionWidth,
            photonEnergy: data.photonEnergy,
            photonFlux: Number(data.photonFlux),
            junctionType: data.junctionType,
            intrinsicConcentration: data.intrinsicConcentration,
          });
        }
      }
    }

    return combinations;
  };

  const mutation = useMutation({
    mutationFn: async (params: any) => {
      const response = await axios.post(
        "http://localhost:8000/api/simulate",
        params
      );
      return response.data;
    },
  });

  const runParameterSweep = async (data: ParameterSweepParams) => {
    setIsSimulating(true);
    setSimulationResults([]);
    setDetailedResults([]);
    cancelRef.current = false;
    setIsCancelling(false);

    // Calculate total number of simulations
    const numSimulations = calculateTotalSimulations(data);
    setTotalSimulations(numSimulations);

    // Generate all parameter combinations
    const parameterCombinations = generateParameterCombinations(data);

    // Initialize progress
    setSimulationProgress(0);
    setStartTime(Date.now());

    const results: SweepSimulationResult[] = [];
    const detailedResultsArray: any[] = [];

    // Run first 5 simulations to estimate total time
    for (let i = 0; i < Math.min(5, parameterCombinations.length); i++) {
      // Check if cancellation is requested
      if (cancelRef.current) {
        setIsSimulating(false);
        setIsCancelling(false);
        toast({
          title: "Sweep cancelled",
          description: `Completed ${i} of ${numSimulations} simulations before cancellation.`,
          status: "info",
          duration: 5000,
          isClosable: true,
        });
        return;
      }

      try {
        const result = await mutation.mutateAsync(parameterCombinations[i]);

        // Save full result for detailed export
        detailedResultsArray.push({
          ...result,
          donorConcentration: parameterCombinations[i].donorConcentration,
          acceptorConcentration: parameterCombinations[i].acceptorConcentration,
          percent: parameterCombinations[i].percent,
        });

        // Extract relevant data from result
        const sweepResult: SweepSimulationResult = {
          donorConcentration: parameterCombinations[i].donorConcentration,
          acceptorConcentration: parameterCombinations[i].acceptorConcentration,
          percent: parameterCombinations[i].percent,
          totalCurrent:
            result.currentGenerationData.solar_cell_parameters.total_current,
          voc: result.currentGenerationData.solar_cell_parameters.voc,
          radPerHour:
            result.currentGenerationData.solar_cell_parameters.rad_per_hour,
          deviceLifetime:
            600000000 /
            result.currentGenerationData.solar_cell_parameters.rad_per_hour,
        };

        results.push(sweepResult);

        // Update progress
        setSimulationProgress(i + 1);
      } catch (error) {
        console.error("Error in simulation:", error);
      }
    }

    // Calculate estimated time after first 5 simulations
    if (parameterCombinations.length > 5) {
      const currentTime = Date.now();
      const timeElapsed = currentTime - (startTime as number);
      const estimatedTotalTime = (timeElapsed / 5) * numSimulations;
      setEstimatedTime(estimatedTotalTime);
    }

    // Run remaining simulations
    for (let i = 5; i < parameterCombinations.length; i++) {
      // Check if cancellation is requested
      if (cancelRef.current) {
        setIsSimulating(false);
        setIsCancelling(false);
        toast({
          title: "Sweep cancelled",
          description: `Completed ${i} of ${numSimulations} simulations before cancellation.`,
          status: "info",
          duration: 5000,
          isClosable: true,
        });
        return;
      }

      try {
        const result = await mutation.mutateAsync(parameterCombinations[i]);

        // Save full result for detailed export
        detailedResultsArray.push({
          ...result,
          donorConcentration: parameterCombinations[i].donorConcentration,
          acceptorConcentration: parameterCombinations[i].acceptorConcentration,
          percent: parameterCombinations[i].percent,
        });

        // Extract relevant data from result
        const sweepResult: SweepSimulationResult = {
          donorConcentration: parameterCombinations[i].donorConcentration,
          acceptorConcentration: parameterCombinations[i].acceptorConcentration,
          percent: parameterCombinations[i].percent,
          totalCurrent:
            result.currentGenerationData.solar_cell_parameters.total_current,
          voc: result.currentGenerationData.solar_cell_parameters.voc,
          radPerHour:
            result.currentGenerationData.solar_cell_parameters.rad_per_hour,
          deviceLifetime:
            600000000 /
            result.currentGenerationData.solar_cell_parameters.rad_per_hour,
        };

        results.push(sweepResult);

        // Update progress
        setSimulationProgress(i + 1);
      } catch (error) {
        console.error("Error in simulation:", error);
      }
    }

    setSimulationResults(results);
    setDetailedResults(detailedResultsArray);
    setIsSimulating(false);

    // Save results to CSV files
    try {
      // Save summary results
      const response = await axios.post(
        "http://localhost:8000/api/save-sweep-results",
        results
      );
      console.log("Summary results saved to CSV:", response.data);

      // Save detailed results
      const detailedResponse = await axios.post(
        "http://localhost:8000/api/save-detailed-results",
        detailedResultsArray
      );
      console.log("Detailed results saved to CSV:", detailedResponse.data);

      toast({
        title: "Results saved",
        description: `Successfully saved ${results.length} simulation results to CSV files. Detailed data saved to ${detailedResponse.data.filename}.`,
        status: "success",
        duration: 5000,
        isClosable: true,
      });
    } catch (error) {
      console.error("Error saving results to CSV:", error);
      toast({
        title: "Error saving results",
        description: "Failed to save some simulation results to CSV files.",
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const cancelSweep = () => {
    if (isSimulating) {
      setIsCancelling(true);
      cancelRef.current = true;
      toast({
        title: "Cancelling simulation sweep",
        description:
          "The sweep will stop after the current simulation completes.",
        status: "warning",
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const onSubmit = (data: ParameterSweepParams) => {
    runParameterSweep(data);
  };

  // Format minutes and seconds
  const formatTime = (milliseconds: number) => {
    const seconds = Math.floor(milliseconds / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  };

  // Add function to manually save results
  const saveResultsToCSV = async () => {
    if (simulationResults.length === 0) {
      toast({
        title: "No results to save",
        description: "Please run a parameter sweep first.",
        status: "warning",
        duration: 5000,
        isClosable: true,
      });
      return;
    }

    try {
      const response = await axios.post(
        "http://localhost:8000/api/save-sweep-results",
        simulationResults
      );
      console.log("Results saved to CSV:", response.data);
      toast({
        title: "Results saved",
        description: `Successfully saved ${simulationResults.length} simulation results to the CSV file.`,
        status: "success",
        duration: 5000,
        isClosable: true,
      });
    } catch (error) {
      console.error("Error saving results to CSV:", error);
      toast({
        title: "Error saving results",
        description: "Failed to save simulation results to CSV file.",
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    }
  };

  // Add function to manually save detailed results
  const saveDetailedResultsToCSV = async () => {
    if (detailedResults.length === 0) {
      toast({
        title: "No detailed results to save",
        description: "Please run a parameter sweep first.",
        status: "warning",
        duration: 5000,
        isClosable: true,
      });
      return;
    }

    try {
      const response = await axios.post(
        "http://localhost:8000/api/save-detailed-results",
        detailedResults
      );
      console.log("Detailed results saved to CSV:", response.data);
      toast({
        title: "Detailed results saved",
        description: `Successfully saved ${detailedResults.length} detailed simulation results to ${response.data.filename}.`,
        status: "success",
        duration: 5000,
        isClosable: true,
      });
    } catch (error) {
      console.error("Error saving detailed results to CSV:", error);
      toast({
        title: "Error saving detailed results",
        description: "Failed to save detailed simulation results to CSV file.",
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    }
  };

  return (
    <VStack spacing={8} align="stretch" w="100%">
      <Box
        as="form"
        onSubmit={handleSubmit(onSubmit)}
        bg="white"
        p={8}
        borderRadius="lg"
        shadow="base"
      >
        <Heading as="h2" size="md" mb={6} color="blue.600">
          Parameter Sweep
        </Heading>

        <Text mb={6} color="gray.600">
          Specify parameter ranges to run multiple simulations and find optimal
          configurations. Each combination of parameters will be simulated, so
          choose ranges carefully to avoid long processing times.
        </Text>

        <Grid templateColumns="repeat(3, 1fr)" gap={4}>
          <GridItem colSpan={1}>
            <FormControl isInvalid={!!errors.temperature}>
              <FormLabel>Temperature (K)</FormLabel>
              <Input
                type="number"
                {...register("temperature", {
                  required: "Temperature is required",
                  min: { value: 0, message: "Temperature cannot be negative" },
                  valueAsNumber: true,
                })}
              />
              <FormErrorMessage>
                {errors.temperature && errors.temperature.message}
              </FormErrorMessage>
            </FormControl>
          </GridItem>

          <GridItem colSpan={3}>
            <FormControl>
              <FormLabel>Donor Concentration Range (cm⁻³)</FormLabel>
              <Flex gap={4}>
                <Box>
                  <FormLabel fontSize="xs" mb={1}>
                    Minimum
                  </FormLabel>
                  <Input
                    placeholder="Min"
                    type="text"
                    {...register("donorConcentrationRange.min", {
                      required: "Min value is required",
                      min: { value: 0, message: "Cannot be negative" },
                      valueAsNumber: true,
                      pattern: {
                        value: /^[0-9eE+-.]+$/,
                        message: "Please enter a valid number",
                      },
                    })}
                  />
                </Box>
                <Box>
                  <FormLabel fontSize="xs" mb={1}>
                    Maximum
                  </FormLabel>
                  <Input
                    placeholder="Max"
                    type="text"
                    {...register("donorConcentrationRange.max", {
                      required: "Max value is required",
                      min: { value: 0, message: "Cannot be negative" },
                      valueAsNumber: true,
                      pattern: {
                        value: /^[0-9eE+-.]+$/,
                        message: "Please enter a valid number",
                      },
                    })}
                  />
                </Box>
                <Box>
                  <FormLabel fontSize="xs" mb={1}>
                    Number of Steps
                  </FormLabel>
                  <Input
                    placeholder="Steps"
                    type="number"
                    min={1}
                    {...register("donorConcentrationRange.step", {
                      required: "Number of steps is required",
                      min: { value: 1, message: "Must be at least 1" },
                      valueAsNumber: true,
                    })}
                  />
                </Box>
              </Flex>
              <Flex mt={2} alignItems="center">
                <input
                  type="checkbox"
                  id="donorLogScale"
                  {...register("donorConcentrationRange.useLogScale")}
                />
                <FormLabel htmlFor="donorLogScale" mb={0} ml={2} fontSize="sm">
                  Use powers of 10 (log scale) instead of linear steps
                </FormLabel>
              </Flex>
              <Text fontSize="xs" color="gray.500" mt={1}>
                When log scale is enabled, values will be powers of 10 (e.g.,
                1e15, 1e16, 1e17...) within the range
              </Text>
              {(errors.donorConcentrationRange?.min ||
                errors.donorConcentrationRange?.max ||
                errors.donorConcentrationRange?.step) && (
                <FormErrorMessage>
                  Please enter valid concentration values in scientific notation
                  (e.g., 1e15)
                </FormErrorMessage>
              )}
            </FormControl>
          </GridItem>

          <GridItem colSpan={3}>
            <FormControl>
              <FormLabel>Acceptor Concentration Range (cm⁻³)</FormLabel>
              <Flex gap={4}>
                <Box>
                  <FormLabel fontSize="xs" mb={1}>
                    Minimum
                  </FormLabel>
                  <Input
                    placeholder="Min"
                    type="text"
                    {...register("acceptorConcentrationRange.min", {
                      required: "Min value is required",
                      min: { value: 0, message: "Cannot be negative" },
                      valueAsNumber: true,
                      pattern: {
                        value: /^[0-9eE+-.]+$/,
                        message: "Please enter a valid number",
                      },
                    })}
                  />
                </Box>
                <Box>
                  <FormLabel fontSize="xs" mb={1}>
                    Maximum
                  </FormLabel>
                  <Input
                    placeholder="Max"
                    type="text"
                    {...register("acceptorConcentrationRange.max", {
                      required: "Max value is required",
                      min: { value: 0, message: "Cannot be negative" },
                      valueAsNumber: true,
                      pattern: {
                        value: /^[0-9eE+-.]+$/,
                        message: "Please enter a valid number",
                      },
                    })}
                  />
                </Box>
                <Box>
                  <FormLabel fontSize="xs" mb={1}>
                    Number of Steps
                  </FormLabel>
                  <Input
                    placeholder="Steps"
                    type="number"
                    min={1}
                    {...register("acceptorConcentrationRange.step", {
                      required: "Number of steps is required",
                      min: { value: 1, message: "Must be at least 1" },
                      valueAsNumber: true,
                    })}
                  />
                </Box>
              </Flex>
              <Flex mt={2} alignItems="center">
                <input
                  type="checkbox"
                  id="acceptorLogScale"
                  {...register("acceptorConcentrationRange.useLogScale")}
                />
                <FormLabel
                  htmlFor="acceptorLogScale"
                  mb={0}
                  ml={2}
                  fontSize="sm"
                >
                  Use powers of 10 (log scale) instead of linear steps
                </FormLabel>
              </Flex>
              <Text fontSize="xs" color="gray.500" mt={1}>
                When log scale is enabled, values will be powers of 10 (e.g.,
                1e15, 1e16, 1e17...) within the range
              </Text>
              {(errors.acceptorConcentrationRange?.min ||
                errors.acceptorConcentrationRange?.max ||
                errors.acceptorConcentrationRange?.step) && (
                <FormErrorMessage>
                  Please enter valid concentration values in scientific notation
                  (e.g., 1e15)
                </FormErrorMessage>
              )}
            </FormControl>
          </GridItem>

          <GridItem colSpan={3}>
            <FormControl>
              <FormLabel>Percent Range</FormLabel>
              <Flex gap={4}>
                <Box>
                  <FormLabel fontSize="xs" mb={1}>
                    Minimum
                  </FormLabel>
                  <Input
                    placeholder="Min"
                    type="number"
                    step="0.01"
                    min={0}
                    max={1}
                    {...register("percentRange.min", {
                      required: "Min value is required",
                      min: { value: 0, message: "Cannot be negative" },
                      max: { value: 1, message: "Cannot be greater than 1" },
                      valueAsNumber: true,
                    })}
                  />
                </Box>
                <Box>
                  <FormLabel fontSize="xs" mb={1}>
                    Maximum
                  </FormLabel>
                  <Input
                    placeholder="Max"
                    type="number"
                    step="0.01"
                    min={0}
                    max={1}
                    {...register("percentRange.max", {
                      required: "Max value is required",
                      min: { value: 0, message: "Cannot be negative" },
                      max: { value: 1, message: "Cannot be greater than 1" },
                      valueAsNumber: true,
                    })}
                  />
                </Box>
                <Box>
                  <FormLabel fontSize="xs" mb={1}>
                    Number of Steps
                  </FormLabel>
                  <Input
                    placeholder="Steps"
                    type="number"
                    min={1}
                    {...register("percentRange.step", {
                      required: "Number of steps is required",
                      min: { value: 1, message: "Must be at least 1" },
                      valueAsNumber: true,
                    })}
                  />
                </Box>
              </Flex>
            </FormControl>
          </GridItem>

          <GridItem colSpan={1}>
            <FormControl isInvalid={!!errors.nRegionWidth}>
              <FormLabel>N-Region Width (μm)</FormLabel>
              <Input
                type="number"
                step="0.1"
                {...register("nRegionWidth", {
                  required: "N-region width is required",
                  min: { value: 0, message: "Width cannot be negative" },
                  valueAsNumber: true,
                })}
              />
              <FormErrorMessage>
                {errors.nRegionWidth && errors.nRegionWidth.message}
              </FormErrorMessage>
            </FormControl>
          </GridItem>

          <GridItem colSpan={1}>
            <FormControl isInvalid={!!errors.pRegionWidth}>
              <FormLabel>P-Region Width (μm)</FormLabel>
              <Input
                type="number"
                step="0.1"
                {...register("pRegionWidth", {
                  required: "P-region width is required",
                  min: { value: 0, message: "Width cannot be negative" },
                  valueAsNumber: true,
                })}
              />
              <FormErrorMessage>
                {errors.pRegionWidth && errors.pRegionWidth.message}
              </FormErrorMessage>
            </FormControl>
          </GridItem>

          <GridItem colSpan={1}>
            <FormControl isInvalid={!!errors.photonEnergy}>
              <FormLabel>Photon Energy (eV)</FormLabel>
              <Input
                type="number"
                step="0.01"
                {...register("photonEnergy", {
                  required: "Photon energy is required",
                  min: { value: 0, message: "Energy cannot be negative" },
                  valueAsNumber: true,
                })}
              />
              <FormErrorMessage>
                {errors.photonEnergy && errors.photonEnergy.message}
              </FormErrorMessage>
            </FormControl>
          </GridItem>

          <GridItem colSpan={1}>
            <FormControl isInvalid={!!errors.photonFlux}>
              <FormLabel>Photon Flux (photons/cm²·s)</FormLabel>
              <Input
                type="text"
                {...register("photonFlux", {
                  required: "Photon flux is required",
                  min: { value: 0, message: "Flux cannot be negative" },
                  valueAsNumber: true,
                  pattern: {
                    value: /^[0-9eE+-.]+$/,
                    message: "Please enter a valid number",
                  },
                })}
              />
              <FormErrorMessage>
                {errors.photonFlux && errors.photonFlux.message}
              </FormErrorMessage>
            </FormControl>
          </GridItem>

          <GridItem colSpan={2}>
            <FormControl>
              <FormLabel>Junction Type</FormLabel>
              <Controller
                control={control}
                name="junctionType"
                render={({ field }) => (
                  <RadioGroup {...field}>
                    <Stack direction="row">
                      <Radio value="PN">PN</Radio>
                      <Radio value="PIN">PIN</Radio>
                    </Stack>
                  </RadioGroup>
                )}
              />
            </FormControl>
          </GridItem>

          {junctionType === "PIN" && (
            <GridItem colSpan={1}>
              <FormControl isInvalid={!!errors.intrinsicConcentration}>
                <FormLabel>Intrinsic Concentration (cm⁻³)</FormLabel>
                <Input
                  {...register("intrinsicConcentration", {
                    required:
                      "Intrinsic concentration is required for PIN junctions",
                    min: { value: 0, message: "Cannot be negative" },
                    valueAsNumber: true,
                  })}
                />
                <FormErrorMessage>
                  {errors.intrinsicConcentration &&
                    errors.intrinsicConcentration.message}
                </FormErrorMessage>
              </FormControl>
            </GridItem>
          )}
        </Grid>

        <Button
          mt={8}
          colorScheme="blue"
          isLoading={isSimulating}
          type="submit"
          width="full"
        >
          Run Parameter Sweep
        </Button>
      </Box>

      {isSimulating && (
        <Box mt={6} bg="white" p={6} borderRadius="lg" shadow="base">
          <Flex justifyContent="space-between" alignItems="center" mb={3}>
            <Heading as="h3" size="sm">
              Simulation Progress
            </Heading>
            <Button
              colorScheme="red"
              size="sm"
              onClick={cancelSweep}
              isLoading={isCancelling}
              loadingText="Cancelling"
            >
              Cancel Sweep
            </Button>
          </Flex>
          <Progress
            value={(simulationProgress / totalSimulations) * 100}
            size="lg"
            colorScheme="blue"
            hasStripe
            mb={3}
          />
          <Flex justifyContent="space-between">
            <Text>
              {simulationProgress} of {totalSimulations} simulations completed
            </Text>
            {estimatedTime && (
              <Text>
                Estimated time remaining:{" "}
                {formatTime(
                  estimatedTime * (1 - simulationProgress / totalSimulations)
                )}
              </Text>
            )}
          </Flex>
        </Box>
      )}

      {simulationResults.length > 0 && (
        <Box mt={8}>
          <Heading as="h3" size="md" mb={4}>
            Simulation Results
          </Heading>

          {/* Add Save to CSV buttons */}
          <Flex gap={4} mb={4}>
            <Button
              colorScheme="green"
              leftIcon={<DownloadIcon />}
              onClick={saveResultsToCSV}
            >
              Save Summary to CSV
            </Button>

            <Button
              colorScheme="blue"
              leftIcon={<DownloadIcon />}
              onClick={saveDetailedResultsToCSV}
            >
              Save Detailed Results to CSV
            </Button>
          </Flex>

          <Tabs>
            <TabList>
              <Tab>Top Results by Current</Tab>
              <Tab>Top Results by Voc</Tab>
            </TabList>

            <TabPanels>
              <TabPanel>
                <TableContainer>
                  <Table variant="simple" size="sm">
                    <Thead>
                      <Tr>
                        <Th>Donor Conc.</Th>
                        <Th>Acceptor Conc.</Th>
                        <Th>Percent</Th>
                        <Th isNumeric>Current (A/cm²)</Th>
                        <Th isNumeric>Voc (V)</Th>
                        <Th isNumeric>Rad/Hour</Th>
                        <Th isNumeric>Device Lifetime (h)</Th>
                      </Tr>
                    </Thead>
                    <Tbody>
                      {[...simulationResults]
                        .sort((a, b) => b.totalCurrent - a.totalCurrent)
                        .slice(0, 10)
                        .map((result, idx) => (
                          <Tr key={idx}>
                            <Td>
                              {result.donorConcentration.toExponential(2)}
                            </Td>
                            <Td>
                              {result.acceptorConcentration.toExponential(2)}
                            </Td>
                            <Td>{result.percent.toFixed(2)}</Td>
                            <Td isNumeric>
                              {result.totalCurrent.toExponential(2)}
                            </Td>
                            <Td isNumeric>{result.voc.toExponential(2)}</Td>
                            <Td isNumeric>
                              {result.radPerHour.toExponential(2)}
                            </Td>
                            <Td isNumeric>
                              {result.deviceLifetime.toExponential(2)}
                            </Td>
                          </Tr>
                        ))}
                    </Tbody>
                  </Table>
                </TableContainer>
              </TabPanel>
              <TabPanel>
                <TableContainer>
                  <Table variant="simple" size="sm">
                    <Thead>
                      <Tr>
                        <Th>Donor Conc.</Th>
                        <Th>Acceptor Conc.</Th>
                        <Th>Percent</Th>
                        <Th isNumeric>Current (A/cm²)</Th>
                        <Th isNumeric>Voc (V)</Th>
                        <Th isNumeric>Rad/Hour</Th>
                        <Th isNumeric>Device Lifetime (h)</Th>
                      </Tr>
                    </Thead>
                    <Tbody>
                      {[...simulationResults]
                        .sort((a, b) => b.voc - a.voc)
                        .slice(0, 10)
                        .map((result, idx) => (
                          <Tr key={idx}>
                            <Td>
                              {result.donorConcentration.toExponential(2)}
                            </Td>
                            <Td>
                              {result.acceptorConcentration.toExponential(2)}
                            </Td>
                            <Td>{result.percent.toFixed(2)}</Td>
                            <Td isNumeric>
                              {result.totalCurrent.toExponential(2)}
                            </Td>
                            <Td isNumeric>{result.voc.toExponential(2)}</Td>
                            <Td isNumeric>
                              {result.radPerHour.toExponential(2)}
                            </Td>
                            <Td isNumeric>
                              {result.deviceLifetime.toExponential(2)}
                            </Td>
                          </Tr>
                        ))}
                    </Tbody>
                  </Table>
                </TableContainer>
              </TabPanel>
            </TabPanels>
          </Tabs>
        </Box>
      )}
    </VStack>
  );
};

export default ParameterSweepForm;
