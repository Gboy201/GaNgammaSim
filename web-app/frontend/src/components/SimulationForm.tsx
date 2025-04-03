import React from "react";
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
} from "@chakra-ui/react";
import { useForm, Controller } from "react-hook-form";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import axios from "axios";
import SimulationResults from "./SimulationResults";

interface SimulationParams {
  temperature: number;
  donorConcentration: number;
  acceptorConcentration: number;
  nRegionWidth: number;
  pRegionWidth: number;
  photonEnergy: number;
  photonFlux: number;
  junctionType: "PN" | "PIN";
  intrinsicConcentration?: number;
  percent?: number;
}

interface SimulationResponse {
  builtInPotential: number;
  message: string;
}

interface SimulationResult {
  position: number;
  electricField: number;
}

const SimulationForm: React.FC = () => {
  const toast = useToast();
  const queryClient = useQueryClient();
  const {
    register,
    handleSubmit,
    control,
    watch,
    formState: { errors },
  } = useForm<SimulationParams>({
    defaultValues: {
      temperature: 300,
      donorConcentration: 1e15,
      acceptorConcentration: 1e15,
      nRegionWidth: 1,
      pRegionWidth: 1,
      photonEnergy: 149796.0,
      photonFlux: 1e13,
      junctionType: "PN",
      intrinsicConcentration: 1e14,
    },
  });

  const junctionType = watch("junctionType");
  const [lastSimulationResponse, setLastSimulationResponse] =
    React.useState<SimulationResponse>();

  const mutation = useMutation<SimulationResponse, Error, SimulationParams>({
    mutationFn: async (formData: SimulationParams) => {
      console.log("Starting simulation with data:", formData);
      const response = await axios.post<SimulationResponse>(
        "http://localhost:8000/api/simulate",
        formData
      );
      console.log("Simulation response:", response.data);
      return response.data;
    },
    onSuccess: async (data: SimulationResponse) => {
      console.log("Simulation completed successfully:", data);
      // Store the simulation response
      setLastSimulationResponse(data);

      // Schedule a refetch of simulation results
      console.log("Starting refetch process...");
      setTimeout(async () => {
        try {
          console.log("Fetching results from /api/results...");
          const resultsResponse = await axios.get<SimulationResult[]>(
            "http://localhost:8000/api/results"
          );
          console.log("Results fetched successfully:", resultsResponse.data);
          queryClient.setQueryData(["simulationResults"], resultsResponse.data);
        } catch (error) {
          console.error("Error fetching results:", error);
          toast({
            title: "Error",
            description: "Failed to fetch simulation results",
            status: "error",
            duration: 5000,
            isClosable: true,
          });
        }
      }, 1000);
    },
    onError: (error: Error) => {
      console.error("Simulation error:", error);
      toast({
        title: "Error",
        description: "Failed to run simulation",
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    },
  });

  const onSubmit = (data: SimulationParams) => {
    mutation.mutate(data);
  };

  return (
    <VStack spacing={8} align="stretch" w="100%" p={4}>
      <Box
        as="form"
        onSubmit={handleSubmit(onSubmit)}
        bg="white"
        p={8}
        borderRadius="lg"
        shadow="base"
        w="50%"
        mx="auto"
      >
        <VStack spacing={8} align="stretch">
          <FormControl isInvalid={!!errors.temperature}>
            <FormLabel>Temperature (K)</FormLabel>
            <NumberInput min={0} defaultValue={300}>
              <NumberInputField
                {...register("temperature", {
                  required: "Temperature is required",
                  min: { value: 0, message: "Temperature cannot be negative" },
                  valueAsNumber: true,
                })}
              />
              <NumberInputStepper>
                <NumberIncrementStepper />
                <NumberDecrementStepper />
              </NumberInputStepper>
            </NumberInput>
            <FormErrorMessage>
              {errors.temperature && errors.temperature.message}
            </FormErrorMessage>
          </FormControl>

          <FormControl isInvalid={!!errors.donorConcentration}>
            <FormLabel>Donor Concentration (cm⁻³)</FormLabel>
            <Input
              type="text"
              defaultValue="1e15"
              {...register("donorConcentration", {
                required: "Donor concentration is required",
                min: {
                  value: 0,
                  message: "Concentration cannot be negative",
                },
                valueAsNumber: true,
                pattern: {
                  value: /^[0-9eE+-.]+$/,
                  message: "Please enter a valid number",
                },
              })}
            />
            <FormErrorMessage>
              {errors.donorConcentration && errors.donorConcentration.message}
            </FormErrorMessage>
          </FormControl>

          <FormControl isInvalid={!!errors.acceptorConcentration}>
            <FormLabel>Acceptor Concentration (cm⁻³)</FormLabel>
            <Input
              type="text"
              defaultValue="1e15"
              {...register("acceptorConcentration", {
                required: "Acceptor concentration is required",
                min: {
                  value: 0,
                  message: "Concentration cannot be negative",
                },
                valueAsNumber: true,
                pattern: {
                  value: /^[0-9eE+-.]+$/,
                  message: "Please enter a valid number",
                },
              })}
            />
            <FormErrorMessage>
              {errors.acceptorConcentration &&
                errors.acceptorConcentration.message}
            </FormErrorMessage>
          </FormControl>

          <FormControl isInvalid={!!errors.nRegionWidth}>
            <FormLabel>N-Region Width (μm)</FormLabel>
            <NumberInput min={0} step={0.1} defaultValue={1}>
              <NumberInputField
                {...register("nRegionWidth", {
                  required: "N-region width is required",
                  min: { value: 0, message: "Width cannot be negative" },
                  valueAsNumber: true,
                })}
              />
              <NumberInputStepper>
                <NumberIncrementStepper />
                <NumberDecrementStepper />
              </NumberInputStepper>
            </NumberInput>
            <FormErrorMessage>
              {errors.nRegionWidth && errors.nRegionWidth.message}
            </FormErrorMessage>
          </FormControl>

          <FormControl isInvalid={!!errors.pRegionWidth}>
            <FormLabel>P-Region Width (μm)</FormLabel>
            <NumberInput min={0} step={0.1} defaultValue={1}>
              <NumberInputField
                {...register("pRegionWidth", {
                  required: "P-region width is required",
                  min: { value: 0, message: "Width cannot be negative" },
                  valueAsNumber: true,
                })}
              />
              <NumberInputStepper>
                <NumberIncrementStepper />
                <NumberDecrementStepper />
              </NumberInputStepper>
            </NumberInput>
            <FormErrorMessage>
              {errors.pRegionWidth && errors.pRegionWidth.message}
            </FormErrorMessage>
          </FormControl>

          <FormControl isInvalid={!!errors.photonEnergy}>
            <FormLabel>Photon Energy (eV)</FormLabel>
            <NumberInput min={0} step={0.01} defaultValue={1.12}>
              <NumberInputField
                {...register("photonEnergy", {
                  required: "Photon energy is required",
                  min: { value: 0, message: "Energy cannot be negative" },
                  valueAsNumber: true,
                })}
              />
              <NumberInputStepper>
                <NumberIncrementStepper />
                <NumberDecrementStepper />
              </NumberInputStepper>
            </NumberInput>
            <FormErrorMessage>
              {errors.photonEnergy && errors.photonEnergy.message}
            </FormErrorMessage>
          </FormControl>

          <FormControl isInvalid={!!errors.photonFlux}>
            <FormLabel>Photon Flux (cm⁻²s⁻¹)</FormLabel>
            <NumberInput min={0} step={1e12} defaultValue={1e13}>
              <NumberInputField
                {...register("photonFlux", {
                  required: "Photon flux is required",
                  min: { value: 0, message: "Flux cannot be negative" },
                  valueAsNumber: true,
                })}
              />
              <NumberInputStepper>
                <NumberIncrementStepper />
                <NumberDecrementStepper />
              </NumberInputStepper>
            </NumberInput>
            <FormErrorMessage>
              {errors.photonFlux && errors.photonFlux.message}
            </FormErrorMessage>
          </FormControl>

          <FormControl>
            <FormLabel>Junction Type</FormLabel>
            <Controller
              name="junctionType"
              control={control}
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

          {junctionType === "PIN" && (
            <>
              <FormControl isInvalid={!!errors.intrinsicConcentration}>
                <FormLabel>Intrinsic Concentration (cm⁻³)</FormLabel>
                <NumberInput min={0} step={1e10} defaultValue={1e10}>
                  <NumberInputField
                    {...register("intrinsicConcentration", {
                      required: "Intrinsic concentration is required for PIN",
                      min: {
                        value: 0,
                        message: "Concentration cannot be negative",
                      },
                      valueAsNumber: true,
                    })}
                  />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
                <FormErrorMessage>
                  {errors.intrinsicConcentration &&
                    errors.intrinsicConcentration.message}
                </FormErrorMessage>
              </FormControl>

              <FormControl isInvalid={!!errors.percent}>
                <FormLabel>Percent (%)</FormLabel>
                <NumberInput min={0} max={100} step={1} defaultValue={50}>
                  <NumberInputField
                    {...register("percent", {
                      required: "Percentage is required for PIN",
                      min: {
                        value: 0,
                        message: "Percentage must be between 0 and 100",
                      },
                      max: {
                        value: 100,
                        message: "Percentage must be between 0 and 100",
                      },
                      valueAsNumber: true,
                    })}
                  />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
                <FormErrorMessage>
                  {errors.percent && errors.percent.message}
                </FormErrorMessage>
              </FormControl>
            </>
          )}

          <Button
            mt={4}
            colorScheme="blue"
            isLoading={mutation.isPending}
            type="submit"
            width="full"
          >
            Run Simulation
          </Button>
        </VStack>
      </Box>

      {/* Results Section */}
      <Box bg="white" p={8} borderRadius="lg" shadow="base" w="100%">
        <SimulationResults simulationResponse={lastSimulationResponse} />
      </Box>
    </VStack>
  );
};

export default SimulationForm;
