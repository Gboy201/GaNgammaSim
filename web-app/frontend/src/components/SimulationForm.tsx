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
      current: number;
    };
    depletion: {
      generated_carriers: number;
      current: number;
    };
    base: {
      generated_carriers: number;
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
      // Create a unique identifier for logging
      const runId = `single-${formData.donorConcentration}-${formData.acceptorConcentration}`;

      // Ensure all numeric parameters have the correct type
      const formattedParams = {
        temperature: Number(formData.temperature),
        donorConcentration: Number(formData.donorConcentration),
        acceptorConcentration: Number(formData.acceptorConcentration),
        nRegionWidth: Number(formData.nRegionWidth),
        pRegionWidth: Number(formData.pRegionWidth),
        photonEnergy: Number(formData.photonEnergy),
        photonFlux: Number(formData.photonFlux),
        percent:
          formData.percent !== undefined ? Number(formData.percent) : undefined,
        intrinsicConcentration:
          formData.intrinsicConcentration !== undefined
            ? Number(formData.intrinsicConcentration)
            : undefined,
        junctionType: formData.junctionType,
      };

      // Log the exact parameters being sent in a format that can be compared with backend logs
      console.log(
        `[${runId}] Individual simulation sending raw params:`,
        JSON.stringify(formData, null, 2)
      );
      console.log(
        `[${runId}] Individual simulation sending formatted params:`,
        JSON.stringify(formattedParams, null, 2)
      );

      try {
        const response = await axios.post<SimulationResponse>(
          "http://localhost:8000/api/simulate",
          formattedParams
        );

        // More detailed debugging
        console.log(`[${runId}] Received response from simulation`);
        console.log(`[${runId}] Response status:`, response.status);
        console.log(`[${runId}] Response headers:`, response.headers);

        // Check if response data exists
        if (!response.data) {
          console.error(`[${runId}] Response data is empty or null`);
          throw new Error("Empty response received from server");
        }

        // Validate critical parts of the response
        if (!response.data.currentGenerationData) {
          console.error(`[${runId}] Missing currentGenerationData in response`);
        } else if (!response.data.currentGenerationData.jdr) {
          console.error(`[${runId}] Missing jdr in currentGenerationData`);
        } else {
          console.log(
            `[${runId}] JDR keys:`,
            Object.keys(response.data.currentGenerationData.jdr)
          );
        }

        // Log the simulation response ID to correlate with backend logs
        console.log(`[${runId}] Results summary:`, {
          builtInPotential: response.data.builtInPotential,
          depletionWidth: response.data.depletionWidth,
          currentTotal:
            response.data.currentGenerationData?.solar_cell_parameters
              ?.total_current,
          electronLifetime:
            response.data.currentGenerationData?.jdr?.["Electron-lifetime"],
          holeLifetime:
            response.data.currentGenerationData?.jdr?.["Hole-lifetime"],
        });

        return response.data;
      } catch (error) {
        console.error(`[${runId}] Error in simulation request:`, error);
        if (axios.isAxiosError(error)) {
          console.error(`[${runId}] Response status:`, error.response?.status);
          console.error(`[${runId}] Response data:`, error.response?.data);
        }
        throw error;
      }
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
    // Do not convert percent from percentage (0-100) to decimal (0-1)
    // The backend expects percent to be in the range 0-100 and will apply the division by 100 itself

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
                  value: 1e15,
                  message: "Donor concentration must be at least 1e15 cm⁻³",
                },
                valueAsNumber: true,
                pattern: {
                  value: /^[0-9eE+-.]+$/,
                  message: "Please enter a valid number",
                } as any,
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
                  value: 1e15,
                  message: "Acceptor concentration must be at least 1e15 cm⁻³",
                },
                valueAsNumber: true,
                pattern: {
                  value: /^[0-9eE+-.]+$/,
                  message: "Please enter a valid number",
                } as any,
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
