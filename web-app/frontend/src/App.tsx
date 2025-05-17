import {
  ChakraProvider,
  Container,
  Box,
  Heading,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
} from "@chakra-ui/react";
import SimulationForm from "./components/SimulationForm";
import ParameterSweepForm from "./components/ParameterSweepForm";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ChakraProvider>
        <Box bg="gray.50" minH="100vh" py={8}>
          <Container maxW="container.xl">
            <Heading as="h1" mb={8} textAlign="center" color="blue.600">
              Physics Simulation Dashboard
            </Heading>

            <Tabs isFitted variant="enclosed" colorScheme="blue">
              <TabList mb="1em">
                <Tab fontWeight="semibold">Single Simulation</Tab>
                <Tab fontWeight="semibold">Parameter Sweep</Tab>
              </TabList>
              <TabPanels>
                <TabPanel>
                  <SimulationForm />
                </TabPanel>
                <TabPanel>
                  <ParameterSweepForm />
                </TabPanel>
              </TabPanels>
            </Tabs>
          </Container>
        </Box>
      </ChakraProvider>
    </QueryClientProvider>
  );
}

export default App;
