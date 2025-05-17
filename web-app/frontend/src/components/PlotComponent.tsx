import React from "react";
import Plot from "react-plotly.js";
import { Box } from "@chakra-ui/react";

interface PlotComponentProps {
  plotData: any;
}

const PlotComponent: React.FC<PlotComponentProps> = ({ plotData }) => {
  // Default layout configuration
  const layout = {
    autosize: true,
    title: plotData.title || "Parameter Sweep Results",
    xaxis: {
      title: plotData.xaxis?.title || "Donor Concentration (cm⁻³)",
      type: "log",
      autorange: true,
    },
    yaxis: {
      title: plotData.yaxis?.title || "Current (A/cm²)",
      type: "log",
      autorange: true,
    },
    margin: { l: 50, r: 50, b: 80, t: 80 },
    showlegend: true,
  };

  // Check if plotData is provided as an array or single object
  const data = Array.isArray(plotData) ? plotData : [plotData];

  return (
    <Box width="100%" height="100%">
      <Plot
        data={data}
        layout={layout}
        useResizeHandler={true}
        style={{ width: "100%", height: "100%" }}
      />
    </Box>
  );
};

export default PlotComponent;
