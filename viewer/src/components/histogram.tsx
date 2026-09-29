import React from 'react';
import Plot from 'react-plotly.js';
import { FeatureInfo } from '../types'

// Histogram of the feature's (positive) activation at the decision position, split by gold label.
// Each group is normalized to sum to 1 so NEI and SUP/REF can be compared despite different sizes.
const HistogramDisplay = ({ data }: {data: FeatureInfo}) => {
  const hb = data.hist_by_label
  let traces: any[] = []
  if (hb) {
    const centers = hb.edges.slice(0, -1).map((e, i) => (e + hb.edges[i + 1]) / 2)
    const groups: [string, string][] = [['SUP/REF', '#2a78d6'], ['NEI', '#f0883e']]
    traces = groups.map(([name, color]) => {
      const ys = hb[name] || []
      const tot = ys.reduce((a, b) => a + b, 0) || 1
      return {x: centers, y: ys.map((v) => v / tot), name: `${name} (active)`, type: 'scatter', mode: 'lines',
              line: {shape: 'hvh', color}, fill: 'tozeroy'}
    })
  } else {
    const BIN_WIDTH = data.bin_width || 0.2;
    const keys = Object.keys(data.hist).map(Number)
    const min_bin = Math.min(...keys), max_bin = Math.max(...keys)
    const ys = Array.from({length: max_bin - min_bin + 1}, (_, i) => data.hist[min_bin + i] || 0)
    const xs = Array.from({length: max_bin - min_bin + 2}, (_, i) => (min_bin + i) * BIN_WIDTH)
    traces = [{x: xs, y: ys, type: 'scatter', mode: 'lines', line: {shape: 'hvh'}, fill: 'tozeroy'}]
  }
  const layout = {
    legend: {orientation: 'h', y: 1.12, font: {size: 11}},
    xaxis: {title: {text: 'activation at the decision position', font: {size: 11}}},
    yaxis: {title: {text: 'fraction of group', font: {size: 11}}},
    margin: {l: 45, r: 5, b: 40, t: 10},
    autosize: true,
  };
  return <Plot style={{width: '100%', height: '100%'}} data={traces} layout={layout} config={{displayModeBar: false}}/>
}

export default HistogramDisplay;
