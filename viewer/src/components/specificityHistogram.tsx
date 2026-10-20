import React from 'react';
import Plot from 'react-plotly.js';
import { FeatureInfo } from '../types'

// Anthropic-style activation histogram (Bricken et al., 2023; Templeton et al., 2024): one bar per activation range,
// its height = how many claims fall there, its colors = how those claims split between NEI and SUP/REF.
// Data: hist_by_label of the feature JSON (claims where the feature does not fire are left out and counted in the caption).
const SpecificityHistogram = ({ data }: {data: FeatureInfo}) => {
  const hb = data.hist_by_label
  if (!hb || !data.fc) return <div className="text-sm text-gray-500">no distribution data for this feature</div>
  const nei = hb['NEI'] || [], dec = hb['SUP/REF'] || []
  const mid = nei.map((_, i) => (hb.edges[i] + hb.edges[i + 1]) / 2), w = hb.edges[1] - hb.edges[0]
  const fmt = (x: number) => Math.round(x).toLocaleString('en-US')
  const hover = nei.map((v, i) => { const t = v + (dec[i] || 0)
    return `activation ${hb.edges[i].toFixed(3)}–${hb.edges[i + 1].toFixed(3)}<br><b>${fmt(t)} claims</b>: ` +
           `${fmt(v)} NEI (${t ? Math.round(100 * v / t) : 0}%), ${fmt(dec[i] || 0)} SUP/REF` })
  const bar = (y: number[], name: string, color: string) =>
    ({x: mid, y, width: w * 0.9, name, type: 'bar', marker: {color}, hovertext: hover, hoverinfo: 'text'})
  const T = nei.reduce((a, b) => a + b, 0) + dec.reduce((a, b) => a + b, 0)
  const off = Math.round(T / data.fc.frac_active) - T
  const traces = [bar(dec, 'SUP/REF claims (evidence enough)', '#7fa7d9'), bar(nei, 'NEI claims (evidence not enough)', '#e8772e')]
  const layout = {
    barmode: 'stack', bargap: 0,
    xaxis: {title: {text: `how strongly the feature fires →${off >= 1 ? `   (not shown: ${fmt(off)} claims where it does not fire)` : ''}`,
            font: {size: 11}}},
    yaxis: {title: {text: 'number of claims', font: {size: 11}}, tickformat: ',d'},
    legend: {orientation: 'h', x: 1, xanchor: 'right', y: 1.14}, hovermode: 'closest',
    hoverlabel: {bgcolor: '#fff', bordercolor: '#999', font: {size: 12, color: '#111'}, align: 'left'},
    margin: {l: 60, r: 10, b: 45, t: 30}, autosize: true,
  };
  return <Plot style={{width: '100%', height: '100%'}} data={traces} layout={layout} config={{displayModeBar: false}}/>
}

export default SpecificityHistogram;
