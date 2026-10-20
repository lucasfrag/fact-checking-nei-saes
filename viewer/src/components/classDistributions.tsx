import React from 'react';
import Plot from 'react-plotly.js';
import { FeatureInfo } from '../types'

// How strongly does the feature fire on NEI claims vs SUP/REF claims? One curve per group, each normalized by its own
// total (share of the group's claims at each activation), the usual way to compare a feature or a direction across two
// groups (e.g. Arditi et al., 2024: harmful vs harmless; Ferrando et al., 2025: known vs unknown entities).
// Data: hist_by_label of the feature JSON (claims where the feature does not fire are left out and counted in the caption).
const ClassDistributions = ({ data }: {data: FeatureInfo}) => {
  const hb = data.hist_by_label
  if (!hb || !data.fc) return <div className="text-sm text-gray-500">no distribution data for this feature</div>
  const nei = hb['NEI'] || [], dec = hb['SUP/REF'] || []
  const mid = nei.map((_, i) => (hb.edges[i] + hb.edges[i + 1]) / 2)
  const sN = nei.reduce((a, b) => a + b, 0) || 1, sD = dec.reduce((a, b) => a + b, 0) || 1
  const fmt = (x: number) => Math.round(x).toLocaleString('en-US')
  const p = (x: number) => x >= 1 ? x.toFixed(1) : x.toFixed(2)
  const hover = nei.map((v, i) => `activation ${hb.edges[i].toFixed(3)}–${hb.edges[i + 1].toFixed(3)}<br>` +
    `<b>NEI</b>: ${p(100 * v / sN)}% of NEI claims<br><b>SUP/REF</b>: ${p(100 * (dec[i] || 0) / sD)}% of SUP/REF claims`)
  const wave = (y: number[], name: string, color: string, fill: string) =>
    ({x: mid, y, name, type: 'scatter', mode: 'lines', line: {shape: 'spline', smoothing: 0.8, color, width: 2.5},
      fill: 'tozeroy', fillcolor: fill, hovertext: hover, hoverinfo: 'text'})
  const T = sN + sD
  const off = Math.round(T / data.fc.frac_active) - T
  const traces = [wave(dec.map((v) => 100 * v / sD), 'SUP/REF claims (evidence enough)', '#2a78d6', 'rgba(42,120,214,0.25)'),
                  wave(nei.map((v) => 100 * v / sN), 'NEI claims (evidence not enough)', '#e8772e', 'rgba(240,136,62,0.35)')]
  const layout = {
    xaxis: {title: {text: `how strongly the feature fires →${off >= 1 ? `   (not shown: ${fmt(off)} claims, ${(100 * off / (T + off)).toFixed(1)}% of the dataset, where the activation is 0)` : ''}`,
            font: {size: 11}}},
    yaxis: {title: {text: '% of the claims of each group', font: {size: 11}}, ticksuffix: '%'},
    legend: {orientation: 'h', x: 1, xanchor: 'right', y: 1.14}, hovermode: 'closest',
    hoverlabel: {bgcolor: '#fff', bordercolor: '#999', font: {size: 12, color: '#111'}, align: 'left'},
    margin: {l: 55, r: 10, b: 45, t: 30}, autosize: true,
  };
  return <Plot style={{width: '100%', height: '100%'}} data={traces} layout={layout} config={{displayModeBar: false}}/>
}

export default ClassDistributions;
