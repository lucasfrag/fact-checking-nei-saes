import React from 'react';
import Plot from 'react-plotly.js';
import { FeatureIndex } from '../types'

// Map of every latent that fires at the verdict position: how often it fires (x, log scale) against how well its
// activation separates NEI from SUP/REF claims (y, AUC; 0.5 = chance). Click a point to open the feature.
const MIN_FRAC = 1e-6  // index.json rounds frac_active to 5 decimals; very rare latents are drawn at this floor

type Props = {index: FeatureIndex, highlighted: {atom: number, label: string}[], onSelect: (atom: number) => void}

export default function FeatureMap({index, highlighted, onSelect}: Props) {
  const fs = index.features.filter((f) => f.auc !== null)
  const x = fs.map((f) => Math.max(f.frac_active, MIN_FRAC))
  const y = fs.map((f) => f.auc as number)
  const text = fs.map((f) => `#${f.atom}<br>AUC ${(f.auc as number).toFixed(3)}<br>active on ${(100 * f.frac_active).toFixed(3)}% of verdicts` +
                             `<br>NEI when active ${(100 * f.nei_rate_when_active).toFixed(1)}%`)
  const hl = highlighted.map((h) => fs.find((f) => f.atom === h.atom)).filter(Boolean) as typeof fs
  const traces: any[] = [
    {x, y, text, customdata: fs.map((f) => f.atom), type: 'scattergl', mode: 'markers', hoverinfo: 'text',
     name: 'latents', marker: {size: 5, color: fs.map((f) => f.nei_rate_when_active), colorscale: 'RdBu', reversescale: true,
       cmin: 0, cmax: 2 * index.meta.nei_base, opacity: 0.75,
       colorbar: {title: {text: 'NEI rate<br>when active', font: {size: 10}}, thickness: 10, len: 0.8}}},
    {x: hl.map((f) => Math.max(f.frac_active, MIN_FRAC)), y: hl.map((f) => f.auc), customdata: hl.map((f) => f.atom),
     text: hl.map((f) => `#${f.atom}`), type: 'scatter', mode: 'markers+text', textposition: 'middle left', hoverinfo: 'skip',
     name: 'highlighted', marker: {size: 12, color: 'rgba(0,0,0,0)', line: {color: 'black', width: 2}}, textfont: {size: 12}},
  ]
  const layout = {
    xaxis: {type: 'log', title: {text: 'fraction of verdicts where the latent is active (log)', font: {size: 11}}},
    yaxis: {title: {text: 'AUC: NEI vs. SUP/REF', font: {size: 11}}},
    shapes: [{type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0.5, y1: 0.5, line: {dash: 'dot', color: 'gray', width: 1}}],
    showlegend: false, margin: {l: 50, r: 10, b: 45, t: 10}, autosize: true, hovermode: 'closest',
  }
  return <Plot style={{width: '100%', height: '420px'}} data={traces} layout={layout} useResizeHandler={true}
    config={{displayModeBar: false}} onClick={(e: any) => e.points && e.points[0] && onSelect(e.points[0].customdata)}/>
}
