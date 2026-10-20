import React from 'react';
import Plot from 'react-plotly.js';
import { FeatureIndex } from '../types'

// Map of every feature that fires at the verdict position: how often it fires (x, log scale) against how well its
// activation separates NEI from SUP/REF claims (y, AUC; 0.5 = no link). Click a point to open the feature.
// Hover text is in plain language (non-specialist readers); highlighted features also show the steering result.
const MIN_FRAC = 1e-6  // index.json rounds frac_active to 5 decimals; very rare features are drawn at this floor

type HL = {atom: number, label: string, description?: string}
type Props = {index: FeatureIndex, highlighted: HL[], onSelect: (atom: number) => void}

const pctStr = (f: number) => f >= 0.01 ? `${(100 * f).toFixed(0)}%` : f >= 0.0001 ? `${(100 * f).toFixed(2)}%` : '<0.01%'
const zone = (auc: number) => auc >= 0.55 ? 'fires more on <b>NEI</b> claims (evidence not enough)'
  : auc <= 0.45 ? 'fires more on <b>SUP/REF</b> claims (evidence enough to decide)' : 'no clear link to NEI'

export default function FeatureMap({index, highlighted, onSelect}: Props) {
  const fs = index.features.filter((f) => f.auc !== null)
  const hover = (f: typeof fs[0], extra = '') =>
    `<b>Feature #${f.atom}</b><br>appears in ${pctStr(f.frac_active)} of answers<br>${zone(f.auc as number)}<br>` +
    `link score <b>${(f.auc as number).toFixed(3)}</b> (0.5 = no link; the farther from 0.5, the stronger)<br>` +
    `NEI among its 20 strongest activations: ${f.nei_top20 === null ? 'n/a' : Math.round(100 * (f.nei_top20 as number)) + '%'}${extra}`
  const hl = highlighted.map((h) => ({h, f: fs.find((f) => f.atom === h.atom)})).filter((x) => x.f) as {h: HL, f: typeof fs[0]}[]
  const traces: any[] = [
    {x: fs.map((f) => Math.max(f.frac_active, MIN_FRAC)), y: fs.map((f) => f.auc), text: fs.map((f) => hover(f)),
     customdata: fs.map((f) => f.atom), type: 'scattergl', mode: 'markers', hoverinfo: 'text',
     name: 'features', marker: {size: 5, color: fs.map((f) => f.nei_top20 ?? f.nei_rate_when_active), colorscale: 'RdBu',
       reversescale: true, cmin: 0, cmax: 2 * index.meta.nei_base, opacity: 0.8,
       colorbar: {title: {text: 'NEI among its<br>20 strongest', font: {size: 10}}, thickness: 10, len: 0.8,
                  tickvals: [0, index.meta.nei_base, 2 * index.meta.nei_base],
                  ticktext: ['0%', `${Math.round(100 * index.meta.nei_base)}% (avg)`, `≥${Math.round(200 * index.meta.nei_base)}%`]}}},
    {x: hl.map(({f}) => Math.max(f.frac_active, MIN_FRAC)), y: hl.map(({f}) => f.auc), customdata: hl.map(({f}) => f.atom),
     text: hl.map(({f}) => `#${f.atom}`), hovertext: hl.map(({h, f}) => hover(f, h.description ? `<br><br><b>${h.label}</b><br>${h.description}` : '')),
     type: 'scatter', mode: 'markers+text', textposition: 'middle left', hoverinfo: 'text',
     name: 'highlighted', marker: {size: 12, color: 'rgba(0,0,0,0)', line: {color: 'black', width: 2}}, textfont: {size: 12}},
  ]
  const layout = {
    xaxis: {type: 'log', title: {text: 'how often the feature appears (% of answers, log scale)', font: {size: 11}},
            tickvals: [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1], ticktext: ['rare', '0.01%', '0.1%', '1%', '10%', '100%']},
    yaxis: {title: {text: 'link to "not enough evidence" (AUC)', font: {size: 11}}},
    shapes: [{type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0.5, y1: 0.5, line: {dash: 'dot', color: 'gray', width: 1}}],
    annotations: [
      {xref: 'paper', yref: 'paper', x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top', showarrow: false,
       text: '<b>↑ fires more on NEI claims (evidence not enough)</b>', font: {size: 13, color: '#c0392b'},
       bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#c0392b', borderwidth: 1, borderpad: 4},
      {xref: 'paper', yref: 'paper', x: 0.01, y: 0.01, xanchor: 'left', yanchor: 'bottom', showarrow: false,
       text: '<b>↓ fires more on SUP/REF claims (evidence enough to decide)</b>', font: {size: 13, color: '#2a78d6'},
       bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#2a78d6', borderwidth: 1, borderpad: 4}],
    showlegend: false, margin: {l: 50, r: 10, b: 45, t: 10}, autosize: true, hovermode: 'closest', dragmode: 'pan',
    hoverlabel: {bgcolor: '#fff', bordercolor: '#999', font: {size: 12, color: '#111'}, align: 'left'},
  }
  return <Plot style={{width: '100%', height: '420px'}} data={traces} layout={layout} useResizeHandler={true}
    config={{displayModeBar: false, scrollZoom: true, doubleClick: 'reset'}} onClick={(e: any) => e.points && e.points[0] && onSelect(e.points[0].customdata)}/>
}
