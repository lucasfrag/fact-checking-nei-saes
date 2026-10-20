import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';
import { DATA_BASE } from '../interpAPI'

// "How the feature was chosen", as a picture: Cohen's d (x) × cosine with the NEI direction (y) for every
// feature that fires at the verdict. The chosen feature is the one best on BOTH criteria (top-right corner).
// Data: viewer/<model>/layer-<L>/selection.json (fcsae.analysis.selection_map).
type HL = {atom: number, label: string, description?: string}
type Sel = {features: [number, number, number][]}

const size = (d: number) => { const a = Math.abs(d); return a < 0.2 ? 'negligible' : a < 0.5 ? 'small' : a < 0.8 ? 'medium' : 'large' }

export default function SelectionScatter({path, highlighted, onSelect}: {path: string, highlighted: HL[], onSelect: (atom: number) => void}) {
  const [data, setData] = useState(null as Sel | null)
  const [err, setErr] = useState(null as string | null)
  useEffect(() => {
    setData(null); setErr(null)
    fetch(`${DATA_BASE}/${path}/selection.json`).then((r) => r.ok ? r.json() : Promise.reject(r.status))
      .then(setData).catch((e) => setErr(String(e)))
  }, [path])
  if (err) return null
  if (!data) return <div className="text-sm text-gray-500">loading…</div>

  const fs = data.features
  const byAtom = new Map(fs.map((f) => [f[0], f]))
  const hl = highlighted.map((h) => ({h, f: byAtom.get(h.atom)})).filter((x) => x.f) as {h: HL, f: [number, number, number]}[]
  const best = fs.reduce((b, f) => (f[1] > 0 && f[2] > 0 && f[1] * f[2] > b[1] * b[2]) ? f : b, fs[0])
  const hover = (f: [number, number, number], extra = '') =>
    `<b>Feature #${f[0]}</b><br>separation (Cohen's d): <b>${f[1].toFixed(3)}</b> (${size(f[1])}; ` +
    `${f[1] >= 0 ? 'stronger on NEI claims' : 'stronger on SUP/REF claims'})<br>` +
    `alignment with the NEI direction (cosine): <b>${f[2].toFixed(3)}</b>${extra}<br><i>click to open</i>`
  const traces: any[] = [
    {x: fs.map((f) => f[1]), y: fs.map((f) => f[2]), customdata: fs.map((f) => f[0]), text: fs.map((f) => hover(f)),
     hoverinfo: 'text', type: 'scattergl', mode: 'markers',
     marker: {size: 5, opacity: 0.75, color: fs.map((f) => f[1]), colorscale: 'RdBu', reversescale: true, cmin: -0.8, cmax: 0.8}},
    {x: hl.map(({f}) => f[1]), y: hl.map(({f}) => f[2]), customdata: hl.map(({f}) => f[0]), text: hl.map(({f}) => `#${f[0]}`),
     hovertext: hl.map(({h, f}) => hover(f, h.description ? `<br><br><b>${h.label}</b><br>${h.description}` : '')),
     hoverinfo: 'text', type: 'scatter', mode: 'markers+text', textposition: 'middle left', textfont: {size: 12},
     marker: {size: 12, color: 'rgba(0,0,0,0)', line: {color: 'black', width: 2}}},
  ]
  const box = (x: number, y: number, xa: string, ya: string, text: string, color: string) =>
    ({xref: 'paper', yref: 'paper', x, y, xanchor: xa, yanchor: ya, showarrow: false, text: `<b>${text}</b>`,
      font: {size: 12, color}, bgcolor: 'rgba(255,255,255,0.85)', bordercolor: color, borderwidth: 1, borderpad: 4})
  const layout = {
    xaxis: {title: {text: "separation between NEI and SUP/REF (Cohen's d) →", font: {size: 11}}, zeroline: true, zerolinecolor: '#999'},
    yaxis: {title: {text: 'alignment with the NEI direction (cosine) →', font: {size: 11}}, zeroline: true, zerolinecolor: '#999'},
    annotations: [
      box(0.99, 0.99, 'right', 'top', '↗ best on BOTH criteria = chosen', '#c0392b'),
      box(0.01, 0.01, 'left', 'bottom', '↙ mirror: fires more on SUP/REF claims (evidence enough to decide)', '#2a78d6'),
      {x: best[1], y: best[2], text: `chosen: #${best[0]}`, showarrow: true, arrowhead: 2, ax: -60, ay: 30,
       font: {size: 12, color: '#c0392b'}, arrowcolor: '#c0392b'}],
    showlegend: false, margin: {l: 55, r: 10, b: 45, t: 10}, autosize: true, hovermode: 'closest', dragmode: 'pan',
    hoverlabel: {bgcolor: '#fff', bordercolor: '#999', font: {size: 12, color: '#111'}, align: 'left'},
  }
  return (
    <div className="mt-3">
      <p className="text-sm text-gray-700 mb-1" style={{maxWidth: 900}}>
        Each dot is a feature, placed by the <b>two criteria used to choose it</b>: how differently it fires on NEI vs SUP/REF
        claims (<b>right</b> = more on NEI) and how much its direction points the same way as the NEI direction (<b>up</b> = more aligned).
        The chosen feature is the one in the <b>top-right corner</b>, best on both. <i>Scroll to zoom, drag to move, double-click
        to reset.</i>
      </p>
      <div style={{height: 420}}>
        <Plot style={{width: '100%', height: '100%'}} data={traces} layout={layout} useResizeHandler={true}
          config={{displayModeBar: false, scrollZoom: true, doubleClick: 'reset'}}
          onClick={(e: any) => e.points && e.points[0] && onSelect(e.points[0].customdata)}/>
      </div>
    </div>
  )
}
