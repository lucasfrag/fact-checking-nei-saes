import React, { useEffect, useState } from 'react'
import Plot from 'react-plotly.js'
import { DATA_BASE } from '../interpAPI'

// "How strong is the steering push?": the size of the numbers inside each model and the ruler used for the dose.
// Data: viewer/scale.json (decision positions of the SAE training data: median |value|, the dimension with the largest
// typical |value|, its share of the squared length, mean length R = E‖x‖ and centered length R_c = E‖x − μ‖).
type M = {model: string, name: string, layer: number, d: number, typical_value: number, largest_dim: number,
          largest_dim_value: number, R: number, R_c: number, share_norm2_largest_dim: number}

export default function SteeringScale() {
  const [ms, setMs] = useState(null as M[] | null)
  useEffect(() => { fetch(`${DATA_BASE}/scale.json`).then((r) => r.ok ? r.json() : null).then((d) => d && setMs(d.models)).catch(() => null) }, [])
  if (!ms) return null
  const names = ms.map((m) => `${m.name} (layer ${m.layer})`)
  const ratio = ms.map((m) => m.R / m.R_c)
  const bars = [
    {type: 'bar', orientation: 'h', y: names, x: ms.map(() => 1), name: 'variation between claims (R_c) = 1',
     marker: {color: '#2a78d6'}, hoverinfo: 'skip', width: 0.35, offset: -0.4},
    {type: 'bar', orientation: 'h', y: names, x: ratio, name: 'total length of the vector',
     marker: {color: '#b8c2cc'}, text: ratio.map((r) => `${r.toFixed(1)} × the variation`), textposition: 'outside',
     cliponaxis: false, hoverinfo: 'skip', width: 0.35, offset: 0.02},
  ]
  const g = ms.find((m) => m.share_norm2_largest_dim > 0.5)
  return (
    <details className="mt-6">
      <summary className="text-xl font-bold" style={{cursor: 'pointer'}}>How strong is the steering push?</summary>
      <p className="text-sm text-gray-700 mt-2" style={{maxWidth: 950}}>
        The steering push is measured in units of <b style={{color: '#2a78d6'}}>R<sub>c</sub></b>: how much the model's internal vector
        typically changes from one claim to another. Dose 3 = a push of three such "variations". The chart compares that unit with
        the <b style={{color: '#7c8896'}}>total length</b> of the vector.
      </p>
      <div style={{height: 230, maxWidth: 950}}>
        <Plot style={{width: '100%', height: '100%'}} data={bars as any} config={{displayModeBar: false}} useResizeHandler={true}
          layout={{barmode: 'overlay', xaxis: {range: [0, Math.max(...ratio) * 1.18], title: {text: 'size, in units of the variation between claims (R_c = 1)', font: {size: 11}}},
                   yaxis: {automargin: true}, legend: {orientation: 'h', y: 1.18}, margin: {l: 10, r: 10, t: 30, b: 45}, autosize: true}}/>
      </div>
      <p className="text-sm text-gray-700 mt-2" style={{maxWidth: 950}}>
        In Llama and Qwen3 the vector is 5 to 6 times longer than its variation between claims.
        {g && <> In {g.name} it is {(g.R / g.R_c).toFixed(0)} times longer, because a single dimension (dimension {g.largest_dim}, about{' '}
          {Math.round(g.largest_dim_value).toLocaleString('en-US')} on every claim) takes {(100 * g.share_norm2_largest_dim).toFixed(0)}% of the
          length; this is known as a "massive activation" (Sun et al., 2024).</>} That is why the dose is measured in variations (α/R<sub>c</sub>)
        and not in lengths: the same dose then means a push of the same relative strength in every model.
      </p>
    </details>
  )
}
