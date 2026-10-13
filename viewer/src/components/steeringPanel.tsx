import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import Plot from 'react-plotly.js';
import { SteeringInfo } from '../types'

const COND_COLORS: {[key: string]: string} = {none: '#888888', single: '#2a78d6', iterative: '#d9534f'}
const METRICS: [string, string, string][] = [
  ['nei_rate', 'NEI rate', 'share of claims answered NOT ENOUGH EVIDENCE'],
  ['acc', 'Accuracy', '4-label accuracy against the AVeriTeC gold label'],
  ['macro_f1', 'Macro-F1', 'mean F1 of SUPPORTED, REFUTED and NOT ENOUGH EVIDENCE'],
  ['err_non_nei', 'Error outside NEI', 'error rate among the claims the model did NOT abstain on'],
  ['hit_cap', 'Looping output', 'share of answers that run until the generation limit (1,536 tokens) without finishing, usually a repetition loop'],
]
const LABEL_COLORS: {[key: string]: string} = {NEI: '#f0883e', SUP: '#2a9d5c', REF: '#d9534f', CONF: '#7a5cc4'}
const Badge = ({label}: {label: string}) => (
  <span style={{background: LABEL_COLORS[label] || '#999', color: 'white', borderRadius: 3, padding: '1px 6px',
                fontWeight: 'bold', fontSize: 11}}>{label}</span>)
// selected state = light fill + bold: the inherited stylesheet forces black button text (color: #000 !important)
const tab = (on: boolean) => on ? {background: '#dbeafe', borderColor: '#2a78d6', borderWidth: 2, fontWeight: 700}
                                : {background: '#ffffff'}
const fmt = (b: number) => (b > 0 ? '+' : '') + b.toFixed(2)
const DIST_LABELS = ['SUP', 'REF', 'NEI', 'CONF']

// "What the model answered at each dose": accuracy and the share of each label, one row per α/R (plus the gold answers).
const DistCell = ({v, label}: {v: number, label: string}) => (
  <td className="px-2" style={{whiteSpace: 'nowrap'}}>
    <span style={{display: 'inline-block', width: 60, height: 8, background: '#eee', borderRadius: 4, marginRight: 6, verticalAlign: 'middle'}}>
      <span style={{display: 'block', width: 60 * v, height: 8, borderRadius: 4, background: LABEL_COLORS[label]}}/>
    </span>
    <span style={{fontWeight: v >= 0.5 ? 700 : 400, color: v === 0 ? '#bbb' : undefined}}>{(100 * v).toFixed(1)}%</span>
  </td>)

const DistTable = ({points}: {points: SteeringInfo['curves'][string]}) => {
  if (!points.length || !points[0].dist) return null
  return (
    <table className="text-sm mb-1" style={{borderCollapse: 'collapse'}}>
      <thead><tr style={{borderBottom: '1px solid #999'}}>
        <th className="px-2 text-left" title="dose: α/R < 0 decreases the feature, > 0 increases it">α/R</th>
        <th className="px-2 text-left" title="share of claims whose verdict matches the AVeriTeC gold label">accuracy</th>
        {DIST_LABELS.map((l) => <th key={l} className="px-2 text-left" title="share of claims that received this verdict"><Badge label={l}/></th>)}
      </tr></thead>
      <tbody>
        {points.map((p) => (
          <tr key={p.alpha_over_R} style={{borderBottom: '1px solid #f0f0f0', background: p.alpha_over_R === 0 ? '#f3f3f3' : undefined}}>
            <td className="px-2">{fmt(p.alpha_over_R)}{p.alpha_over_R === 0 ? ' (none)' : ''}</td>
            <td className="px-2">{p.acc.toFixed(3)}</td>
            {DIST_LABELS.map((l) => <DistCell key={l} v={p.dist![l] || 0} label={l}/>)}
          </tr>))}
      </tbody>
    </table>)
}

export default function SteeringPanel({data, path}: {data: SteeringInfo, path: string}) {
  const [metric, setMetric] = useState('nei_rate')
  const conds = Object.keys(data.curves)
  const [cond, setCond] = useState(conds.includes('single') ? 'single' : conds[0])
  const [exIdx, setExIdx] = useState(0)
  const examples = data.examples[cond] || []
  const ex = examples[Math.min(exIdx, examples.length - 1)]

  const traces = conds.map((c) => ({
    x: data.curves[c].map((p) => p.alpha_over_R), y: data.curves[c].map((p) => (p as any)[metric]),
    name: data.conditions[c], type: 'scatter', mode: 'lines+markers', line: {color: COND_COLORS[c] || '#333'},
    hovertemplate: `${data.conditions[c]}<br>α/R = %{x:+.2f}<br>%{y:.3f}<extra></extra>`,
  }))
  const layout = {
    xaxis: {title: {text: 'α/R  (← decrease feature · increase feature →)', font: {size: 11}}, zeroline: false},
    yaxis: {title: {text: METRICS.find((m) => m[0] === metric)![1], font: {size: 11}}, rangemode: 'tozero'},
    shapes: [{type: 'line', x0: 0, x1: 0, yref: 'paper', y0: 0, y1: 1, line: {dash: 'dot', color: 'gray', width: 1}}],
    legend: {orientation: 'h', y: 1.15}, margin: {l: 50, r: 10, b: 45, t: 10}, autosize: true,
  }

  return (
    <div className="mt-4 mb-6 p-3" style={{border: '1px solid #ddd', borderRadius: 6}}>
      <h3 className="text-lg font-bold mb-1">Steering this feature (causal test)</h3>
      <div className="text-sm text-gray-700 mb-2" style={{maxWidth: 950}}>
        The page above is <i>correlational</i> (where the feature fires). Here we <b>intervene</b>: while the model writes its
        verdict on {data.n_claims} claims of the {data.dataset}, we add α times the feature's decoder direction to the residual
        stream at layer {data.layer} (every position; Ferrando et al., 2025, Eq. 4). α is shown as <b>α/R</b>, where
        R = {data.R.toFixed(2)} is the typical residual norm at the verdict position: <b>α/R &lt; 0 decreases</b> the feature,
        <b> α/R &gt; 0 increases</b> it, α/R = 0 is the unmodified model. {conds.length > 1
          ? <>Conditions: <b>No evidence</b> (claim only), <b>Single-step</b> and <b>Iterative</b> (the evidence retrieved by each question-generation method).</>
          : <>Only the <b>no-evidence</b> condition (claim only) has been run for this feature so far{data.curves[conds[0]].length < 9 ? ', and only some doses' : ''}.</>}
        The verdict is the first full label the model writes (NEI if none).
      </div>
      <div className="mb-1">
        {METRICS.map(([key, label, tip]) => (
          <button key={key} title={tip} onClick={() => setMetric(key)}
            className="text-sm rounded-md px-2 py-1 mr-2 mb-1 border border-gray-300" style={tab(metric === key)}>
            {label}</button>))}
      </div>
      <p className="text-xs text-gray-500 mb-1">{METRICS.find((m) => m[0] === metric)![2]}</p>
      <div style={{width: '100%', maxWidth: 760, height: 320}}>
        <Plot style={{width: '100%', height: '100%'}} data={traces} layout={layout} useResizeHandler={true}
          config={{displayModeBar: false}}/>
      </div>

      <h4 className="text-md font-bold mt-3 mb-1">What the model answered at each dose</h4>
      <div className="mb-2 text-sm">
        {conds.map((c) => (
          <button key={c} onClick={() => { setCond(c); setExIdx(0) }}
            className="rounded-md px-2 py-1 mr-2 border border-gray-300" style={tab(cond === c)}>
            {data.conditions[c]}</button>))}
      </div>
      <p className="text-xs text-gray-500 mb-1">Accuracy and the share of the {data.n_claims} claims that received each verdict, for the
        condition selected above.</p>
      <DistTable points={data.curves[cond] || []}/>

      <h4 className="text-md font-bold mt-3 mb-1">Examples: the same claim at every dose
        <Link to={`/steering/${path.split('/')[0]}/${data.feature}`} className="text-sm font-normal ml-3"
          style={{color: '#2a78d6'}}>browse all {data.n_claims} claims →</Link></h4>
      <div className="mb-2 text-sm">
        {conds.map((c) => (
          <button key={c} onClick={() => { setCond(c); setExIdx(0) }}
            className="rounded-md px-2 py-1 mr-2 border border-gray-300" style={tab(cond === c)}>
            {data.conditions[c]}</button>))}
        <div className="mt-2">
          <span className="mr-2">Example:</span>
          <select value={exIdx} onChange={(e) => setExIdx(parseInt(e.target.value))}
            className="border border-gray-300 rounded-md p-2" style={{width: '100%', maxWidth: 950}}>
            {examples.map((e, i) => (
              <option key={i} value={i}>{`${i + 1}. [${e.kind}] ${e.claim.slice(0, 110)}${e.claim.length > 110 ? '…' : ''}`}</option>))}
          </select>
        </div>
      </div>
      {ex &&
        <div className="text-sm">
          <p className="mb-1"><b>Claim:</b> {ex.claim} <span className="ml-2">gold: <Badge label={ex.gold}/></span></p>
          <p className="text-gray-600 mb-1"><i>Why shown:</i> {ex.kind} (first seen at α/R = {fmt(ex.alpha_over_R_selected)}).</p>
          {ex.evidence.length > 0 &&
            <div className="text-gray-500 mb-2"><i>Evidence given to the model (first {ex.evidence.length}):</i>
              <ul style={{listStyle: 'disc', paddingLeft: 20}}>
                {ex.evidence.map((e, i) => <li key={i}>{e.text}{e.url ? <> (<a className="text-blue-500" href={e.url} target="_blank" rel="noreferrer">source</a>)</> : null}</li>)}
              </ul></div>}
          <table className="activations-table" style={{fontSize: 12}}>
            <thead><tr><th>α/R</th><th>Verdict</th><th>Model output (truncated)</th></tr></thead>
            <tbody>
              {ex.responses.map((r) => (
                <tr key={r.alpha_over_R} style={r.alpha_over_R === 0 ? {background: '#f3f3f3'} : {}}>
                  <td className="center">{fmt(r.alpha_over_R)}{r.alpha_over_R === 0 ? ' (none)' : ''}</td>
                  <td className="center"><Badge label={r.pred}/></td>
                  <td className="p-2" style={{whiteSpace: 'pre-wrap'}}>{r.text}</td>
                </tr>))}
            </tbody>
          </table>
        </div>}
      <p className="text-xs text-gray-500 mt-2">Claims and evidence: AVeriTeC (Schlichtkrull et al., 2023), CC BY-NC 4.0.</p>
    </div>
  )
}
