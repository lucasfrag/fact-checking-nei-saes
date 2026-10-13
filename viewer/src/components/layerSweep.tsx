import React, { useEffect, useState } from 'react'
import { DATA_BASE } from '../interpAPI'

// Layer sweep: for every layer of each model, how well the abstention direction (mean NEI − mean SUP/REF, no SAE)
// separates NEI from SUP/REF claims on held-out data (AUC). Used to choose the layer where the SAE is trained.
// Data: viewer/layer_sweeps/<model>.json (fcsae.analysis.layer_sweep).
type Sweep = {model: string, n_per_class: number, layer_used: number, auc_by_layer: number[], best_layer: number, best_auc: number}
const MODELS: [string, string][] = [
  ['llama-3.1-8b-instruct', 'Llama-3.1-8B'], ['qwen3-8b', 'Qwen3-8B'], ['gemma-3-4b-it', 'Gemma-3-4B'],
  ['deepseek-r1-distill-llama-8b', 'DeepSeek-R1-Distill-Llama-8B']]
const LO = 0.5, HI = 0.9   // bar scale

export default function LayerSweep() {
  const [data, setData] = useState({} as {[m: string]: Sweep})
  useEffect(() => {
    Promise.all(MODELS.map(([m]) => fetch(`${DATA_BASE}/layer_sweeps/${m}.json`).then((r) => r.ok ? r.json() : null).catch(() => null)))
      .then((xs) => setData(Object.fromEntries(xs.filter(Boolean).map((x: Sweep) => [x.model, x]))))
  }, [])
  const ms = MODELS.filter(([m]) => data[m])
  if (!ms.length) return null
  const nL = Math.max(...ms.map(([m]) => data[m].auc_by_layer.length))

  const cell = (s: Sweep, l: number) => {
    const v = s.auc_by_layer[l]
    if (v === undefined) return <td key={s.model} className="px-2" style={{color: '#ccc'}}>—</td>
    const used = l === s.layer_used, best = l === s.best_layer
    const w = Math.max(0, Math.min(1, (v - LO) / (HI - LO))) * 70
    return (
      <td key={s.model} className="px-2 py-0" style={{background: used ? '#fff1d6' : undefined, whiteSpace: 'nowrap'}}>
        <span style={{display: 'inline-block', width: 70, height: 8, background: '#eee', borderRadius: 4, marginRight: 6, verticalAlign: 'middle'}}>
          <span style={{display: 'block', width: w, height: 8, borderRadius: 4, background: used ? '#e8772e' : '#7fa7d9'}}/>
        </span>
        <span style={{fontWeight: used ? 700 : 400}}>{v.toFixed(3)}</span>
        {best && <span title="best layer" style={{color: '#c0392b'}}> ★</span>}
        {used && <span style={{fontSize: 11, color: '#b35c00'}}> used</span>}
      </td>)
  }
  return (
    <details className="mt-6">
      <summary className="text-xl font-bold" style={{cursor: 'pointer'}}>Which layer? Abstention direction AUC by layer</summary>
      <p className="text-sm text-gray-700 mt-2 mb-2" style={{maxWidth: 950}}>
        For every layer, we take the difference between the average internal state on NEI claims and on SUP/REF claims
        (the <b>abstention direction</b>, no SAE) and measure on held-out claims how well it separates the two
        (<b>AUC</b>: 0.5 = chance, 1 = perfect). The SAE is trained at a layer where this is high: the
        <span style={{background: '#fff1d6', padding: '0 4px'}}>highlighted</span> row is the layer used, ★ is the best layer.
        Each model "lights up" at a different depth (Qwen3 only from layer ~18). Rule for every model: the layer with the highest AUC
        (3 decimals; ties → the first one).
      </p>
      <table className="text-sm" style={{borderCollapse: 'collapse'}}>
        <thead><tr style={{borderBottom: '1px solid #999'}}>
          <th className="px-2 text-left">layer</th>
          {ms.map(([m, label]) => <th key={m} className="px-2 text-left">{label}
            <div style={{fontWeight: 400, fontSize: 11, color: '#777'}}>{data[m].n_per_class} claims per class</div></th>)}
        </tr></thead>
        <tbody>
          {Array.from({length: nL}, (_, l) => (
            <tr key={l} style={{borderBottom: '1px solid #f0f0f0'}}>
              <td className="px-2 center" style={{color: '#555'}}>{l}</td>
              {ms.map(([m]) => cell(data[m], l))}
            </tr>))}
        </tbody>
      </table>
    </details>
  )
}
