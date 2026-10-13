import React, { useEffect, useMemo, useState } from 'react'
import { DATA_BASE, get_index } from '../interpAPI'

// Sortable table of every feature that fires at the verdict: candidate rank (by |Cohen's d|, the criterion used to pick
// the candidates), Cohen's d, cosine, AUC, NEI among its 20 strongest, frequency and the steering result.
// Data: viewer/<model>/layer-<L>/selection.json (d, cosine) + index.json (frequency, AUC, NEI top-20).
export type Verdict = 'controls' | 'no' | 'partial'
type Row = {atom: number, rank: number, d: number, cos: number, auc: number | null, top20: number | null, freq: number, steer?: Verdict}
type Key = 'rank' | 'atom' | 'd' | 'cos' | 'auc' | 'top20' | 'freq' | 'steer'

const VERDICT: {[v: string]: [string, string]} = {
  controls: ['✓ controls', '#2a9d5c'], no: ['✗ no effect', '#c0392b'], partial: ['◐ one side only', '#d4880f']}
const COLS: {key: Key, label: string, tip: string}[] = [
  {key: 'rank', label: 'candidate', tip: "Position when all features are ranked by |Cohen's d| (how the candidates were picked)"},
  {key: 'atom', label: 'feature', tip: 'Feature number (click a row to open it)'},
  {key: 'd', label: "Cohen's d", tip: 'Separation NEI vs SUP/REF in standard deviations; positive = fires more on NEI claims'},
  {key: 'cos', label: 'cosine', tip: 'Alignment of the feature direction with the abstention direction (1 same, 0 unrelated, −1 opposite)'},
  {key: 'auc', label: 'AUC', tip: 'Link score NEI vs SUP/REF from this feature alone (0.5 = no link); approximate, from histograms'},
  {key: 'top20', label: 'NEI in top 20', tip: 'Share of NEI among the 20 claims that activate it most (average 20%)'},
  {key: 'freq', label: 'appears in', tip: 'Share of answers in which the feature fires at the verdict'},
  {key: 'steer', label: 'steering', tip: 'Result of pushing the model along this feature (only the tested ones)'},
]

export default function CandidatesTable({path, steering, onSelect}: {path: string, steering: {[atom: number]: Verdict}, onSelect: (atom: number) => void}) {
  const [rows, setRows] = useState(null as Row[] | null)
  const [sort, setSort] = useState({key: 'rank' as Key, asc: true})
  const [n, setN] = useState(25)
  const [q, setQ] = useState('')
  useEffect(() => {
    setRows(null); setN(25)
    Promise.all([fetch(`${DATA_BASE}/${path}/selection.json`).then((r) => r.json()), get_index(path)]).then(([sel, idx]) => {
      const meta = new Map(idx.features.map((f) => [f.atom, f]))
      const base = (sel.features as [number, number, number][]).map(([atom, d, cos]) => {
        const m = meta.get(atom)
        return {atom, d, cos, auc: m ? m.auc : null, top20: m ? m.nei_top20 : null, freq: m ? m.frac_active : 0, steer: steering[atom], rank: 0}
      })
      base.sort((a, b) => Math.abs(b.d) - Math.abs(a.d)).forEach((r, i) => { r.rank = i + 1 })
      setRows(base)
    }).catch(() => setRows([]))
  }, [path])

  const shown = useMemo(() => {
    if (!rows) return []
    const val = (r: Row): number => sort.key === 'steer' ? (r.steer ? ['controls', 'partial', 'no'].indexOf(r.steer) : 9)
      : ((r as any)[sort.key] ?? -Infinity)
    const f = q.trim() ? rows.filter((r) => String(r.atom).includes(q.trim())) : rows
    return [...f].sort((a, b) => (sort.asc ? 1 : -1) * (val(a) - val(b)))
  }, [rows, sort, q])
  if (!rows || !rows.length) return null

  const click = (key: Key) => setSort(sort.key === key ? {key, asc: !sort.asc} : {key, asc: key === 'rank' || key === 'atom' || key === 'steer'})
  const pct = (x: number | null) => x === null ? '—' : `${(100 * x).toFixed(x < 0.01 ? 2 : 0)}%`
  return (
    <div className="mt-3">
      <h3 className="text-md font-bold mb-1">All features, ranked</h3>
      <p className="text-sm text-gray-600 mb-1">Click a column title to sort (again to reverse). Candidates are ranked by |Cohen's d|;
        the tested ones show their steering result.
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="find feature #" className="border border-gray-300 rounded-md ml-2 px-2"
          style={{width: 130}}/></p>
      <table className="text-sm" style={{borderCollapse: 'collapse'}}>
        <thead><tr style={{borderBottom: '1px solid #999'}}>
          {COLS.map((c) => (
            <th key={c.key} title={c.tip} onClick={() => click(c.key)} className="px-2 py-1"
                style={{cursor: 'pointer', whiteSpace: 'nowrap', userSelect: 'none'}}>
              {c.label}{sort.key === c.key ? (sort.asc ? ' ▲' : ' ▼') : ''}
            </th>))}
        </tr></thead>
        <tbody>
          {shown.slice(0, n).map((r) => (
            <tr key={r.atom} onClick={() => onSelect(r.atom)} style={{borderBottom: '1px solid #eee', cursor: 'pointer',
                background: r.steer ? '#fff8ec' : undefined}}>
              <td className="px-2 center">{r.rank}°</td>
              <td className="px-2 center text-blue-600">#{r.atom}</td>
              <td className="px-2 center">{r.d.toFixed(3)}</td>
              <td className="px-2 center">{r.cos.toFixed(3)}</td>
              <td className="px-2 center">{r.auc === null ? '—' : r.auc.toFixed(3)}</td>
              <td className="px-2 center">{pct(r.top20)}</td>
              <td className="px-2 center">{pct(r.freq)}</td>
              <td className="px-2 center" style={{color: r.steer ? VERDICT[r.steer][1] : '#aaa', fontWeight: r.steer ? 600 : 400}}>
                {r.steer ? VERDICT[r.steer][0] : 'not tested'}</td>
            </tr>))}
        </tbody>
      </table>
      {n < shown.length &&
        <button onClick={() => setN(n + 50)} className="text-sm text-gray-600 mt-1">show more ({shown.length - n} left)</button>}
    </div>
  )
}
