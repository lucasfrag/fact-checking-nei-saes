import React, { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { MODELS, FC_FAMILY, pathForFeature } from '../autoencoder_registry'
import { get_steering, get_steering_all } from '../interpAPI'
import { SteeringInfo, SteeringAll } from '../types'

// "Browse all claims" of a steering experiment: one row per claim with the verdict at every dose (α/R) side by side;
// click a row to see the model's answer at each dose. Data: viewer/<path>/steering/<atom>_all_<cond>.json.
const LABEL_COLORS: {[key: string]: string} = {NEI: '#f0883e', SUP: '#2a9d5c', REF: '#d9534f', CONF: '#7a5cc4', NOLABEL: '#9aa0a6'}
const Badge = ({label, faded}: {label: string, faded?: boolean}) => (
  <span style={{background: LABEL_COLORS[label] || '#999', color: 'white', borderRadius: 3, padding: '1px 5px',
                fontWeight: 'bold', fontSize: 11, opacity: faded ? 0.35 : 1}}>{label === 'NOLABEL' ? 'no label' : label}</span>)
const tab = (on: boolean) => on ? {background: '#dbeafe', borderColor: '#2a78d6', borderWidth: 2, fontWeight: 700} : {background: '#ffffff'}
const fmt = (b: number) => (b > 0 ? '+' : '') + b.toFixed(2)
const PAGE = 50

export default function SteeringBrowser() {
  const {model, atom} = useParams()
  const path = `${model}/layer-${MODELS[model as string]?.layer}`
  const feat = parseInt(atom as string)
  const [info, setInfo] = useState(null as SteeringInfo | null)
  const [cond, setCond] = useState('')
  const [data, setData] = useState(null as SteeringAll | null)
  const [err, setErr] = useState(null as string | null)
  const [gold, setGold] = useState('all')
  const [onlyChanged, setOnlyChanged] = useState(false)
  const [q, setQ] = useState('')
  const [page, setPage] = useState(0)
  const [open, setOpen] = useState(null as number | null)

  useEffect(() => {
    get_steering(path, feat).then((s) => {
      setInfo(s); const cs = Object.keys(s.curves); setCond(cs.includes('single') ? 'single' : cs[0])
    }).catch((e) => setErr(String(e)))
  }, [path, feat])
  useEffect(() => {
    if (!cond) return
    setData(null); setOpen(null); setPage(0)
    get_steering_all(path, feat, cond).then(setData).catch((e) => setErr(String(e)))
  }, [path, feat, cond])

  const rows = useMemo(() => {
    if (!data) return []
    const z = data.betas.indexOf(0)
    return data.rows.filter((r) => (gold === 'all' || r.gold === gold)
      && (!onlyChanged || r.pred.some((p) => p !== r.pred[z]))
      && (!q.trim() || r.claim.toLowerCase().includes(q.trim().toLowerCase())))
  }, [data, gold, onlyChanged, q])
  useEffect(() => { setPage(0); setOpen(null) }, [gold, onlyChanged, q])

  const featurePage = pathForFeature({atom: feat, autoencoder: FC_FAMILY.get_ae({model: model as string})} as any)
  if (err) return <div className="p-4">Could not load the steering data ({err}).</div>
  if (!info) return <div className="p-4">loading…</div>
  const shown = rows.slice(page * PAGE, (page + 1) * PAGE)
  const nPages = Math.max(1, Math.ceil(rows.length / PAGE))

  return (
    <div className="p-4" style={{maxWidth: 1400, margin: '0 auto'}}>
      <Link className="text-blue-500 hover:text-blue-700 text-sm" to={featurePage}>← back to feature {feat}</Link>
      <h1 className="text-2xl font-bold mt-1">Steering feature {feat}: all {info.n_claims} claims</h1>
      <p className="text-sm text-gray-700 mt-1 mb-2" style={{maxWidth: 1000}}>
        {MODELS[model as string]?.label}. Each row is one AVeriTeC claim; the colored badges are the model's verdict at each dose α/R_c
        (left = feature decreased, right = increased; the gray column is the unmodified model). Click a row to read the model's answer
        at every dose.
      </p>
      <div className="mb-2 text-sm">
        {Object.keys(info.curves).map((c) => (
          <button key={c} onClick={() => setCond(c)} className="rounded-md px-2 py-1 mr-2 border border-gray-300"
            style={tab(cond === c)}>{info.conditions[c]}</button>))}
      </div>
      <div className="mb-2 text-sm" style={{display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center'}}>
        <span>correct answer:{' '}
          <select value={gold} onChange={(e) => setGold(e.target.value)} className="border border-gray-300 rounded-md px-1">
            {['all', 'SUP', 'REF', 'NEI', 'CONF'].map((g) => <option key={g} value={g}>{g}</option>)}
          </select></span>
        <label style={{cursor: 'pointer'}}><input type="checkbox" checked={onlyChanged} onChange={(e) => setOnlyChanged(e.target.checked)}/>
          {' '}only claims whose verdict changes with the dose</label>
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="search claim text"
          className="border border-gray-300 rounded-md px-2" style={{width: 220}}/>
        <span className="text-gray-600">{data ? `${rows.length} of ${data.rows.length} claims` : 'loading…'}</span>
      </div>
      {data &&
        <table className="text-sm" style={{borderCollapse: 'collapse', width: '100%'}}>
          <thead><tr style={{borderBottom: '1px solid #999'}}>
            <th className="px-2 text-left">#</th>
            <th className="px-2 text-left">claim</th>
            <th className="px-2">correct</th>
            {data.betas.map((b) => <th key={b} className="px-1" style={{whiteSpace: 'nowrap', background: b === 0 ? '#f3f3f3' : undefined}}>
              {fmt(b)}</th>)}
          </tr></thead>
          <tbody>
            {shown.map((r) => (
              <React.Fragment key={r.claim_id}>
                <tr onClick={() => setOpen(open === r.claim_id ? null : r.claim_id)}
                    style={{borderBottom: '1px solid #eee', cursor: 'pointer', background: open === r.claim_id ? '#eef4fc' : undefined}}>
                  <td className="px-2 text-gray-500">{r.claim_id}</td>
                  <td className="px-2">{open === r.claim_id ? '▾ ' : '▸ '}{r.claim}</td>
                  <td className="px-2 center"><Badge label={r.gold}/></td>
                  {r.pred.map((p, k) => <td key={k} className="px-1 center" style={{background: data.betas[k] === 0 ? '#f3f3f3' : undefined}}>
                    <Badge label={p} faded={p !== r.gold}/></td>)}
                </tr>
                {open === r.claim_id &&
                  <tr><td colSpan={3 + data.betas.length} className="p-2" style={{background: '#fafcff'}}>
                    {r.evidence.length > 0 &&
                      <div className="text-gray-500 mb-2"><i>Evidence given to the model (first {r.evidence.length}):</i>
                        <ul style={{listStyle: 'disc', paddingLeft: 20}}>
                          {r.evidence.map((e, i) => <li key={i}>{e.text}{e.url ? <> (<a className="text-blue-500" href={e.url} target="_blank" rel="noreferrer">source</a>)</> : null}</li>)}
                        </ul></div>}
                    <table className="activations-table" style={{fontSize: 12}}>
                      <thead><tr><th>α/R<sub>c</sub></th><th>Verdict</th><th>Model output (truncated)</th></tr></thead>
                      <tbody>
                        {data.betas.map((b, k) => (
                          <tr key={b} style={b === 0 ? {background: '#f3f3f3'} : {}}>
                            <td className="center">{fmt(b)}{b === 0 ? ' (none)' : ''}</td>
                            <td className="center"><Badge label={r.pred[k]}/></td>
                            <td className="p-2" style={{whiteSpace: 'pre-wrap'}}>{r.text[k]}</td>
                          </tr>))}
                      </tbody>
                    </table>
                  </td></tr>}
              </React.Fragment>))}
          </tbody>
        </table>}
      {data && nPages > 1 &&
        <div className="mt-2 text-sm" style={{display: 'flex', gap: 8, alignItems: 'center'}}>
          <button className="border border-gray-300 rounded-md px-2" disabled={page === 0} onClick={() => setPage(page - 1)}>← previous</button>
          <span>page {page + 1} of {nPages}</span>
          <button className="border border-gray-300 rounded-md px-2" disabled={page >= nPages - 1} onClick={() => setPage(page + 1)}>next →</button>
        </div>}
      <p className="text-xs text-gray-500 mt-3">Faded badges = verdict differs from the correct answer. Claims and evidence: AVeriTeC
        (Schlichtkrull et al., 2023), CC BY-NC 4.0.</p>
    </div>
  )
}
