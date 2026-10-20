import React, { useState } from 'react'

// "How the NEI feature was chosen": a short card for the SELECTED model (the chart below shows all features).
// Each number explains itself on hover. Values: results/<preset>/abstention.json and feature_<id>_examples.json;
// steering verdicts: nb03 (Llama) and FAQ 22 (DeepSeek, without evidence); Qwen3: steering pending.
type Card = {feature: number, d: number, cos: number, top200: number, controls: boolean | null, steering: string, others?: number[]}
const CARDS: {[path: string]: Card} = {
  'llama-3.1-8b-instruct/layer-15': {feature: 19219, d: 1.276, cos: 0.716, top200: 0.935, controls: true,
    steering: 'controls NEI'},
  'deepseek-r1-distill-llama-8b/layer-15': {feature: 7531, d: 0.68, cos: 0.35, top200: 0.535, controls: false,
    steering: 'no control', others: [18658, 32642, 25349]},
  'qwen3-8b/layer-22': {feature: 12482, d: 1.078, cos: 0.655, top200: 0.875, controls: null, steering: 'done, verdict pending (results on the feature page)'},
  'gemma-3-4b-it/layer-19': {feature: 12903, d: 0.928, cos: 0.471, top200: 0.94, controls: true, steering: 'controls NEI'},
}
const BASE = 0.199   // share of NEI among the 598,930 training decisions
const size = (d: number) => { const a = Math.abs(d); return a < 0.2 ? 'negligible' : a < 0.5 ? 'small' : a < 0.8 ? 'medium' : 'large' }

const Tip = ({children, tip, plain}: {children: React.ReactNode, tip: React.ReactNode, plain?: boolean}) => {
  const [open, setOpen] = useState(false)
  return (
    <span style={{position: 'relative', cursor: 'help', borderBottom: plain ? 'none' : '1px dotted #666', display: 'inline-block'}}
          onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)} onClick={() => setOpen(!open)}>
      {children}
      {open &&
        <span style={{position: 'absolute', top: '120%', left: 0, zIndex: 20, width: 340, fontWeight: 400, fontSize: 12,
                      lineHeight: 1.4, background: '#fff', border: '1px solid #bbb', borderRadius: 6, padding: '8px 10px',
                      boxShadow: '0 4px 12px rgba(0,0,0,.15)', color: '#222'}}>{tip}</span>}
    </span>
  )
}
const How = ({children}: {children: React.ReactNode}) =>
  <span style={{display: 'block', marginTop: 6, paddingTop: 6, borderTop: '1px solid #eee'}}><b>How it is computed: </b>{children}</span>

const Stat = ({value, label, sub, tip}: {value: string, label: string, sub?: string, tip: React.ReactNode}) => (
  <Tip tip={tip} plain>
    <span style={{display: 'inline-flex', flexDirection: 'column', alignItems: 'center', minWidth: 120, padding: '8px 14px',
                  background: '#fff', border: '1px solid #e2e2e2', borderRadius: 10}}>
      <span style={{fontSize: 22, fontWeight: 700, lineHeight: 1.1}}>{value}</span>
      <span style={{fontSize: 12, color: '#555', borderBottom: '1px dotted #999'}}>{label}</span>
      {sub && <span style={{fontSize: 11, color: '#888'}}>{sub}</span>}
    </span>
  </Tip>
)

export default function SelectionMethod({selectedPath, onSelect}: {selectedPath: string, onSelect: (atom: number) => void}) {
  const c = CARDS[selectedPath]
  if (!c) return null
  const Pill = ({atom, big}: {atom: number, big?: boolean}) => (
    <span role="link" tabIndex={0} onClick={() => onSelect(atom)} onKeyDown={(e) => { if (e.key === 'Enter') onSelect(atom) }}
       style={{display: 'inline-block', padding: big ? '2px 12px' : '1px 9px', marginLeft: 6, borderRadius: 999, cursor: 'pointer',
               fontSize: big ? 22 : 13, fontWeight: 700, color: '#4c4fb8', background: big ? '#eef0ff' : '#ececf8'}}>#{atom}</span>)
  return (
    <div className="mt-6 mb-2">
      <h2 className="text-xl font-bold mb-2">How the NEI feature was chosen: Cohen's d × cosine</h2>
      <div style={{border: '1px solid #e2e2e2', borderRadius: 12, background: '#fafafa', maxWidth: 950, padding: '14px 18px'}}>
        <div style={{display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 18}}>
          <div style={{display: 'flex', alignItems: 'center'}}>
            <span style={{fontSize: 13, color: '#555'}}>Chosen feature</span><Pill atom={c.feature} big/>
          </div>
          <div style={{display: 'flex', flexWrap: 'wrap', gap: 10}}>
            <Stat value={c.d.toFixed(3)} label="Cohen's d" sub={size(c.d)} tip={<>How differently the feature fires on NEI vs
              SUP/REF claims, in standard deviations (≈0.2 small, 0.5 medium, 0.8 large; Cohen, 1988). Positive = fires more on NEI claims.
              <How>(mean on NEI − mean on SUP/REF) / pooled standard deviation of the feature's activation, over all decisions.</How></>}/>
            <Stat value={c.cos.toFixed(3)} label="cosine" sub="1 = same direction" tip={<>Angle between the feature's direction and the
              NEI direction: 1 = same direction, 0 = unrelated, −1 = opposite. Two random directions here have ≈0.
              <How>NEI direction v = mean internal state on NEI claims − mean on SUP/REF claims; the feature's direction d is
                its decoder row; cos = (d · v) / (‖d‖ ‖v‖).</How></>}/>
            <Stat value={`${Math.round(100 * c.top200)}%`} label="NEI in its top 200" sub={`average ${Math.round(100 * BASE)}%`}
              tip={<>Of the 200 claims that activate the feature most, the share whose correct answer is NEI.
              <How>sort all 598,930 decisions by the feature's activation, take the top 200, count the NEI ones.</How></>}/>
          </div>
        </div>
        <div style={{marginTop: 12, display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 6, fontSize: 14}}>
          <Tip tip={<>The metrics above are correlational. Steering pushes the model along the feature's direction and checks
            whether its answers change (Ferrando et al., 2025). Only this shows whether the feature controls NEI.</>}>
            <span style={{color: '#555'}}>Steering</span></Tip>
          <span style={{padding: '2px 10px', borderRadius: 999, fontWeight: 700, color: '#fff',
                        background: c.controls === null ? '#888' : c.controls ? '#2a9d5c' : '#c0392b'}}>
            {c.controls === null ? `… ${c.steering}` : c.controls ? '✓ controls NEI' : '✗ does not control NEI'}</span>
          {c.others && <><span style={{color: '#555', marginLeft: 6}}>features that do:</span>{c.others.map((a) => <Pill key={a} atom={a}/>)}</>}
        </div>
        <div style={{marginTop: 10, fontSize: 12, color: '#777'}}>Rule: among all features of the SAE, the one ranking first on
          cosine and on Cohen's d among the features that fire more on NEI (top-right corner of the chart below). Switch the
          model at the top to compare.</div>
      </div>
    </div>
  )
}
