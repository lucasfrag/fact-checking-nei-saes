import React, { useEffect, useState } from "react"
import { useNavigate } from "react-router-dom"
import { Feature, FeatureIndex, FeatureIndexEntry } from "./types"
import FeatureSelect from "./components/featureSelect"
import FeatureMap from "./components/featureMap"
import SelectionMethod from "./components/selectionMethod"
import SelectionScatter from "./components/selectionScatter"
import CandidatesTable, { Verdict } from "./components/candidatesTable"
import LayerSweep from "./components/layerSweep"
import { pathForFeature, DEFAULT_AUTOENCODER } from "./autoencoder_registry"
import { get_index } from "./interpAPI"

// Features highlighted in the accompanying study (per model folder).
const HIGHLIGHTED: {[path: string]: {atom: number, label: string, description: string}[]} = {
  'llama-3.1-8b-instruct/layer-15': [
    {atom: 19219, label: '19219: the abstention feature', description: 'Steering CONFIRMED: turning it down makes the model decide; turning it up makes it say "not enough evidence".'},
    {atom: 14953, label: '14953: the mirror case', description: 'Fires more on SUP/REF claims (evidence enough to decide). Not tested with steering.'},
  ],
  'qwen3-8b/layer-22': [
    {atom: 12482, label: '12482: the abstention candidate', description: 'Top feature on BOTH selection criteria (Cohen\'s d and cosine), as 19219 in Llama. Steering results on its page: turning it down makes the model answer SUPPORTED to almost everything.'},
    {atom: 21289, label: '21289: the mirror case', description: 'Fires more on SUP/REF claims (evidence enough to decide). Not tested with steering.'},
  ],
  'gemma-3-4b-it/layer-19': [
    {atom: 12903, label: '12903: the abstention candidate', description: 'Top on cosine and on positive Cohen\'s d; pushes the output towards "Depends", "somewhat", "maybe". Steering not run yet.'},
    {atom: 2170, label: '2170: the mirror case', description: 'Largest |Cohen\'s d|, but negative: fires more on SUP/REF claims (evidence enough to decide).'},
  ],
  // DeepSeek: abstention is spread over several features in this SAE; steering results without evidence (FAQ 22)
  'deepseek-r1-distill-llama-8b/layer-15': [
    {atom: 18658, label: '18658: controls abstention', description: 'Steering CONFIRMED in both directions, with the widest range without breaking the model.'},
    {atom: 32642, label: '32642: controls abstention', description: 'Steering CONFIRMED in both directions.'},
    {atom: 25349, label: '25349: controls abstention (mirror)', description: 'Steering CONFIRMED: turning it UP makes the model abstain LESS.'},
    {atom: 19789, label: '19789: partial (mirror)', description: 'Steering: only turning it up has an effect (less abstention).'},
    {atom: 7531, label: '7531: no control', description: 'Strongest link score, but steering it does NOT change abstention: a link is not the same as control.'},
  ],
}

// short, model-specific notes (what the map shows differs between models)
const NOTES: {[path: string]: {map?: string, highlighted?: string}} = {
  'llama-3.1-8b-instruct/layer-15': {map: 'Here one star (19219) stands far above all the others.'},
  'qwen3-8b/layer-22': {map: 'As in Llama, one feature (12482) stands out on both selection criteria.'},
  'gemma-3-4b-it/layer-19': {map: 'Weaker signal than Llama and Qwen3: 12903 leads on cosine, and the largest |d| belongs to a mirror feature (2170).'},
  'deepseek-r1-distill-llama-8b/layer-15': {
    map: 'Here no single star stands far above the rest: the signal is shared by several features.',
    highlighted: 'In DeepSeek no single feature does the whole job: several features each carry part of the "not enough evidence" signal. Steering showed that the feature with the strongest link (7531) does not control abstention, while three others do (18658, 32642, 25349).',
  },
}

// steering results (Llama: nb03; DeepSeek: FAQ 22, without evidence)
const STEERING: {[path: string]: {[atom: number]: Verdict}} = {
  'llama-3.1-8b-instruct/layer-15': {19219: 'controls'},
  'deepseek-r1-distill-llama-8b/layer-15': {7531: 'no', 32642: 'controls', 18658: 'controls', 25349: 'controls', 19789: 'partial'},
}

export default function Welcome() {
  const navigate = useNavigate()
  const [feature, setFeature] = useState({atom: 19219, autoencoder: DEFAULT_AUTOENCODER} as Feature)
  const [index, setIndex] = useState(null as FeatureIndex | null)
  const [err, setErr] = useState(null as string | null)
  useEffect(() => {
    get_index(feature.autoencoder.path).then(setIndex).catch((e) => setErr(String(e)))
  }, [feature.autoencoder.path])

  const go = (atom: number) => navigate(pathForFeature({atom, autoencoder: feature.autoencoder}))
  const ranked = (key: 'auc' | 'nei_top20', n: number, ascending = false) => {
    if (!index) return [] as FeatureIndexEntry[]
    const sign = ascending ? -1 : 1
    return index.features.filter((f) => f[key] !== null && f.frac_active >= 0.001)
      .sort((a, b) => sign * ((b[key] as number) - (a[key] as number))).slice(0, n)
  }
  const highlighted = HIGHLIGHTED[feature.autoencoder.path] || []

  return (
    <div className="flex flex-col" style={{padding: '40px 60px', maxWidth: 1100}}>
      <h1 className="text-2xl font-bold mb-2">Fact-Checking Abstention SAE Viewer</h1>
      <p className="mb-2">
        Features of <b>domain TopK sparse autoencoders</b> trained on the residual stream of instruction-tuned LLMs while they
        read <b>fact-checking verdict prompts</b> (claim + evidence + instructions), to study <b>abstention</b>{" "}
        ("NOT ENOUGH EVIDENCE"). Statistics are computed at the <b>decision position</b> over {index ? index.meta.n_decisions.toLocaleString() : '…'} claims.
      </p>
      <p className="mb-4 text-sm">
        SAEs: <a className="text-blue-500" href="https://huggingface.co/lucasfrag/fact-checking-abstention-saes">lucasfrag/fact-checking-abstention-saes</a> ·
        data: <a className="text-blue-500" href="https://huggingface.co/datasets/lucasfrag/fact-checking-abstention-saes-data">lucasfrag/fact-checking-abstention-saes-data</a> ·
        viewer adapted from <a className="text-blue-500" href="https://github.com/openai/sparse_autoencoder/tree/main/sae-viewer">OpenAI's sae-viewer</a> (MIT).
      </p>
      <FeatureSelect init_feature={feature} onFeatureChange={(f: Feature) => setFeature(f)}
        onFeatureSubmit={(f: Feature) => navigate(pathForFeature(f))} show_go={true}/>
      {err ? <div style={{color: 'red'}}>Could not load the feature index: {err}</div> : null}

      {highlighted.length > 0 &&
        <div className="mt-4">
          <h2 className="text-xl font-bold mb-2">Highlighted</h2>
          {NOTES[feature.autoencoder.path]?.highlighted && <p className="text-sm text-gray-700 mb-2" style={{maxWidth: 900}}>{NOTES[feature.autoencoder.path].highlighted}</p>}
          {highlighted.map((h) => (
            <button key={h.atom} onClick={() => go(h.atom)} title={h.description}
              className="text-blue-500 hover:text-blue-700 mr-4">{h.label}</button>
          ))}
        </div>}

      <SelectionMethod selectedPath={feature.autoencoder.path} onSelect={go}/>
      <SelectionScatter path={feature.autoencoder.path} highlighted={highlighted} onSelect={go}/>
      <CandidatesTable path={feature.autoencoder.path} steering={STEERING[feature.autoencoder.path] || {}} onSelect={go}/>
      <LayerSweep/>

      {index &&
        <div className="mt-6">
          <h2 className="text-xl font-bold mb-1">Feature map: link score (AUC) × frequency</h2>
          <p className="text-sm text-gray-700 mb-1" style={{maxWidth: 900}}>
            Each dot is one feature. <b>Only look at the height:</b> the higher the dot, the more the feature is linked to
            NEI claims (evidence not enough); the lower, to SUP/REF claims (evidence enough to decide); dots near the dotted middle line have no link.
            <b> Circled</b> dots are the features we studied (hover for what steering showed). <i>Scroll to zoom, drag to move, double-click to reset.</i> {NOTES[feature.autoencoder.path]?.map || ''}
          </p>
          <FeatureMap index={index} highlighted={highlighted} onSelect={go}/>
        </div>}

      <div className="mt-4 flex flex-row flex-wrap" style={{gap: 40}}>
        {([['auc', 'Highest AUC (NEI vs. SUP/REF)', false], ['auc', 'Lowest AUC (fire more on SUP/REF claims)', true],
           ['nei_top20', 'Most NEI among top-20 activations', false]] as [string, string, boolean][]).map(([key, title, asc]) => (
          <div key={title}>
            <h2 className="text-lg font-bold mb-1">{title}</h2>
            <table className="text-sm">
              <thead><tr><th>feature</th><th>AUC</th><th>NEI top-20</th><th>active</th></tr></thead>
              <tbody>
                {ranked(key as 'auc' | 'nei_top20', 25, asc).map((f) => (
                  <tr key={f.atom}>
                    <td><button className="text-blue-500 hover:text-blue-700" onClick={() => go(f.atom)}>#{f.atom}</button></td>
                    <td className="px-2">{f.auc?.toFixed(3)}</td>
                    <td className="px-2">{f.nei_top20 !== null ? `${(100 * (f.nei_top20 as number)).toFixed(0)}%` : ''}</td>
                    <td className="px-2">{(100 * f.frac_active).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </div>
    </div>
  )
}
