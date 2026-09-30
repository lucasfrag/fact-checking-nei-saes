import React, { useEffect, useState } from "react"
import { useNavigate } from "react-router-dom"
import { Feature, FeatureIndex, FeatureIndexEntry } from "./types"
import FeatureSelect from "./components/featureSelect"
import FeatureMap from "./components/featureMap"
import { pathForFeature, DEFAULT_AUTOENCODER } from "./autoencoder_registry"
import { get_index } from "./interpAPI"

// Features highlighted in the accompanying study (per model folder).
const HIGHLIGHTED: {[path: string]: {atom: number, label: string, description: string}[]} = {
  'llama-3.1-8b-instruct/layer-15': [
    {atom: 19219, label: '19219 — abstention', description: 'candidate abstention latent: most aligned with the NEI-vs-decided direction; its magnitude grows when evidence is insufficient'},
    {atom: 14953, label: '14953 — lowest AUC', description: 'the mirror case: fires more strongly when the evidence is enough to decide (SUP/REF)'},
  ],
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
          {highlighted.map((h) => (
            <button key={h.atom} onClick={() => go(h.atom)} title={h.description}
              className="text-blue-500 hover:text-blue-700 mr-4">{h.label}</button>
          ))}
        </div>}

      {index &&
        <div className="mt-6">
          <h2 className="text-xl font-bold mb-1">Feature map</h2>
          <p className="text-sm text-gray-700 mb-1" style={{maxWidth: 900}}>
            Each point is one of the {index.features.length.toLocaleString()} latents that fire at the verdict position
            (the other {(index.meta.num_features - index.features.length).toLocaleString()} fire only on other tokens of the prompt).
            <b> Horizontal:</b> how often it fires (log scale). <b>Vertical:</b> how well its activation separates NEI from
            SUP/REF claims (AUC; the dotted line at 0.5 is chance): points <b>above</b> fire more when the evidence is
            <i> not enough</i>, points <b>below</b> fire more when the claim can be decided. <b>Color:</b> share of NEI claims
            among the verdicts where it fires (red = above the {(100 * index.meta.nei_base).toFixed(0)}% base rate).
            Most latents are rare and sit near 0.5; the abstention latent stands alone at the top right. Click a point to open it.
          </p>
          <FeatureMap index={index} highlighted={highlighted} onSelect={go}/>
        </div>}

      <div className="mt-4 flex flex-row flex-wrap" style={{gap: 40}}>
        {([['auc', 'Highest AUC (NEI vs. SUP/REF)', false], ['auc', 'Lowest AUC (fire when the claim can be decided)', true],
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
