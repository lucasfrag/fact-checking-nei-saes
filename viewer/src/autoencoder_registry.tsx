import { Feature } from './types';

// Domain SAEs trained on fact-checking verdict prompts (lucasfrag/fact-checking-nei-saes).
// To add a model: append it to MODELS (folder name in the dataset repo, layer, label).
// `feature` = the model's candidate NEI feature (opened by default when the model is selected).
export const MODELS: {[key: string]: {layer: number, label: string, hf: string, feature?: number}} = {
  'llama-3.1-8b-instruct': {layer: 15, label: 'Llama-3.1-8B-Instruct · layer 15', hf: 'meta-llama/Llama-3.1-8B-Instruct', feature: 19219},
  // 'deepseek-r1-distill-llama-8b': {layer: 15, label: 'DeepSeek-R1-Distill-Llama-8B · layer 15', hf: 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'},   // removido 06/10/2026: DeepSeek sendo refeito do zero
  'qwen3-8b': {layer: 22, label: 'Qwen3-8B · layer 22', hf: 'Qwen/Qwen3-8B', feature: 12482},
  'gemma-3-4b-it': {layer: 19, label: 'Gemma-3-4B-it · layer 19', hf: 'google/gemma-3-4b-it', feature: 12903},
  // 'gemma-4-e4b-it': {layer: ?, label: 'Gemma-4-E4B-it · layer ?', hf: 'google/gemma-4-E4B-it'},
  // 'ministral-3-8b-instruct': {layer: ?, label: 'Ministral-3-8B-Instruct · layer ?', hf: 'mistralai/Ministral-3-8B-Instruct-2512'},
};

// Features with steering results (viewer/<path>/steering/<atom>.json).
export const STEERED: {[path: string]: number[]} = {
  'llama-3.1-8b-instruct/layer-15': [19219],
  'qwen3-8b/layer-22': [12482],
  'gemma-3-4b-it/layer-19': [12903],
  // 'deepseek-r1-distill-llama-8b/layer-15': [18658, 32642, 25349, 19789, 7531],   // sem evidência (FAQ 22)
};

export const FC_FAMILY = {
  subject: 'fact-checking',
  name: 'fc_topk_32k',
  label: 'TopK SAE, n=32768, k=32 (Gao et al., 2024), decision position of verdict prompts',
  selectors: [
    {key: 'model', label: 'Model', values: Object.keys(MODELS)},
  ],
  default_H: (H: {[key: string]: string}) => ({
    model: H.model || Object.keys(MODELS)[0],
  }),
  get_ae: (H: {[key: string]: string}) => {
    const model = H.model || Object.keys(MODELS)[0];
    return {
      subject: 'fact-checking',
      family: 'fc_topk_32k',
      H: {model},
      path: `${model}/layer-${MODELS[model].layer}`,
      num_features: 32768,
    }
  },
}

export const DEFAULT_AUTOENCODER = FC_FAMILY.get_ae(FC_FAMILY.default_H({}));

export const AUTOENCODER_FAMILIES = Object.fromEntries(
  [FC_FAMILY].map((family) => [family.name, family])
);

export const SUBJECT_MODELS = ['fact-checking'];

export function pathForFeature(feature: Feature) {
  let res = `/model/${feature.autoencoder.subject}/family/${feature.autoencoder.family}`;
  for (const selector of AUTOENCODER_FAMILIES[feature.autoencoder.family].selectors) {
    res += `/${selector.key}/${feature.autoencoder.H[selector.key]}`;
  }
  res += `/feature/${feature.atom}`;
  return res
}
