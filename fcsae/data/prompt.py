"""Template do prompt de veredito.

O ponto-chave do projeto: renderizamos todos os datasets NESTE template para que a
POSIÇÃO DA DECISÃO (o último token, onde o modelo emitiria o veredito) fique
representada na distribuição de treino do SAE — que é exatamente onde os SAEs
oficiais reconstroem mal. As instruções ficam no fim, deixando o veredito por último.
"""
from __future__ import annotations

from typing import List

SYSTEM_PROMPT = (
    "You are a fact-checking assistant.\n\n"
    "Respond in English.\n\n"
    "General rules:\n- Be factual and precise\n- Do not hallucinate\n"
    "- Follow the instructions strictly"
)

INSTRUCTIONS = (
    "Instructions:\n"
    "- Determine whether the evidence SUPPORTS or REFUTES the claim.\n"
    "- If the evidence confirms the claim, it is SUPPORTED.\n"
    "- If the evidence contradicts the claim, it is REFUTED.\n"
    "- If the evidence is insufficient or unclear, it is NOT ENOUGH EVIDENCE.\n"
    "Return ONLY one label: SUPPORTED, REFUTED, or NOT ENOUGH EVIDENCE."
)


def build_verdict_prompt(claim: str, evidence: List[str], max_evidence: int = 4) -> str:
    ev = evidence[:max_evidence]
    ev_block = ("\n".join(f"[{i+1}] {e}" for i, e in enumerate(ev))
                if ev else "(no evidence retrieved)")
    return (f"Task: Determine the verdict of a claim based on evidence.\n\n"
            f"Claim:\n{claim}\n\n"
            f"Evidence:\n{ev_block}\n\n"
            f"{INSTRUCTIONS}")


# Modelos de raciocínio (DeepSeek-R1-Distill): o generation prompt termina em "<think>\n"; fechar o bloco vazio
# faz o modelo escrever o veredito direto, e o último token do prompt volta a ser a posição da decisão.
EMPTY_THINK = "\n</think>\n\n"


def apply_empty_think(text: str, empty_think: bool) -> str:
    return text + EMPTY_THINK if empty_think and text.endswith("<think>\n") else text


def templated_input_ids(tok, prompt: str, device, max_seq_len: int = 512, empty_think: bool = False):
    """Aplica o chat template (com generation prompt), tokeniza e trunca. Retorna
    (input_ids [1,seq], decision_index) — decision_index é o último token (a posição
    da decisão)."""
    try:
        text = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=False,
            enable_thinking=False,   # Qwen3: veredito direto (ignorado por Llama/Gemma/DeepSeek)
        )
    except TypeError:                # templates que não aceitam o argumento
        text = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=False,
        )
    text = apply_empty_think(text, empty_think)
    ids = tok(text, add_special_tokens=False, return_tensors="pt",
              truncation=True, max_length=max_seq_len).input_ids.to(device)
    return ids, ids.shape[1] - 1
