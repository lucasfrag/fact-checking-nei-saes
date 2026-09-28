"""fcsae — treino de Sparse Autoencoders de domínio (fact-checking) para estudar a
abstenção em modelos de linguagem.

Pipeline: dados (datasets de fact-checking unificados) -> harvest (ativações do
residual stream cacheadas em disco) -> treino do SAE (JumpReLU) -> avaliação
(FVU, inclusive na posição da decisão) -> análise da abstenção na base do SAE.
"""
__version__ = "0.1.0"
