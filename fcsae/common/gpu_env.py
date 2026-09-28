"""Correção de ambiente para o bitsandbytes nesta máquina (WSL2 + cu13).

O bitsandbytes não acha `libnvJitLink.so.13` (do wheel nvidia-nvjitlink) pelo caminho
de busca normal e cai num stub quebrado — a conversão 4-bit falha com um erro
aparentemente não relacionado. Pré-carregar a lib com RTLD_GLOBAL resolve.
"""
import ctypes
import importlib.util
import os


def ensure_bitsandbytes_cuda_libs():
    spec = importlib.util.find_spec("nvidia.cu13")
    if not spec or not spec.submodule_search_locations:
        return
    lib_path = os.path.join(list(spec.submodule_search_locations)[0], "lib", "libnvJitLink.so.13")
    if os.path.exists(lib_path):
        ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
