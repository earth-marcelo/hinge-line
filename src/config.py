"""Configuração central do projeto Hinge Line.

Mantém em um só lugar a área de interesse (AOI) e os caminhos de dados, para
que todos os módulos de aquisição/processamento usem a mesma referência.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Caminhos
# --------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# --------------------------------------------------------------------------
# Área de interesse (AOI)
#
# Caixa envolvente provisória cobrindo a Bacia de Santos e as zonas de
# fratura de Florianópolis (FFZ) e Rio de Janeiro (RJFZ), da porção oceânica
# até a margem continental. Longitudes em referência -180/180 (oeste
# negativo). AJUSTAR depois de conferir contra a extensão das Figuras 8/9 de
# Carvalho et al. (2022) — este valor é só um ponto de partida generoso.
# --------------------------------------------------------------------------

AOI = {
    "north": -18.0,   # limite norte (grau, positivo = N)
    "south": -30.0,   # limite sul (grau, negativo = S)
    "west": -52.0,    # limite oeste (grau, -180..180)
    "east": -36.0,    # limite leste (grau, -180..180)
}
