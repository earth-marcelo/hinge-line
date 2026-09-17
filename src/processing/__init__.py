"""Processamento — separação regional-residual e fusão de fontes heterogêneas.

Submódulos:
    regional_residual.py — critério objetivo (validação cruzada) para
                            escolher o grau da superfície de tendência
                            polinomial que separa campo regional de
                            residual. Não depende de fusão nem de dados
                            terrestres — funciona em cima de qualquer fonte
                            isolada (começamos com o Sandwell).
    fusion.py             — (a implementar) fusão estatística de fontes
                            heterogêneas (terrestre + marinho + satélite)
                            perto da margem continental.
"""
