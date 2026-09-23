"""Processamento — separação regional-residual e fusão de fontes heterogêneas.

Submódulos:
    regional_residual.py — critério objetivo (validação cruzada) para
                            escolher o grau da superfície de tendência
                            polinomial que separa campo regional de
                            residual. Não depende de fusão nem de dados
                            terrestres — funciona em cima de qualquer fonte
                            isolada (começamos com o Sandwell).
    marine_crossover.py   — crossovers entre cruzeiros marinhos, deslocamento
                            por levantamento (sem usar o Sandwell) e política
                            de uso do marinho ("ok" | "ok+corrected").
    terrestrial_merge.py  — conjunto terrestre CPRM + RGFB (RGFB somada, não
                            calibração), ar-livre recalculado em GRS80 para todas
                            as fontes e QC das estações RGFB.
    fusion.py             — fusão remove–calcula–restaura sobre o Sandwell:
                            resíduos decimados em células de 1′, covariância
                            terra/mar ajustada no semivariograma, ruído por
                            fonte (CPRM = pepita), LSC local e validação
                            cruzada em blocos para comparar as políticas.
"""
