# Hinge Line

O projeto Hinge Line desenvolve um método objetivo para separar campo regional de residual e mesclar dados gravimétricos heterogêneos (terrestres, marinhos e de satélite) na transição entre continente e oceano, onde esse tipo de análise historicamente mais falha. O método se sustenta em critério estatístico, não em escolha visual. Ele é testado num caso concreto: verificar se as zonas de fratura de Florianópolis e Rio de Janeiro, mapeadas no Atlântico Sul oceânico, de fato continuam como estruturas reais sob a margem continental brasileira, na Bacia de Santos. A questão foi levantada e deixada em aberto em Carvalho et al. (2022).

Dossiê completo do projeto (escopo, revista-alvo, referências, checklist): https://claude.ai/artifact/F9rq6D3L8qPhnZrCXqFPe9

## Objetivo

A contribuição central é metodológica: um critério objetivo, não visual e não por tentativa, para separar campo regional de residual e para mesclar fontes de dados gravimétricos heterogêneas na faixa onde elas se encontram, perto da linha de costa.

O estudo de caso que demonstra o método: as zonas de fratura de Florianópolis e Rio de Janeiro continuam, como estruturas reais, para dentro do continente? Ou a continuidade traçada em Carvalho et al. (2022) é interpretação visual não sustentada com rigor pelos dados?

## Revista-alvo

Annals of Geophysics (INGV): Diamond Open Access, gratuita, Scopus Q2 em Geophysics.

## Área de estudo

AOI provisória: 18°S–30°S, 36°W–52°W (em `src/config.py`). Ainda falta ajustá-la contra as Figuras 8/9 de Carvalho et al. (2022).

## Dados em uso

| Fonte | Tipo | Módulo |
|---|---|---|
| Sandwell & Smith V33.1 | satélite, grade de 1′ (~1,85 km) | `src/acquisition/sandwell.py` |
| NCEI / MGD77 (25 cruzeiros com gravimetria) | marinho | `src/acquisition/marine.py` |
| SGB-CPRM, estações até 2022 (2.816 na AOI) | terrestre | `src/acquisition/terrestrial.py` |
| RGFB, Rede Gravimétrica Fundamental Brasileira (124 estações) | terrestre | `src/processing/terrestrial_merge.py` |
| GEBCO 2026 | batimetria/topografia; máscara terra/mar | `src/acquisition/gebco.py` |
| EMAG2 v3 | magnetometria | `src/acquisition/emag2.py` |

Aguardando: BNDG/ANP (pedido feito) e BDGON (portal fora do ar).

## Estrutura do repositório

```
hinge-line/
├── data/
│   ├── raw/              # dados brutos (não versionados, exceto o CSV da RGFB)
│   └── processed/        # saídas dos módulos (não versionadas)
├── logs/                 # diário do projeto, um arquivo por sessão
├── notebooks/            # notebooks de estudo, passo a passo
├── src/
│   ├── config.py         # AOI e caminhos
│   ├── acquisition/      # download e leitura de cada fonte
│   └── processing/       # QC, crossovers, fusão, regional-residual
├── PROGRESS.md           # aponta para o log mais recente
└── requirements.txt
```

## Pipeline de processamento

Todos os comandos rodam a partir da raiz do projeto:

```
python -m src.acquisition.marine build           # pontos marinhos na AOI + QC por cruzeiro
python -m src.processing.marine_crossover        # crossovers e correção de offset por cruzeiro
python -m src.processing.terrestrial_merge       # CPRM + RGFB, ar-livre recalculado em GRS80
python -m src.processing.fusion                  # grade fundida (Sandwell + correção)
python -m src.processing.fusion --compare        # validação cruzada das políticas de dados
python -m src.processing.regional_residual --input <arquivo>   # grau regional por validação cruzada
```

## Notebooks

- `01_residuo_sandwell.ipynb`: onde as medidas reais discordam do Sandwell, e por que terra e mar se comportam diferente.

## Situação atual

Ver o log mais recente em `logs/` (índice em `PROGRESS.md`).
