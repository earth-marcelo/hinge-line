# Hinge Line

O projeto Hinge Line desenvolve um método objetivo — sustentado por critério estatístico, não por escolha visual — para separar campo regional de residual e mesclar dados gravimétricos heterogêneos (terrestres, marinhos e de satélite) na transição entre continente e oceano, onde esse tipo de análise historicamente mais falha. O método é testado num caso concreto: verificar se as zonas de fratura de Florianópolis e Rio de Janeiro, mapeadas no Atlântico Sul oceânico, de fato continuam como estruturas reais sob a margem continental brasileira, na Bacia de Santos — uma questão levantada e deixada em aberto em Carvalho et al. (2022).

Dossiê completo do projeto (escopo, revista-alvo, referências, checklist): https://claude.ai/artifact/F9rq6D3L8qPhnZrCXqFPe9

## Objetivo

A contribuição central é metodológica: um critério objetivo — não visual, não por tentativa — para separar campo regional de residual e para mesclar fontes de dados gravimétricos heterogêneas na faixa onde elas se encontram, perto da linha de costa.

O estudo de caso que demonstra o método: as zonas de fratura de Florianópolis e Rio de Janeiro continuam, como estruturas reais, para dentro do continente — ou a continuidade traçada em Carvalho et al. (2022) é interpretação visual não sustentada com rigor pelos dados?

## Revista-alvo

Annals of Geophysics (INGV) — Diamond Open Access, gratuita, Scopus Q2 em Geophysics.

## Estrutura do repositório

```
hinge-line/
├── data/         # dados brutos e processados (não versionar dados grandes)
├── notebooks/    # exploração e prototipagem
├── src/          # pipeline de processamento
└── requirements.txt
```

## Dados públicos identificados

- Rede Gravimétrica Fundamental Brasileira (IBGE) — terrestre, cobertura a verificar
- Sandwell & Smith / Jason-1 — gravimetria via satélite (~1850 m)
- EMAG2 — magnetometria (~3700 m)
- GEBCO — batimetria global (~460 m)

## Próximos passos

Ver checklist completo no dossiê do projeto (link acima).
