"""Aquisição de dados públicos para o projeto Hinge Line.

Cada módulo aqui cobre uma fonte de dados:
    sandwell.py    — gravimetria por satélite (Sandwell & Smith V33.1), via
                     extração por bounding box no topex.ucsd.edu. PRONTO.
    gebco.py       — batimetria global GEBCO, via GEBCO Grid Subsetting App
                     (download.gebco.net). PRONTO.
    emag2.py       — magnetometria EMAG2v3 (NCEI/NOAA); download único
                     global + recorte local por bounding box (sem API de
                     extração). PRONTO.
    marine.py      — gravimetria marinha real (levantamentos com navio —
                     Vema, Robert D. Conrad etc.), via Marine Trackline
                     Geophysical Database (NCEI/NOAA): descoberta
                     instantânea de levantamentos na área + pedido
                     assíncrono (fila, entrega por e-mail) dos dados
                     MGD77T. PRONTO — 28 levantamentos identificados na
                     AOI, 1 já baixado e testado, pedido dos outros 27 em
                     andamento.
    terrestrial.py — gravimetria terrestre, estações do SGB-CPRM (achado em
                     2026-09-17, shapefile nacional único baixado
                     manualmente do RIGEO — sem API/recorte no servidor).
                     PRONTO — 20.407 estações no Brasil, 2.816 na AOI do
                     projeto. Resolve a pendência que antes dependia do
                     BDGON (IBGE/ON, portal instável) ou do BNDG (ANP,
                     pedido por formulário) — esses dois seguem como
                     pendências à parte, mas não bloqueiam mais o pipeline.
"""
