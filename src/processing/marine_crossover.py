"""Crossovers entre levantamentos marinhos e política de uso do dado marinho.

Problema: o QC de ``src/acquisition/marine.py`` compara cada cruzeiro com o
Sandwell e marca alguns como ``offset`` (nível médio deslocado em relação ao
campo de referência: EW9305 +14,6 mGal, 73003121 +19,6, V2413 −8,0). Corrigir
esses cruzeiros *usando o Sandwell* seria circular, já que o marinho depois é
fundido com o Sandwell. Aqui o deslocamento é estimado só a partir dos
**crossovers** — pontos onde as trilhas de dois cruzeiros diferentes se
cruzam — de modo que o Sandwell fica de fora da correção e serve apenas como
validação independente.

Método:

1. Cada levantamento vira uma poligonal (pontos consecutivos, na ordem do
   arquivo). Segmentos maiores que ``MAX_SEG_KM`` são descartados (lacunas
   onde interpolar não faz sentido).
2. Cruzamentos entre segmentos de levantamentos diferentes são achados com
   KD-tree nos pontos médios + interseção exata de segmentos; FREEAIR é
   interpolado linearmente nos dois segmentos. Descartam-se cruzamentos onde
   o gradiente ao longo da trilha passa de ``MAX_GRAD`` mGal/km (a
   interpolação vira a principal fonte de erro) e cruzamentos quase paralelos
   (ângulo < ``MIN_ANGLE``).
3. Ajuste de um deslocamento ``b_s`` por levantamento a partir das diferenças
   ``d = FA_i − FA_j = b_i − b_j + erro``, por mínimos quadrados robustos
   (IRLS/Huber). O nível absoluto não é identificável só com crossovers: o
   "gauge" é fixado com média zero de ``b`` sobre os levantamentos ``ok``.
4. Só entram nas equações levantamentos ``ok`` e ``offset``; ``ruidoso`` e
   ``poucos_pontos`` ficam de fora (ruído não é corrigível por deslocamento e
   contaminaria o ajuste).

Política (função ``select_marine``):
    ``"ok"``            — só levantamentos ``ok``, sem alteração.
    ``"ok+corrected"``  — ``ok`` + levantamentos ``offset`` com FREEAIR
                          corrigido por ``b_s`` (só se tiverem pelo menos
                          ``min_xover`` crossovers; padrão 4).

Uso:
    python -m src.processing.marine_crossover
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.config import PROCESSED_DIR

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0
MAX_SEG_KM = 10.0     # segmento maior que isso = lacuna, não interpola
MAX_GRAD = 1.5        # mGal/km, em ambos os segmentos do cruzamento
MIN_ANGLE = 15.0      # graus; evita cruzamentos quase paralelos
DEFAULT_MIN_XOVER = 4  # crossovers mínimos p/ aceitar a correção de um "offset"
POLICIES = ("ok", "ok+corrected")

MARINE_CSV = PROCESSED_DIR / "marine_gravity_aoi.csv"
QC_CSV = PROCESSED_DIR / "marine_survey_qc.csv"
OFFSETS_CSV = PROCESSED_DIR / "marine_crossover_offsets.csv"
CROSSOVERS_CSV = PROCESSED_DIR / "marine_crossovers.csv"


# --------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------

def load_marine() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lê os pontos marinhos e o QC (saídas de ``marine.py build``)."""
    df = pd.read_csv(MARINE_CSV)
    df["DATETIME"] = pd.to_datetime(df["DATETIME"], format="ISO8601", errors="coerce")
    qc = pd.read_csv(QC_CSV)
    lat, lon = np.radians(df["LAT"].to_numpy()), np.radians(df["LON"].to_numpy())
    # projeção senoidal (equivalente): basta para detectar cruzamentos
    df["x_km"] = EARTH_RADIUS_KM * lon * np.cos(lat)
    df["y_km"] = EARTH_RADIUS_KM * lat
    return df, qc


# --------------------------------------------------------------------------
# Crossovers
# --------------------------------------------------------------------------

def _segments(df: pd.DataFrame, max_seg_km: float) -> pd.DataFrame:
    """Segmentos consecutivos (ordem do arquivo) dentro de cada levantamento."""
    parts = []
    for survey, g in df.groupby("SURVEY_ID", sort=False):
        x, y, f = g["x_km"].to_numpy(), g["y_km"].to_numpy(), g["FREEAIR"].to_numpy()
        length = np.hypot(np.diff(x), np.diff(y))
        i = np.where((length <= max_seg_km) & (length > 0))[0]
        parts.append(pd.DataFrame({
            "survey": survey, "x0": x[i], "y0": y[i], "x1": x[i + 1], "y1": y[i + 1],
            "f0": f[i], "f1": f[i + 1], "length": length[i],
        }))
    return pd.concat(parts, ignore_index=True)


def find_crossovers(df: pd.DataFrame, max_seg_km: float = MAX_SEG_KM) -> pd.DataFrame:
    """Todos os cruzamentos entre levantamentos diferentes, com FREEAIR
    interpolado nos dois lados (sem filtro de gradiente/ângulo)."""
    seg = _segments(df, max_seg_km)
    mid = np.c_[0.5 * (seg.x0 + seg.x1), 0.5 * (seg.y0 + seg.y1)]
    pairs = cKDTree(mid).query_pairs(r=max_seg_km, output_type="ndarray")
    sv = seg["survey"].to_numpy()
    pairs = pairs[sv[pairs[:, 0]] != sv[pairs[:, 1]]]
    a, b = pairs[:, 0], pairs[:, 1]

    p0 = seg[["x0", "y0"]].to_numpy()
    d = seg[["x1", "y1"]].to_numpy() - p0
    p, r, q, s = p0[a], d[a], p0[b], d[b]
    rxs = r[:, 0] * s[:, 1] - r[:, 1] * s[:, 0]
    qp = q - p
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / rxs
        u = (qp[:, 0] * r[:, 1] - qp[:, 1] * r[:, 0]) / rxs
    keep = np.isfinite(t) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1) & (np.abs(rxs) > 1e-9)
    a, b, t, u, p, r = a[keep], b[keep], t[keep], u[keep], p[keep], r[keep]

    f0, f1, length = seg["f0"].to_numpy(), seg["f1"].to_numpy(), seg["length"].to_numpy()
    fa = f0[a] + t * (f1[a] - f0[a])
    fb = f0[b] + u * (f1[b] - f0[b])
    cos_ang = np.abs((d[a] * d[b]).sum(1) / (length[a] * length[b]))
    return pd.DataFrame({
        "s_i": sv[a], "s_j": sv[b], "fa_i": fa, "fa_j": fb, "d": fa - fb,
        "grad_i": (f1[a] - f0[a]) / length[a], "grad_j": (f1[b] - f0[b]) / length[b],
        "angle": np.degrees(np.arccos(np.clip(cos_ang, 0, 1))),
        "x_km": p[:, 0] + t * r[:, 0], "y_km": p[:, 1] + t * r[:, 1],
    })


def filter_crossovers(xo: pd.DataFrame, max_grad: float = MAX_GRAD,
                      min_angle: float = MIN_ANGLE) -> pd.DataFrame:
    keep = (xo["grad_i"].abs() <= max_grad) & (xo["grad_j"].abs() <= max_grad) & (xo["angle"] >= min_angle)
    return xo[keep].reset_index(drop=True)


# --------------------------------------------------------------------------
# Ajuste dos deslocamentos
# --------------------------------------------------------------------------

def estimate_offsets(xo: pd.DataFrame, anchor: set[str], huber_k: float = 3.0,
                     iters: int = 30) -> tuple[pd.Series, float]:
    """Deslocamento ``b_s`` por levantamento (mGal), mínimos quadrados robustos.

    ``anchor``: levantamentos cuja média de ``b`` é fixada em zero (o nível
    absoluto não é identificável só com crossovers). Retorna ``(b, sigma)``
    com ``sigma`` o desvio robusto (MAD) dos resíduos dos crossovers.
    """
    surveys = sorted(set(xo["s_i"]) | set(xo["s_j"]))
    idx = {s: k for k, s in enumerate(surveys)}
    m, n = len(xo), len(surveys)
    design = np.zeros((m, n))
    design[np.arange(m), xo["s_i"].map(idx)] = 1.0
    design[np.arange(m), xo["s_j"].map(idx)] = -1.0
    d = xo["d"].to_numpy()
    anchor_row = np.array([1.0 if s in anchor else 0.0 for s in surveys])
    anchor_row = 1e3 * anchor_row / max(anchor_row.sum(), 1.0)

    w = np.ones(m)
    sigma = 1.0
    for _ in range(iters):
        sw = np.sqrt(w)
        b, *_ = np.linalg.lstsq(np.vstack([design * sw[:, None], anchor_row]),
                                np.r_[d * sw, 0.0], rcond=None)
        res = d - design @ b
        sigma = max(1.4826 * np.median(np.abs(res - np.median(res))), 0.5)
        limit = huber_k * sigma
        w = np.where(np.abs(res) <= limit, 1.0, limit / np.maximum(np.abs(res), 1e-12))
    return pd.Series(b, index=surveys, name="offset_mgal"), float(sigma)


def compute_offsets(min_xover: int = DEFAULT_MIN_XOVER) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pipeline completo: crossovers → deslocamentos por levantamento.

    Retorna ``(tabela_de_deslocamentos, crossovers_usados)``. A tabela traz,
    para cada levantamento, ``offset_mgal``, ``n_xover``, ``qc_status``, a
    mediana de resíduo contra o Sandwell (só para validação) e
    ``sandwell_apos_corr`` = resíduo Sandwell restante depois de aplicar a
    correção (≈0 significa que o crossover reproduziu o offset de forma
    independente).
    """
    df, qc = load_marine()
    status = qc.set_index("SURVEY_ID")["qc_status"]
    ok = set(status[status == "ok"].index)
    usable = ok | set(status[status == "offset"].index)

    xo = filter_crossovers(find_crossovers(df))
    xo = xo[xo["s_i"].isin(usable) & xo["s_j"].isin(usable)].reset_index(drop=True)
    offsets, sigma = estimate_offsets(xo, ok)
    logger.info("%d crossovers usados, sigma robusto %.2f mGal", len(xo), sigma)

    counts = pd.concat([xo["s_i"], xo["s_j"]]).value_counts().rename("n_xover")
    table = (offsets.to_frame()
             .join(counts)
             .join(status)
             .join(qc.set_index("SURVEY_ID")["resid_median"].rename("sandwell_resid_median")))
    table["sandwell_apos_corr"] = table["sandwell_resid_median"] - table["offset_mgal"]
    return table, xo


# --------------------------------------------------------------------------
# Política de uso
# --------------------------------------------------------------------------

def select_marine(policy: str = "ok", min_xover: int = DEFAULT_MIN_XOVER,
                  offsets: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pontos marinhos prontos para a fusão segundo a política escolhida.

    Colunas extras: ``FREEAIR_CORR`` (FREEAIR após correção; igual a FREEAIR
    nos levantamentos ``ok``), ``OFFSET_APPLIED`` (mGal, 0 se não houve) e
    ``N_XOVER`` (crossovers que sustentam a correção; NaN nos ``ok``).
    """
    if policy not in POLICIES:
        raise ValueError(f"policy deve ser uma de {POLICIES}, recebi {policy!r}")
    df, qc = load_marine()
    status = qc.set_index("SURVEY_ID")["qc_status"]
    df["qc_status"] = df["SURVEY_ID"].map(status)
    df["FREEAIR_CORR"] = df["FREEAIR"]
    df["OFFSET_APPLIED"] = 0.0
    df["N_XOVER"] = np.nan

    keep = df["qc_status"] == "ok"
    if policy == "ok+corrected":
        if offsets is None:
            offsets, _ = compute_offsets(min_xover)
        cand = offsets[(offsets["qc_status"] == "offset") & (offsets["n_xover"] >= min_xover)]
        for survey, row in cand.iterrows():
            sel = df["SURVEY_ID"] == survey
            df.loc[sel, "FREEAIR_CORR"] = df.loc[sel, "FREEAIR"] - row["offset_mgal"]
            df.loc[sel, "OFFSET_APPLIED"] = row["offset_mgal"]
            df.loc[sel, "N_XOVER"] = row["n_xover"]
            keep |= sel
    return df[keep].drop(columns=["x_km", "y_km"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--min-xover", type=int, default=DEFAULT_MIN_XOVER)
    args = parser.parse_args()

    table, xo = compute_offsets(args.min_xover)
    table.to_csv(OFFSETS_CSV)
    xo.to_csv(CROSSOVERS_CSV, index=False)
    with pd.option_context("display.width", 160):
        print(table.round(2).sort_values("qc_status").to_string())
    for policy in POLICIES:
        sel = select_marine(policy, args.min_xover, table)
        print(f"{policy}: {sel['SURVEY_ID'].nunique()} levantamentos, {len(sel)} pontos")
    print(f"-> {OFFSETS_CSV}\n-> {CROSSOVERS_CSV}")


if __name__ == "__main__":
    main()
