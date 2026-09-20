"""Conjunto terrestre unificado: SGB-CPRM + RGFB (Rede Gravimétrica Fundamental).

Política adotada para a RGFB (decisão de 2026-09-19): **somada** ao conjunto
terrestre, não usada como calibração. Motivos, medidos nos dados:

* Nos marcos em que CPRM e RGFB coincidem de fato (< 0,5 km e mesma altitude)
  a diferença de gravidade é ≤ 0,07 mGal → a CPRM já está no datum absoluto da
  RGFB, não há offset a calibrar. O shapefile da CPRM tem uma única "Line"
  (``LGRAV_CPRM``), sem identificador de campanha, então também não haveria o
  que calibrar por campanha.
* A RGFB só tem valor como dado novo: das 124 estações, 96 não têm nenhuma
  estação CPRM em 20 km e 11 estão ao sul de 26,9°S, onde a CPRM não cobre
  (Torres, Osório, Estrela, Itajaí...).

Cuidados embutidos:

* **Fórmula de gravidade normal única.** A ``An_Ar_Livr`` publicada pela CPRM
  usa GRS67 (bate com o recálculo a 0,01 mGal), enquanto o Sandwell usa GRS80;
  a diferença é ≈ +0,8 mGal. Aqui a anomalia ar-livre é *recalculada* de
  gravidade observada + altitude com GRS80 para todas as fontes
  (``FA_GRS80``); a ``An_Ar_Livr`` publicada não é usada.
* **QC da RGFB.** 20 das 124 linhas do CSV têm ar-livre publicado incoerente
  com gravidade + altitude (|dif| > 1,5 mGal) e há gravidades idênticas em
  estações de lugares diferentes (ex.: Belo Horizonte "D" = Rio de Janeiro
  "E"; Vinhedo "B" = Rio de Janeiro "C"). Origem não determinada (arquivo-fonte
  ou parser). As linhas assim são classificadas em ``qc``:
      ``ok``        |dif| ≤ 1,5 mGal
      ``warn``      1,5 < |dif| ≤ 5 (mantida; sigma inflado)
      ``reject``    |dif| > 5 mGal, ou gravidade repetida em outro local
      ``dup_cprm``  a menos de 0,5 km de uma estação CPRM (some da fusão; a
                    CPRM fica, a RGFB só serve de checagem)
  Das 8 ``reject``, pelo menos 4 (Curitiba "A", Formiga "B", Engenheiro
  Passos, São José do Rio Preto "B") têm o ar-livre *recalculado* coerente com
  o Sandwell (±15 mGal), o que sugere erro só no campo publicado — recuperáveis
  se o .txt original for conferido. Enquanto isso ficam de fora.
* Duplicatas com a CPRM não entram duas vezes.

Uso:
    python -m src.processing.terrestrial_merge
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.acquisition.terrestrial import clip_to_aoi, load_cprm_stations
from src.config import AOI, PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)

RGFB_CSV = RAW_DIR / "rgfb" / "RGFB_estacoes_AOI_Bacia_Santos.csv"
OUT_CSV = PROCESSED_DIR / "terrestrial_merged.csv"
RGFB_QC_CSV = PROCESSED_DIR / "rgfb_qc.csv"

EARTH_RADIUS_KM = 6371.0
FREE_AIR_GRADIENT = 0.3086      # mGal/m
DUP_CPRM_KM = 0.5               # RGFB a menos disso de uma CPRM = duplicata
FA_WARN = 1.5                   # mGal, incoerência tolerada vs ar-livre publicado
FA_REJECT = 5.0
CPRM_SIGMA_DEFAULT = np.nan     # a CPRM não publica incerteza por estação


def gamma_grs80(lat_deg: np.ndarray) -> np.ndarray:
    """Gravidade normal GRS80 (Somigliana, forma fechada), mGal, no elipsoide."""
    s2 = np.sin(np.radians(lat_deg)) ** 2
    return 978032.67715 * (1 + 0.001931851353 * s2) / np.sqrt(1 - 0.0066943800229 * s2)


def gamma_grs67(lat_deg: np.ndarray) -> np.ndarray:
    """Gravidade normal GRS67 (usada na ``An_Ar_Livr`` publicada da CPRM)."""
    lat = np.radians(lat_deg)
    return 978031.846 * (1 + 0.0053024 * np.sin(lat) ** 2 - 0.0000059 * np.sin(2 * lat) ** 2)


def free_air(gravity: np.ndarray, lat_deg: np.ndarray, alt_m: np.ndarray, normal=gamma_grs80) -> np.ndarray:
    return gravity - normal(lat_deg) + FREE_AIR_GRADIENT * alt_m


def _xy_km(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.c_[EARTH_RADIUS_KM * lon * np.cos(lat), EARTH_RADIUS_KM * lat]


def load_cprm(aoi: dict | None = None) -> pd.DataFrame:
    """Estações CPRM na AOI, com ar-livre recalculado em GRS80."""
    aoi = aoi or AOI
    c = clip_to_aoi(load_cprm_stations(), aoi["south"], aoi["north"], aoi["west"], aoi["east"])
    out = pd.DataFrame({
        "source": "CPRM", "station": np.arange(len(c)).astype(str),
        "lat": c["Latitude"].to_numpy(), "lon": c["Longitude"].to_numpy(),
        "alt_m": c["Alt_Ortome"].to_numpy(), "gravity_mgal": c["Gravidade"].to_numpy(),
    })
    out["sigma_mgal"] = CPRM_SIGMA_DEFAULT
    out["fa_grs80"] = free_air(out["gravity_mgal"].to_numpy(), out["lat"].to_numpy(), out["alt_m"].to_numpy())
    out["qc"] = "ok"
    return out


def qc_rgfb(rgfb: pd.DataFrame, cprm: pd.DataFrame) -> pd.DataFrame:
    """Classifica cada estação RGFB (colunas ``qc``, ``nn_cprm_km``, ...)."""
    r = rgfb.copy()
    r["fa_grs67"] = free_air(r["gravity_mgal"].to_numpy(), r["lat"].to_numpy(), r["alt_m"].to_numpy(), gamma_grs67)
    r["fa_grs80"] = free_air(r["gravity_mgal"].to_numpy(), r["lat"].to_numpy(), r["alt_m"].to_numpy())
    r["dif_fa_publicado"] = r["fa_grs67"] - r["ar_livre"]

    dist, _ = cKDTree(_xy_km(cprm["lat"].to_numpy(), cprm["lon"].to_numpy())).query(
        _xy_km(r["lat"].to_numpy(), r["lon"].to_numpy()))
    r["nn_cprm_km"] = dist

    # gravidade idêntica em estações a mais de 1 km uma da outra = incoerente
    xy = _xy_km(r["lat"].to_numpy(), r["lon"].to_numpy())
    r["dup_gravity_elsewhere"] = False
    for _, idx in r.groupby("gravity_mgal").groups.items():
        idx = list(idx)
        if len(idx) > 1:
            d = np.hypot(*(xy[idx][:, None, :] - xy[idx][None, :, :]).transpose(2, 0, 1))
            far = (d > 1.0).any(axis=1)
            r.loc[np.array(idx)[far], "dup_gravity_elsewhere"] = True

    absd = r["dif_fa_publicado"].abs()
    qc = np.where(absd <= FA_WARN, "ok", np.where(absd <= FA_REJECT, "warn", "reject"))
    qc = np.where(r["dup_gravity_elsewhere"] & (absd > FA_WARN), "reject", qc)
    qc = np.where(r["nn_cprm_km"] <= DUP_CPRM_KM, "dup_cprm", qc)
    r["qc"] = qc
    return r


def load_rgfb(cprm: pd.DataFrame, csv: Path = RGFB_CSV) -> pd.DataFrame:
    return qc_rgfb(pd.read_csv(csv), cprm)


def build_terrestrial(rgfb_policy: str = "add", csv: Path = RGFB_CSV,
                      aoi: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Conjunto terrestre pronto para a fusão.

    ``rgfb_policy``: ``"add"`` (padrão) soma as RGFB ``ok`` e ``warn``;
    ``"none"`` devolve só a CPRM (para sensibilidade). Retorna
    ``(terrestre, tabela_qc_rgfb)``.
    """
    if rgfb_policy not in ("add", "none"):
        raise ValueError("rgfb_policy deve ser 'add' ou 'none'")
    cprm = load_cprm(aoi)
    rgfb = load_rgfb(cprm, csv)
    parts = [cprm]
    if rgfb_policy == "add":
        use = rgfb[rgfb["qc"].isin(["ok", "warn"])]
        parts.append(pd.DataFrame({
            "source": "RGFB", "station": use["estacao"].to_numpy(),
            "lat": use["lat"].to_numpy(), "lon": use["lon"].to_numpy(),
            "alt_m": use["alt_m"].to_numpy(), "gravity_mgal": use["gravity_mgal"].to_numpy(),
            # sigma inflado nas "warn": a incoerência de ar-livre pode ser altitude
            "sigma_mgal": np.maximum(use["erro_mgal"], np.where(use["qc"] == "warn",
                                     use["dif_fa_publicado"].abs(), 0.0)),
            "fa_grs80": use["fa_grs80"].to_numpy(), "qc": use["qc"].to_numpy(),
        }))
    return pd.concat(parts, ignore_index=True), rgfb


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    terr, rgfb = build_terrestrial("add")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    terr.to_csv(OUT_CSV, index=False)
    rgfb.to_csv(RGFB_QC_CSV, index=False)
    print("QC da RGFB:", rgfb["qc"].value_counts().to_dict())
    print(terr.groupby("source").size().to_string())
    print(f"-> {OUT_CSV}\n-> {RGFB_QC_CSV}")


if __name__ == "__main__":
    main()
