"""Gravimetria terrestre — estações do SGB-CPRM (Serviço Geológico do Brasil).

Fonte encontrada em 2026-09-17 (via notícia do IBRAM sobre a liberação de
dados gravimétricos terrestres pela CPRM): repositório institucional RIGEO
do SGB, item "Estações gravimétricas adquiridas pelo SGB-CPRM até o ano de
2022":
    https://rigeo.sgb.gov.br/items/7f29f00a-fa8c-40d6-830f-fe756f71f9d4
    (download direto: .../bitstreams/d53eb862-7c63-4b23-b737-f5a7de9119d7/download)

Diferente do Sandwell/GEBCO/EMAG2, aqui não há API de extração nem recorte
por bounding box no servidor — é um único shapefile nacional pequeno
(~1,4 MB zipado), baixado manualmente uma vez. Não é preciso repetir o
download a cada execução; o portal do GeoSGB (https://geosgb.sgb.gov.br/,
menu Downloads > Geofísica Terrestre > Gravimetria) deve ter versões mais
recentes no futuro ("Estações Gravimétricas", com a data no nome) — vale
conferir periodicamente se uma atualização compensa.

Conteúdo confirmado do shapefile (`Gravimetria_CPRM_2022_vr1.shp` + .dbf
etc., CRS EPSG:4674 — SIRGAS2000 geográfico):
    20.407 estações em todo o Brasil. Colunas:
        Line        — identificador da linha/campanha (ex.: "LGRAV_CPRM")
        Longitude, Latitude  — graus decimais (SIRGAS2000)
        Alt_Ortome  — altitude ortométrica (m)
        Gravidade   — gravidade observada (mGal)
        An_Ar_Livr  — anomalia ar-livre (mGal)
        An_Bouguer  — anomalia Bouguer (mGal)
    Recortando para a AOI do projeto (N -18/S -30/O -52/L -36): 2.816
    estações — cobertura boa na Bacia de Santos e no entorno das FFZ/RJFZ.

Este módulo espera o shapefile já extraído em disco (baixe e descompacte o
zip manualmente, ou aponte --shapefile para onde ele estiver).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.config import AOI, RAW_DIR

logger = logging.getLogger(__name__)

DEFAULT_SHAPEFILE = RAW_DIR / "terrestrial_cprm" / "Gravimetria_CPRM_2022_vr1.shp"

KEEP_COLUMNS = ["Line", "Longitude", "Latitude", "Alt_Ortome", "Gravidade", "An_Ar_Livr", "An_Bouguer"]


def load_cprm_stations(shapefile_path: Path = DEFAULT_SHAPEFILE) -> gpd.GeoDataFrame:
    """Lê o shapefile nacional de estações gravimétricas do SGB-CPRM."""
    gdf = gpd.read_file(shapefile_path)
    logger.info("%d estações carregadas de %s (CRS %s)", len(gdf), shapefile_path, gdf.crs)
    return gdf


def clip_to_aoi(
    gdf: gpd.GeoDataFrame,
    south: float,
    north: float,
    west: float,
    east: float,
) -> pd.DataFrame:
    """Recorta as estações pela AOI (graus decimais) e retorna um DataFrame simples."""
    mask = (
        (gdf["Latitude"] >= south)
        & (gdf["Latitude"] <= north)
        & (gdf["Longitude"] >= west)
        & (gdf["Longitude"] <= east)
    )
    clipped = gdf.loc[mask, [c for c in KEEP_COLUMNS if c in gdf.columns]].reset_index(drop=True)
    logger.info("%d estações dentro da AOI (de %d no total)", len(clipped), len(gdf))
    return clipped


def clip_aoi_from_file(
    shapefile_path: Path = DEFAULT_SHAPEFILE,
    aoi: dict | None = None,
    out_path: Path | None = None,
) -> Path:
    """Conveniência: lê o shapefile, recorta para a AOI padrão do projeto e salva CSV."""
    aoi = aoi or AOI
    gdf = load_cprm_stations(shapefile_path)
    clipped = clip_to_aoi(gdf, south=aoi["south"], north=aoi["north"], west=aoi["west"], east=aoi["east"])
    out_path = out_path or (RAW_DIR / "terrestrial_cprm" / "gravimetria_cprm_santos.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clipped.to_csv(out_path, index=False)
    logger.info("Recorte salvo em %s", out_path)
    return out_path


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shapefile", type=Path, default=DEFAULT_SHAPEFILE)
    p.add_argument("--south", type=float, default=AOI["south"])
    p.add_argument("--north", type=float, default=AOI["north"])
    p.add_argument("--west", type=float, default=AOI["west"])
    p.add_argument("--east", type=float, default=AOI["east"])
    p.add_argument("--out", type=Path, default=None)
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_arg_parser().parse_args()
    clip_aoi_from_file(
        shapefile_path=args.shapefile,
        aoi={"south": args.south, "north": args.north, "west": args.west, "east": args.east},
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
