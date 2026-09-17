"""Aquisição e recorte do EMAG2v3 (magnetometria).

Ao contrário do Sandwell e do GEBCO, o EMAG2v3 (NCEI/NOAA) não tem uma API
de extração por bounding box: é um único arquivo global.

    Fonte : https://www.ngdc.noaa.gov/geomag/data/EMAG2/EMAG2_V3_20170530.zip
    Tamanho: ~4,5 GB (zip) contendo um CSV também grande — baixe uma vez e
             reaproveite; não rebaixe a cada execução.

Colunas do CSV (confirmadas em 2026-09-17 no EMAG2_readme.txt), sem
cabeçalho:
    1. i        — índice de coluna da grade / longitude
    2. j        — índice de linha da grade / latitude
    3. LON      — longitude geográfica WGS84 (graus decimais)
    4. LAT      — latitude geográfica WGS84 (graus decimais)
    5. SeaLevel — anomalia magnética ao nível do mar (nT)
    6. UpCont   — anomalia magnética continuada para 4 km de altitude (nT)
    7. Code     — código da fonte do dado (ver tabela no readme)
    8. Error    — erro estimado (nT); -888/-999 marcam células ambíguas/sem dado

Este módulo assume que o .zip já foi baixado manualmente (ou via
``download_global``, que pode levar bastante tempo dependendo da conexão) e
faz o recorte por bounding box lendo o CSV em blocos (``chunksize``), sem
carregar os 4,5 GB inteiros na memória.
"""

from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.config import AOI, RAW_DIR

logger = logging.getLogger(__name__)

GLOBAL_ZIP_URL = "https://www.ngdc.noaa.gov/geomag/data/EMAG2/EMAG2_V3_20170530.zip"

COLUMN_NAMES = ["i", "j", "lon", "lat", "sealevel_nt", "upcont_nt", "code", "error_nt"]


def download_global(out_path: Path | None = None, timeout: int = 1800) -> Path:
    """Baixa o .zip global do EMAG2v3 (~4,5 GB). Pode demorar bastante.

    Prefira baixar manualmente com um gerenciador que suporte retomada
    (``wget -c``, ``aria2c``) se a conexão for instável; esta função é só
    uma opção simples de streaming com ``requests``.
    """
    out_path = out_path or (RAW_DIR / "emag2" / "EMAG2_V3_20170530.zip")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Baixando EMAG2v3 global (~4,5 GB) de %s", GLOBAL_ZIP_URL)
    with requests.get(GLOBAL_ZIP_URL, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    logger.info("Salvo %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path


def _find_csv_in_zip(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            raise FileNotFoundError(f"nenhum .csv encontrado dentro de {zip_path}")
        return csvs[0]


def _normalize_lon_180(lon: pd.Series) -> pd.Series:
    """Converte longitude 0..360 (ou já -180..180) para -180..180."""
    return ((lon + 180) % 360) - 180


def clip_to_aoi(
    zip_path: Path,
    south: float,
    north: float,
    west: float,
    east: float,
    out_path: Path | None = None,
    chunksize: int = 2_000_000,
) -> Path:
    """Lê o CSV global (dentro do zip) em blocos e salva só os pontos da AOI."""
    csv_name = _find_csv_in_zip(zip_path)
    out_path = out_path or (RAW_DIR / "emag2" / "emag2v3_santos.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_kept = 0
    n_read = 0
    first_chunk = True
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(csv_name) as f:
            reader = pd.read_csv(f, header=None, names=COLUMN_NAMES, chunksize=chunksize)
            for chunk in reader:
                n_read += len(chunk)
                lon_180 = _normalize_lon_180(chunk["lon"])
                mask = (chunk["lat"] >= south) & (chunk["lat"] <= north) & (lon_180 >= west) & (lon_180 <= east)
                subset = chunk.loc[mask].copy()
                subset["lon"] = lon_180.loc[mask]
                if not subset.empty:
                    subset.to_csv(out_path, mode="w" if first_chunk else "a", header=first_chunk, index=False)
                    first_chunk = False
                    n_kept += len(subset)
                logger.info("Lidas %d linhas (%d mantidas até agora)...", n_read, n_kept)

    if first_chunk:
        raise ValueError("nenhum ponto do EMAG2 caiu dentro da AOI informada — confira os limites")
    logger.info("Recorte salvo em %s (%d pontos)", out_path, n_kept)
    return out_path


def download_and_clip_aoi(
    zip_path: Path | None = None,
    aoi: dict | None = None,
    out_path: Path | None = None,
) -> Path:
    """Conveniência: baixa o global (se necessário) e recorta para a AOI padrão."""
    aoi = aoi or AOI
    zip_path = zip_path or (RAW_DIR / "emag2" / "EMAG2_V3_20170530.zip")
    if not zip_path.exists():
        download_global(zip_path)
    return clip_to_aoi(zip_path, south=aoi["south"], north=aoi["north"], west=aoi["west"], east=aoi["east"], out_path=out_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip-path", type=Path, default=RAW_DIR / "emag2" / "EMAG2_V3_20170530.zip")
    p.add_argument("--south", type=float, default=AOI["south"])
    p.add_argument("--north", type=float, default=AOI["north"])
    p.add_argument("--west", type=float, default=AOI["west"])
    p.add_argument("--east", type=float, default=AOI["east"])
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--download", action="store_true", help="baixar o zip global antes, se ainda não existir")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_arg_parser().parse_args()
    if args.download and not args.zip_path.exists():
        download_global(args.zip_path)
    clip_to_aoi(args.zip_path, south=args.south, north=args.north, west=args.west, east=args.east, out_path=args.out)


if __name__ == "__main__":
    main()
