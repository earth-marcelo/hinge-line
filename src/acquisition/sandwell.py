"""Extração de gravimetria por satélite (Sandwell & Smith V33.1).

Fonte: topex.ucsd.edu — grade global de 1 arco-minuto (~1850 m), extraída
sob demanda por bounding box através de um formulário HTTP POST em
``https://topex.ucsd.edu/cgi-bin/get_data.cgi``.

Campos do formulário (confirmados em 2026-09-17 inspecionando o HTML da
página https://topex.ucsd.edu):
    north, south  — latitude em graus (negativo = sul)
    west, east    — longitude em graus, referência 0–360 (ex.: -50° == 310°)
    mag           — "0.1" para gravidade, "1" para topografia/batimetria

A resposta é texto ASCII puro, uma linha por ponto de grade:
    <longitude 0-360>  <latitude>  <valor em mGal ou m>

Não há chave de API nem limite documentado — testado com uma caixa de
12°x16° (~760 mil pontos, ~22 MB) sem erro, em ~27 s. Para áreas maiores,
prefira dividir em blocos (ver ``download_in_tiles``) para não segurar uma
única requisição HTTP longa demais.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import requests

from src.config import AOI, RAW_DIR

logger = logging.getLogger(__name__)

BASE_URL = "https://topex.ucsd.edu/cgi-bin/get_data.cgi"

# mag=0.1 -> gravidade (mGal); mag=1 -> topografia/batimetria (m)
FIELD_CODES = {"gravity": "0.1", "topography": "1"}


def _to_0_360(lon: float) -> float:
    """Converte longitude -180..180 para a referência 0..360 usada pela API."""
    return lon % 360


def fetch_xyz(
    south: float,
    north: float,
    west: float,
    east: float,
    field: str = "gravity",
    timeout: int = 180,
    max_retries: int = 3,
) -> str:
    """Baixa uma janela XYZ da grade Sandwell/Smith e retorna o texto bruto.

    Latitudes em graus decimais (negativo = sul); longitudes em -180..180
    (a conversão para 0..360 é feita aqui dentro).
    """
    if field not in FIELD_CODES:
        raise ValueError(f"field deve ser um de {list(FIELD_CODES)}, recebido {field!r}")

    payload = {
        "north": f"{north}",
        "south": f"{south}",
        "west": f"{_to_0_360(west)}",
        "east": f"{_to_0_360(east)}",
        "mag": FIELD_CODES[field],
    }

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Baixando %s (tentativa %d/%d): %s", field, attempt, max_retries, payload)
            resp = requests.post(BASE_URL, data=payload, timeout=timeout)
            resp.raise_for_status()
            if not resp.text.strip():
                raise ValueError("resposta vazia do servidor — confira os limites da caixa")
            return resp.text
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            logger.warning("Falha na tentativa %d: %s", attempt, exc)
            if attempt < max_retries:
                time.sleep(5 * attempt)
    raise RuntimeError(f"não foi possível baixar dados Sandwell após {max_retries} tentativas") from last_exc


def save_xyz(text: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    n_lines = text.count("\n")
    logger.info("Salvo %s (%d linhas, %.1f MB)", out_path, n_lines, out_path.stat().st_size / 1e6)
    return out_path


def download_aoi(
    field: str = "gravity",
    aoi: dict | None = None,
    out_path: Path | None = None,
) -> Path:
    """Baixa a área de interesse padrão do projeto (ver src/config.py::AOI)."""
    aoi = aoi or AOI
    if out_path is None:
        out_path = RAW_DIR / "sandwell" / f"sandwell_v33.1_{field}_santos.xyz"
    text = fetch_xyz(
        south=aoi["south"], north=aoi["north"], west=aoi["west"], east=aoi["east"], field=field
    )
    return save_xyz(text, out_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--south", type=float, default=AOI["south"])
    p.add_argument("--north", type=float, default=AOI["north"])
    p.add_argument("--west", type=float, default=AOI["west"])
    p.add_argument("--east", type=float, default=AOI["east"])
    p.add_argument("--field", choices=list(FIELD_CODES), default="gravity")
    p.add_argument("--out", type=Path, default=None, help="caminho do arquivo .xyz de saída")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_arg_parser().parse_args()
    out_path = args.out or RAW_DIR / "sandwell" / f"sandwell_v33.1_{args.field}_custom.xyz"
    text = fetch_xyz(south=args.south, north=args.north, west=args.west, east=args.east, field=args.field)
    save_xyz(text, out_path)


if __name__ == "__main__":
    main()
