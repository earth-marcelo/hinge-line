"""Extração de batimetria GEBCO por bounding box.

Fonte: GEBCO Grid Subsetting App (download.gebco.net) — grade global GEBCO
mais recente (GEBCO_2026, 15 arco-segundos, ~460 m). A API por trás do app
não é documentada publicamente; foi reconstruída em 2026-09-17 inspecionando
as chamadas de rede feitas pelo próprio app ao submeter uma extração real:

    1. POST https://download.gebco.net/api/queue
       corpo JSON com os itens da "cesta" (bounding box, grid_id,
       data_source_ids, formats) — retorna um ``basketId``.
    2. GET  https://download.gebco.net/api/queue/status/<basketId>
       repetir até ``{"status": "finished"}`` (processamento é rápido para
       áreas pequenas — a AOI do projeto processou em poucos segundos).
    3. GET  https://download.gebco.net/api/queue/download/<basketId>
       retorna um .zip contendo o grid no formato pedido (netCDF por padrão).

``grid_id=1`` é a grade "GEBCO 2026 Global"; ``data_source_ids=[1]`` é a
camada de batimetria "Bathymetry" (existem também sub-ice e TID — ver
``GRIDS_INFO_URL``). ``formats=[1]`` é netCDF (ver ``FORMATS_INFO_URL``).
Como essa API não é oficialmente documentada, ela pode mudar sem aviso —
se parar de funcionar, os endpoints ``/api/grids`` e ``/api/formats`` ainda
devem responder e ajudam a diagnosticar o que mudou.

Um endereço de e-mail é exigido pelo formulário do site (some para
notificação, não é validado por confirmação), mas não é bloqueante para uso
via API — a extração acontece e o arquivo é servido de qualquer forma.
"""

from __future__ import annotations

import argparse
import logging
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.config import AOI, RAW_DIR

logger = logging.getLogger(__name__)

BASE_URL = "https://download.gebco.net"
GRIDS_INFO_URL = f"{BASE_URL}/api/grids"
FORMATS_INFO_URL = f"{BASE_URL}/api/formats"

GRID_ID_GEBCO_2026_GLOBAL = 1
DATA_SOURCE_BATHYMETRY = 1
FORMAT_NETCDF = 1


def submit_basket(
    south: float,
    north: float,
    west: float,
    east: float,
    email: str,
    grid_id: int = GRID_ID_GEBCO_2026_GLOBAL,
    data_source_ids: list[int] | None = None,
    format_ids: list[int] | None = None,
    timeout: int = 60,
) -> str:
    """Envia a requisição de extração e retorna o ``basketId``."""
    payload = {
        "id": "0",
        "email": email,
        "submission_date": datetime.now(timezone.utc).isoformat(),
        "processing_status": "new",
        "items": [
            {
                "id": 0,
                "grid_id": grid_id,
                "data_source_ids": data_source_ids or [DATA_SOURCE_BATHYMETRY],
                "formats": format_ids or [FORMAT_NETCDF],
                "left": west,
                "right": east,
                "top": north,
                "bottom": south,
            }
        ],
    }
    resp = requests.post(f"{BASE_URL}/api/queue", json=payload, timeout=timeout)
    resp.raise_for_status()
    basket_id = resp.json()["basketId"]
    logger.info("Basket GEBCO submetida: %s", basket_id)
    return basket_id


def wait_for_basket(basket_id: str, poll_interval: int = 5, max_wait: int = 600) -> None:
    """Espera até o status da basket virar "finished"."""
    elapsed = 0
    while elapsed <= max_wait:
        resp = requests.get(f"{BASE_URL}/api/queue/status/{basket_id}", timeout=30)
        resp.raise_for_status()
        status = resp.json().get("status")
        logger.info("Status da basket %s: %s", basket_id, status)
        if status == "finished":
            return
        if status == "error":
            raise RuntimeError(f"basket {basket_id} terminou com erro")
        time.sleep(poll_interval)
        elapsed += poll_interval
    raise TimeoutError(f"basket {basket_id} não terminou em {max_wait}s")


def download_basket(
    basket_id: str,
    out_dir: Path,
    extract: bool = True,
    timeout: int = 300,
    max_retries: int = 4,
) -> Path:
    """Baixa o .zip da basket pronta; se extract=True, também descompacta.

    A conexão com o servidor do GEBCO já se mostrou instável em downloads
    de ~20 MB (``IncompleteRead`` no meio da transferência) mesmo com o
    processamento já ``finished`` — por isso há retry aqui, e o zip baixado
    é validado (``testzip``) antes de ser aceito como completo.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{basket_id}.zip"

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Baixando basket %s (tentativa %d/%d)", basket_id, attempt, max_retries)
            resp = requests.get(f"{BASE_URL}/api/queue/download/{basket_id}", timeout=timeout, stream=True)
            resp.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            with zipfile.ZipFile(zip_path) as zf:
                bad_file = zf.testzip()
                if bad_file is not None:
                    raise zipfile.BadZipFile(f"arquivo corrompido dentro do zip: {bad_file}")
            logger.info("Salvo %s (%.1f MB)", zip_path, zip_path.stat().st_size / 1e6)
            if extract:
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(out_dir)
                logger.info("Extraído em %s", out_dir)
            return zip_path
        except (requests.RequestException, zipfile.BadZipFile) as exc:
            last_exc = exc
            logger.warning("Falha na tentativa %d: %s", attempt, exc)
            zip_path.unlink(missing_ok=True)
            if attempt < max_retries:
                time.sleep(5 * attempt)
    raise RuntimeError(f"não foi possível baixar a basket {basket_id} após {max_retries} tentativas") from last_exc


def download_aoi(email: str, aoi: dict | None = None, out_dir: Path | None = None) -> Path:
    """Baixa a área de interesse padrão do projeto (ver src/config.py::AOI)."""
    aoi = aoi or AOI
    out_dir = out_dir or (RAW_DIR / "gebco")
    basket_id = submit_basket(south=aoi["south"], north=aoi["north"], west=aoi["west"], east=aoi["east"], email=email)
    wait_for_basket(basket_id)
    return download_basket(basket_id, out_dir)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--south", type=float, default=AOI["south"])
    p.add_argument("--north", type=float, default=AOI["north"])
    p.add_argument("--west", type=float, default=AOI["west"])
    p.add_argument("--east", type=float, default=AOI["east"])
    p.add_argument("--email", required=True, help="e-mail exigido pelo formulário do GEBCO (não confirmado)")
    p.add_argument("--out-dir", type=Path, default=None)
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_arg_parser().parse_args()
    out_dir = args.out_dir or (RAW_DIR / "gebco")
    basket_id = submit_basket(south=args.south, north=args.north, west=args.west, east=args.east, email=args.email)
    wait_for_basket(basket_id)
    download_basket(basket_id, out_dir)


if __name__ == "__main__":
    main()
