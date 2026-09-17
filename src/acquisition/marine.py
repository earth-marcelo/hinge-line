"""Descoberta e pedido de dados gravimétricos marinhos (levantamentos com navio).

Fonte: Marine Trackline Geophysical Database (NCEI/NOAA) — cruzeiros
oceanográficos reais (Vema, Robert D. Conrad, Oceanographer etc.), diferente
do Sandwell/Smith (que é gravimetria derivada de satélite/altimetria, não
levantamento com navio).

Duas etapas, reconstruídas em 2026-09-17 inspecionando o app oficial
(https://www.ncei.noaa.gov/maps/trackline-geophysics/ e
https://www.ngdc.noaa.gov/trackline/request/):

1. **Descobrir quais levantamentos existem na área** — instantâneo, sem fila,
   via o serviço ArcGIS que alimenta o mapa:
       GET https://gis.ngdc.noaa.gov/arcgis/rest/services/web_mercator/
           trackline_combined_dynamic/MapServer/2/query
       (layer 2 = "Marine Trackline Surveys: Gravity")
   Testado ao vivo para a AOI do projeto: 28 levantamentos encontrados
   (Vema, Oceanographer, Kurchatov, Robert D. Conrad, entre outros,
   1961–1970s+).

2. **Pedir os dados de fato** — isso entra numa fila de processamento da
   NCEI, não é uma extração sob demanda como o Sandwell/GEBCO:
       POST https://www.ngdc.noaa.gov/next-web/rest/orders
   Um teste real feito nesta sessão (levantamento único V1712, pela UI)
   voltou com ``positionInQueue: 10332`` mas foi processado e entregue por
   e-mail em poucos minutos — a posição na fila parece não refletir tempo
   de espera real para pedidos pequenos. Ainda assim, isso é assíncrono:
   não espere resposta imediata como no Sandwell/GEBCO.

   Os parâmetros exatos do item foram confirmados no e-mail de confirmação
   recebido para esse teste (não são só suposição a partir do JS):
       dataset: "Trackline"
       format: "M77T"            (não "MGD77T" — esse é só o nome da
                                   categoria, o valor do formato é M77T)
       categories: "Trackline:MGD77T"
       includeAncillary: true
       surveyIds: "V1712"
       useNGDCId: false
       useFileBundler: true
       zParam / geometry: vazios quando se pede por ``surveyIds`` explícitos

   A resposta traz um ``id`` de pedido e uma ``deliveryUrl`` (formato
   ``.../next-web/rest/orders/<id>/pickup``) que fica disponível por 3 dias
   depois de pronta; o aviso chega por e-mail. O arquivo entregue é um
   ``.tar.gz`` contendo o(s) arquivo(s) MGD77T — precisa ser extraído
   (``tar -xzvf``).

   Não chame ``submit_order`` sem necessidade — cada chamada é um pedido
   real na fila deles e dispara um e-mail de verdade.
"""

from __future__ import annotations

import argparse
import json
import logging
import tarfile
from pathlib import Path

import pandas as pd
import requests

from src.config import AOI, RAW_DIR

logger = logging.getLogger(__name__)

SURVEYS_MAPSERVER_URL = (
    "https://gis.ngdc.noaa.gov/arcgis/rest/services/web_mercator/"
    "trackline_combined_dynamic/MapServer/2/query"
)
ORDER_URL = "https://www.ngdc.noaa.gov/next-web/rest/orders"

# Colunas de interesse do .m77t (formato completo documentado em mgd77.pdf,
# incluído em todo pedido). GRA_OBS raramente vem preenchido nos cruzeiros
# mais antigos (ex.: V1712) — FREEAIR (anomalia ar-livre, mGal) é o campo de
# gravimetria que de fato importa para a fusão com Sandwell/terrestre.
M77T_KEEP_COLUMNS = ["SURVEY_ID", "DATE", "TIME", "LAT", "LON", "GRA_OBS", "EOTVOS", "FREEAIR", "GRA_QUALCO"]


def list_surveys_in_aoi(
    south: float,
    north: float,
    west: float,
    east: float,
    timeout: int = 30,
) -> list[dict]:
    """Lista os levantamentos de gravimetria marinha que cruzam a AOI.

    Rápido e sem fila — é só uma consulta espacial ao serviço de mapa.
    Retorna uma lista de dicts com (pelo menos) SURVEY_ID, PLATFORM,
    START_YR.
    """
    params = {
        "geometry": f"{west},{south},{east},{north}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "SURVEY_ID,PLATFORM,START_YR",
        "returnGeometry": "false",
        "f": "json",
    }
    resp = requests.get(SURVEYS_MAPSERVER_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    surveys = [f["attributes"] for f in data.get("features", [])]
    logger.info("%d levantamentos de gravimetria marinha encontrados na AOI", len(surveys))
    return surveys


def submit_order(
    survey_ids: list[str],
    email: str,
    dataset: str = "Trackline",
    fmt: str = "M77T",
    categories: str = "Trackline:MGD77T",
    include_ancillary: bool = True,
    use_ngdc_id: bool = False,
    use_file_bundler: bool = True,
    timeout: int = 60,
) -> dict:
    """Envia o pedido de dados à fila da NCEI. Retorna a resposta (com id/deliveryUrl).

    Valores-padrão dos parâmetros conferem com um pedido real que funcionou
    nesta sessão (ver docstring do módulo) — evite mudar sem necessidade.

    Atenção: isso é assíncrono (aparece um ``positionInQueue`` na resposta,
    mas na prática processou em minutos num teste com 1 levantamento). Não
    repita chamadas "só para testar" — cada uma é um pedido real na fila
    deles e dispara um e-mail de verdade.
    """
    payload = {
        "email": email,
        "items": [
            {
                "dataset": dataset,
                "format": fmt,
                "categories": categories,
                "includeAncillary": include_ancillary,
                "surveyIds": ",".join(survey_ids),
                "useNGDCId": use_ngdc_id,
                "useFileBundler": use_file_bundler,
                "zParam": "",
                "geometry": "",
            }
        ],
    }
    resp = requests.post(ORDER_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    result = resp.json()
    logger.info(
        "Pedido submetido: id=%s, posição na fila=%s, entrega em %s",
        result.get("id"),
        result.get("positionInQueue"),
        result.get("deliveryUrl"),
    )
    return result


def check_order_status(order_id: int, timeout: int = 30) -> dict:
    """Consulta o status de um pedido já submetido pelo seu id."""
    resp = requests.get(f"{ORDER_URL}/{order_id}", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def extract_order(tar_gz_path: Path, out_dir: Path | None = None) -> Path:
    """Extrai o .tar.gz entregue por e-mail (contém .m77t/.h77t por levantamento)."""
    out_dir = out_dir or tar_gz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_gz_path, "r:gz") as tf:
        tf.extractall(out_dir)
    logger.info("Extraído em %s", out_dir)
    return out_dir


def load_m77t(path: Path) -> pd.DataFrame:
    """Lê um único arquivo .m77t (tab-delimited, com cabeçalho) como DataFrame.

    Formato confirmado em 2026-09-17 com um arquivo real (v1712.m77t):
    colunas incluem SURVEY_ID, DATE, TIME, LAT, LON, GRA_OBS, EOTVOS,
    FREEAIR, GRA_QUALCO, entre outras (batimetria, magnetometria) — ver
    mgd77.pdf, entregue junto no mesmo pedido, para a lista completa.
    """
    df = pd.read_csv(path, sep="\t")
    keep = [c for c in M77T_KEEP_COLUMNS if c in df.columns]
    return df[keep]


def load_order_gravity(order_dir: Path, only_with_freeair: bool = True) -> pd.DataFrame:
    """Concatena todos os .m77t sob ``order_dir`` (procura recursivamente).

    Útil depois de ``extract_order``: junta todos os levantamentos baixados
    num único DataFrame de gravimetria marinha, pronto para recorte/fusão.
    Por padrão descarta linhas sem FREEAIR (a maior parte das linhas de
    navegação não tem leitura de gravímetro simultânea).
    """
    files = sorted(Path(order_dir).rglob("*.m77t"))
    if not files:
        raise FileNotFoundError(f"nenhum .m77t encontrado em {order_dir}")
    dfs = [load_m77t(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    if only_with_freeair and "FREEAIR" in combined.columns:
        combined = combined[combined["FREEAIR"].notna()]
    logger.info("%d pontos de gravimetria marinha carregados de %d arquivo(s)", len(combined), len(files))
    return combined


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="lista levantamentos na AOI (sem fila)")
    p_list.add_argument("--south", type=float, default=AOI["south"])
    p_list.add_argument("--north", type=float, default=AOI["north"])
    p_list.add_argument("--west", type=float, default=AOI["west"])
    p_list.add_argument("--east", type=float, default=AOI["east"])
    p_list.add_argument("--out", type=Path, default=None, help="salvar lista em JSON")

    p_order = sub.add_parser("order", help="submete pedido à fila da NCEI (assíncrono)")
    p_order.add_argument("--survey-ids", required=True, help="IDs separados por vírgula")
    p_order.add_argument("--email", required=True)

    p_status = sub.add_parser("status", help="consulta status de um pedido")
    p_status.add_argument("--order-id", type=int, required=True)

    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_arg_parser().parse_args()
    if args.command == "list":
        surveys = list_surveys_in_aoi(south=args.south, north=args.north, west=args.west, east=args.east)
        for s in surveys:
            print(s)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(surveys, indent=2, ensure_ascii=False))
            logger.info("Lista salva em %s", args.out)
    elif args.command == "order":
        submit_order(survey_ids=args.survey_ids.split(","), email=args.email)
    elif args.command == "status":
        print(check_order_status(args.order_id))


if __name__ == "__main__":
    main()
