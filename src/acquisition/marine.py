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

import numpy as np
import pandas as pd
import requests

from src.config import AOI, PROCESSED_DIR, RAW_DIR

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

MARINE_DIR = RAW_DIR / "marine"
SANDWELL_XYZ = RAW_DIR / "sandwell" / "sandwell_v33.1_gravity_custom.xyz"
DEFAULT_POINTS_OUT = PROCESSED_DIR / "marine_gravity_aoi.csv"
DEFAULT_QC_OUT = PROCESSED_DIR / "marine_survey_qc.csv"

# Colunas lidas ao processar os arquivos bundled (o de 2013, KN210-04, tem
# 3,9 milhões de linhas — só carregamos o necessário, em blocos).
_BUNDLE_USECOLS = ["SURVEY_ID", "DATE", "TIME", "LAT", "LON", "GRA_OBS", "FREEAIR", "GRA_QUALCO"]


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


# --------------------------------------------------------------------------
# Leitura dos arquivos bundled (vários levantamentos por .m77t) — 2026-09-19
#
# Achados ao inspecionar os 5 lotes reais (ver logs/diário do projeto):
#   * Cada lote vem como UM par MGD77_<n>.m77t/.h77t com vários levantamentos
#     concatenados (coluna SURVEY_ID distingue). O nome do arquivo não diz
#     quais levantamentos há dentro.
#   * Os arquivos guardam a trilha INTEIRA de cada cruzeiro (ex.: KIR1 tem
#     pontos em 13°E, África) — é preciso recortar pela AOI.
#   * FREEAIR = GRA_OBS - gravidade normal (fórmula que varia por cruzeiro;
#     NÃO soma Eötvös — GRA_OBS já vem corrigido). Onde FREEAIR existe, ele
#     é o campo utilizável; GRA_OBS sozinho, sem FREEAIR, não serve direto.
#   * OPR470: sem nenhuma gravimetria. KN210-04 (Knorr, 2013): só GRA_OBS a
#     ~1 Hz, cru (resíduo contra a gravidade normal na ordem de ±10.000
#     mGal, sem Eötvös, sem filtro, sem FREEAIR) — inutilizável sem um
#     pré-processamento próprio; fica de fora por padrão (aparece no
#     inventário com ``n_gra_obs_only``).
# --------------------------------------------------------------------------


def read_h77t(path: Path) -> pd.DataFrame:
    """Lê o cabeçalho (.h77t, tab-delimited) — uma linha por levantamento."""
    return pd.read_csv(path, sep="\t", dtype=str)


def load_bundled_gravity(
    marine_dir: Path = MARINE_DIR,
    aoi: dict | None = None,
    chunksize: int = 500_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lê todos os ``MGD77_*.m77t`` de ``marine_dir``, recorta pela AOI e
    mantém só pontos com FREEAIR.

    Retorna ``(pontos, inventario)``:
      * ``pontos`` — SURVEY_ID, DATETIME (UTC, NaT se ilegível), LAT, LON,
        GRA_OBS, FREEAIR, GRA_QUALCO, SOURCE_FILE. Só linhas dentro da AOI
        com FREEAIR preenchido.
      * ``inventario`` — uma linha por levantamento: n_total (trilha
        inteira), n_in_aoi, n_freeair_in_aoi, n_gra_obs_only (na AOI, tem
        GRA_OBS mas não FREEAIR), source_file e metadados do .h77t.
    """
    aoi = aoi or AOI
    files = sorted(Path(marine_dir).glob("MGD77_*.m77t"))
    if not files:
        raise FileNotFoundError(f"nenhum MGD77_*.m77t em {marine_dir}")

    kept: list[pd.DataFrame] = []
    counts: dict[str, dict] = {}
    for f in files:
        reader = pd.read_csv(
            f,
            sep="\t",
            usecols=lambda c: c in _BUNDLE_USECOLS,
            dtype={"SURVEY_ID": str, "DATE": str, "TIME": str},
            chunksize=chunksize,
            low_memory=False,
        )
        for chunk in reader:
            in_aoi = (
                chunk["LAT"].between(aoi["south"], aoi["north"])
                & chunk["LON"].between(aoi["west"], aoi["east"])
            )
            has_fa = chunk["FREEAIR"].notna()
            has_go = chunk["GRA_OBS"].notna()
            for sid, grp in chunk.groupby("SURVEY_ID", sort=False):
                c = counts.setdefault(
                    sid,
                    dict(SURVEY_ID=sid, source_file=f.name, n_total=0, n_in_aoi=0,
                         n_freeair_in_aoi=0, n_gra_obs_only=0),
                )
                idx = grp.index
                c["n_total"] += len(grp)
                c["n_in_aoi"] += int(in_aoi[idx].sum())
                c["n_freeair_in_aoi"] += int((in_aoi[idx] & has_fa[idx]).sum())
                c["n_gra_obs_only"] += int((in_aoi[idx] & has_go[idx] & ~has_fa[idx]).sum())
            sel = chunk[in_aoi & has_fa].copy()
            if not sel.empty:
                sel["SOURCE_FILE"] = f.name
                kept.append(sel)
        logger.info("%s processado", f.name)

    if kept:
        pts = pd.concat(kept, ignore_index=True)
    else:
        pts = pd.DataFrame(columns=_BUNDLE_USECOLS + ["SOURCE_FILE"])
    # TIME vem como HHMM, às vezes com minuto fracionário (ex.: "1935.333" no M14)
    hhmm = pd.to_numeric(pts["TIME"], errors="coerce")
    minutes = (hhmm // 100) * 60 + (hhmm % 100)
    pts["DATETIME"] = pd.to_datetime(pts["DATE"], format="%Y%m%d", errors="coerce") + pd.to_timedelta(minutes, unit="m")
    pts = pts[["SURVEY_ID", "DATETIME", "LAT", "LON", "GRA_OBS", "FREEAIR", "GRA_QUALCO", "SOURCE_FILE"]]

    inv = pd.DataFrame(counts.values())
    headers = pd.concat([read_h77t(h) for h in sorted(Path(marine_dir).glob("MGD77_*.h77t"))], ignore_index=True)
    meta_cols = [c for c in ["SURVEY_ID", "PLATFORM", "DATE_DEP", "GRAV_INSTR", "GRAV_FORMU", "GRAV_RFSYS"] if c in headers.columns]
    inv = inv.merge(headers[meta_cols].drop_duplicates("SURVEY_ID"), on="SURVEY_ID", how="left")
    logger.info(
        "%d pontos com FREEAIR na AOI, %d levantamentos (%d com gravimetria utilizável)",
        len(pts), len(inv), int((inv["n_freeair_in_aoi"] > 0).sum()),
    )
    return pts, inv


def sandwell_reference(xyz_path: Path = SANDWELL_XYZ):
    """Devolve ``f(lat, lon) -> mGal`` interpolando a grade Sandwell/Smith em disco.

    Serve só como referência independente para checar levantamentos com
    navio (QC); a grade tem ~1' e comprimento de onda mínimo maior que o do
    gravímetro de bordo, então resíduos de alguns mGal são esperados.
    """
    from scipy.interpolate import RegularGridInterpolator

    g = pd.read_csv(xyz_path, sep=r"\s+", names=["lon", "lat", "g"])
    g["lon"] = ((g["lon"] + 180) % 360) - 180  # a API entrega 0-360
    lons = np.sort(g["lon"].unique())
    lats = np.sort(g["lat"].unique())
    grid = g.pivot(index="lat", columns="lon", values="g").reindex(index=lats, columns=lons).values
    interp = RegularGridInterpolator((lats, lons), grid, bounds_error=False, fill_value=np.nan)
    return lambda lat, lon: interp(np.column_stack([np.asarray(lat), np.asarray(lon)]))


def qc_by_survey(
    points: pd.DataFrame,
    inventory: pd.DataFrame,
    reference,
    min_points: int = 50,
    mad_max: float = 5.0,
    offset_max: float = 6.0,
) -> pd.DataFrame:
    """Resumo de QC por levantamento: FREEAIR do navio menos ``reference(lat, lon)``.

    ``qc_status`` (limiares HEURÍSTICOS — ajustar com critério, não são
    padrão da literatura):
      * ``sem_gravimetria`` — nenhum ponto com FREEAIR na AOI;
      * ``poucos_pontos``   — menos de ``min_points`` pontos (estatística frágil);
      * ``ruidoso``         — MAD do resíduo > ``mad_max`` mGal: dado
                              inconsistente com a referência, não só
                              deslocado;
      * ``offset``          — MAD ok mas |mediana| > ``offset_max``: bias
                              constante (típico de amarração de datum),
                              corrigível por levantamento;
      * ``ok``              — caso contrário.
    Atenção à circularidade: se a referência for o Sandwell, ela não deve
    ser usada para "corrigir" o marinho que depois será fundido com ela.
    """
    pts = points.copy()
    pts["_ref"] = reference(pts["LAT"].values, pts["LON"].values)
    pts["_res"] = pts["FREEAIR"] - pts["_ref"]

    def _stats(g: pd.DataFrame) -> pd.Series:
        r = g["_res"].dropna()
        med = r.median()
        return pd.Series(
            dict(
                n_compared=len(r),
                resid_median=med,
                resid_mad=1.4826 * np.median(np.abs(r - med)) if len(r) else np.nan,
                resid_p01=r.quantile(0.01) if len(r) else np.nan,
                resid_p99=r.quantile(0.99) if len(r) else np.nan,
                freeair_absmax=g["FREEAIR"].abs().max(),
            )
        )

    stats = pts.groupby("SURVEY_ID").apply(_stats, include_groups=False).reset_index()
    qc = inventory.merge(stats, on="SURVEY_ID", how="left")

    def _status(row) -> str:
        if not row["n_freeair_in_aoi"]:
            return "sem_gravimetria"
        if row["n_compared"] < min_points:
            return "poucos_pontos"
        if row["resid_mad"] > mad_max:
            return "ruidoso"
        if abs(row["resid_median"]) > offset_max:
            return "offset"
        return "ok"

    qc["qc_status"] = qc.apply(_status, axis=1)
    return qc.sort_values("SURVEY_ID").reset_index(drop=True)


def build_marine_gravity(
    marine_dir: Path = MARINE_DIR,
    aoi: dict | None = None,
    points_out: Path = DEFAULT_POINTS_OUT,
    qc_out: Path = DEFAULT_QC_OUT,
    sandwell_xyz: Path = SANDWELL_XYZ,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pipeline completo: bundles -> recorte AOI -> CSV de pontos + CSV de QC.

    Não descarta nada por QC — os pontos saem todos, com ``qc_status`` no
    CSV de QC (por levantamento) para a fusão decidir o que usar.
    """
    pts, inv = load_bundled_gravity(marine_dir, aoi)
    qc = qc_by_survey(pts, inv, sandwell_reference(sandwell_xyz))
    for out in (points_out, qc_out):
        out.parent.mkdir(parents=True, exist_ok=True)
    pts.to_csv(points_out, index=False)
    qc.to_csv(qc_out, index=False)
    logger.info("Pontos salvos em %s; QC em %s", points_out, qc_out)
    return pts, qc


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

    p_build = sub.add_parser("build", help="lê os .m77t bundled, recorta pela AOI e gera CSV de pontos + QC")
    p_build.add_argument("--marine-dir", type=Path, default=MARINE_DIR)
    p_build.add_argument("--points-out", type=Path, default=DEFAULT_POINTS_OUT)
    p_build.add_argument("--qc-out", type=Path, default=DEFAULT_QC_OUT)

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
    elif args.command == "build":
        _, qc = build_marine_gravity(args.marine_dir, points_out=args.points_out, qc_out=args.qc_out)
        print(qc[["SURVEY_ID", "n_freeair_in_aoi", "resid_median", "resid_mad", "qc_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
