"""Fusão de gravimetria heterogênea (satélite + marinho + terrestre) — esqueleto.

Esquema "remove–calcula–restaura" com o Sandwell V33.1 como campo de fundo:

1. **Remove.** Em cada observação (navio ou estação terrestre) calcula-se o
   resíduo ``r = FA_obs − Sandwell`` (Sandwell interpolado bilinearmente).
2. **Calcula.** O resíduo é interpolado na grade do Sandwell por colocação por
   mínimos quadrados (LSC, equivalente a krigagem simples de média zero) com
   ruído próprio por fonte. Longe dos dados a correção volta a zero e o
   produto final volta a ser o Sandwell — nenhuma fonte "inventa" campo onde
   não mede.
3. **Restaura.** ``fundido = Sandwell + correção``, com o desvio-padrão da
   predição (``err``) em cada nó.

Tudo que é parâmetro sai dos dados, não de escolha visual:

* **Covariância do sinal** (quanto o dado local difere do Sandwell e a que
  distância isso é correlacionado) — ajustada no semivariograma empírico dos
  resíduos, **separado por domínio** (terra/mar). Em terra o resíduo tem
  variância ~15× maior que no mar (o Sandwell em terra é essencialmente
  EGM2008), então um modelo estacionário único é ruim. Usa-se covariância não
  estacionária de Paciorek com Matérn 3/2 (= Markov de 2ª ordem): amplitude e
  comprimento de correlação dependem do domínio de cada ponto (máscara terra/mar
  do GEBCO na grade; a fonte, nas observações). É positiva-definida por
  construção e se reduz à estacionária quando os dois pontos são do mesmo
  domínio.
* **Ruído por fonte.**
    - marinho ``ok``: σ = σ_crossover/√2 (o σ robusto dos crossovers é de uma
      diferença entre dois navios);
    - marinho corrigido por crossover: soma em quadratura a incerteza do
      deslocamento, σ_crossover/√n_xover;
    - CPRM (não publica incerteza): **σ = √pepita** do semivariograma de terra.
      A pepita mistura erro de medida, de altitude (0,3086 mGal/m) e sinal
      abaixo de ~1 km — é exatamente o que a interpolação não consegue
      reproduzir, então é o ruído certo para os pesos;
    - RGFB: ``max(σ publicado/inflado, σ_CPRM)`` (o erro de medida publicado,
      ~0,02 mGal, não inclui altitude nem microsinal).
* **Decimação.** Mediana por fonte/levantamento em células de 1′ antes de tudo.
  Resolve o M14 (52.913 pontos a ~60 m → ~2.400 células) sem jogar dado fora.
  No marinho o σ da célula não diminui com o número de pontos (o erro ao longo
  da trilha é correlacionado); no terrestre, σ/√n.
* **Fórmula de gravidade normal.** Terrestre já vem em GRS80
  (``terrestrial_merge``). No marinho, para cada levantamento com ``GRA_OBS``
  identifica-se a fórmula que reproduz o ``FREEAIR`` publicado (GRS80, GRS67
  ou Internacional 1930) e converte-se para GRS80. Sem ``GRA_OBS`` a fórmula não
  é identificável e o FREEAIR fica como está (coluna ``normal``).

Validação cruzada em blocos espaciais (``cross_validate``): as observações são
agrupadas em blocos de ``cv_block_deg`` graus, cada bloco é previsto sem ele
mesmo, e o RMSE de ``obs − previsto`` é comparado ao do Sandwell sozinho. É o
critério para comparar as políticas (marinho ``ok`` vs ``ok+corrected``, RGFB
``add`` vs ``none``) sobre um **mesmo** conjunto de teste.

Uso:
    python -m src.processing.fusion                    # baseline: ok + add
    python -m src.processing.fusion --compare          # 4 combinações, CV
    python -m src.processing.fusion --marine-policy ok+corrected --rgfb-policy none

Limitações conhecidas (a atacar depois):
* Ruído independente por célula: o viés comum de um cruzeiro inteiro não é
  modelado (a correção por crossover trata só os três ``offset``).
* A transição terra/mar da covariância é abrupta na linha de costa do GEBCO —
  é justamente a faixa que o artigo quer tratar; hoje é um degrau.
* Projeção plana local (x = R·lon·cos lat); suficiente para distâncias ≤ 100 km.
* No mar, o semivariograma entre navios diferentes é plano (~11 mGal², sem
  estrutura espacial): o ajuste dá patamar ≈ 0, a correção marinha fica nula e
  ``err`` no mar sai 0. Isso quer dizer "os navios não trazem sinal que o
  Sandwell não tenha", não que o Sandwell seja exato — o erro próprio do
  Sandwell e o erro dos navios não são separáveis só com esses dados.
* LSC local com k = 48 vizinhos: diferença para a solução exata ≤ 0,8 mGal
  (rms 0,05) num teste com 732 estações; k = 96 leva a ≤ 0,05 mGal.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.io import netcdf_file
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

from src.config import PROCESSED_DIR, RAW_DIR
from src.processing.marine_crossover import compute_offsets, select_marine
from src.processing.terrestrial_merge import build_terrestrial, gamma_grs67, gamma_grs80

logger = logging.getLogger(__name__)

SANDWELL_XYZ = RAW_DIR / "sandwell" / "sandwell_v33.1_gravity_custom.xyz"
GEBCO_NC = RAW_DIR / "gebco" / "gebco_2026_n-18.0_s-30.0_w-52.0_e-36.0.nc"

EARTH_RADIUS_KM = 6371.0
CELL_ARCMIN = 1.0             # decimação: células de 1′ (= espaçamento do Sandwell)
K_NEIGHBORS = 48              # vizinhos por nó na LSC local
MAX_RADIUS_KM = 60.0          # além disso a correção é zero
VARIO_BINS_KM = np.array([0, 1, 2, 4, 6, 10, 15, 20, 30, 45, 60, 80, 100], float)
CV_BLOCK_DEG = 0.25
CV_FOLDS = 5
MARINE_POLICIES = ("ok", "ok+corrected")
RGFB_POLICIES = ("add", "none")


# --------------------------------------------------------------------------
# Grades de referência
# --------------------------------------------------------------------------

@dataclass
class Grid:
    lats: np.ndarray          # crescente (Mercator: espaçamento não uniforme)
    lons: np.ndarray          # crescente, -180..180
    values: np.ndarray        # (nlat, nlon)

    def interpolator(self) -> RegularGridInterpolator:
        # fill_value=None extrapola linearmente os pontos a < meia célula da borda
        return RegularGridInterpolator((self.lats, self.lons), self.values,
                                       bounds_error=False, fill_value=None)


def load_sandwell(path: Path = SANDWELL_XYZ) -> Grid:
    """Grade do Sandwell (xyz sem cabeçalho, lon 0–360) como matriz lat × lon."""
    s = pd.read_csv(path, sep=r"\s+", header=None, names=["lon", "lat", "v"])
    s["lon"] = ((s["lon"] + 180) % 360) - 180
    tab = s.pivot(index="lat", columns="lon", values="v").sort_index().sort_index(axis=1)
    if tab.isna().any().any():
        raise ValueError("grade do Sandwell incompleta")
    return Grid(tab.index.to_numpy(), tab.columns.to_numpy(), tab.to_numpy())


def load_land_mask(lats: np.ndarray, lons: np.ndarray, path: Path = GEBCO_NC) -> np.ndarray:
    """Máscara terra (True) nos nós (lats × lons), GEBCO elevação > 0, vizinho
    mais próximo. O GEBCO é netCDF3 clássico → ``scipy.io`` basta."""
    with netcdf_file(path, "r", mmap=False) as nc:
        glat = nc.variables["lat"][:].copy()
        glon = nc.variables["lon"][:].copy()
        elev = nc.variables["elevation"][:].copy()
    i = np.clip(np.searchsorted(glat, lats), 1, len(glat) - 1)
    i -= (lats - glat[i - 1]) < (glat[i] - lats)
    j = np.clip(np.searchsorted(glon, lons), 1, len(glon) - 1)
    j -= (lons - glon[j - 1]) < (glon[j] - lons)
    return elev[np.ix_(i, j)] > 0


def _xy_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_r = np.radians(lat)
    return np.c_[EARTH_RADIUS_KM * np.radians(lon) * np.cos(lat_r), EARTH_RADIUS_KM * lat_r]


# --------------------------------------------------------------------------
# Observações
# --------------------------------------------------------------------------

def _gamma_int1930(lat_deg: np.ndarray) -> np.ndarray:
    lat = np.radians(lat_deg)
    return 978049.0 * (1 + 0.0052884 * np.sin(lat) ** 2 - 0.0000059 * np.sin(2 * lat) ** 2)


NORMAL_FORMULAS = {"GRS80": gamma_grs80, "GRS67": gamma_grs67, "INT1930": _gamma_int1930}


def identify_normal_formula(marine: pd.DataFrame, tol_mgal: float = 0.3) -> pd.DataFrame:
    """Qual fórmula de gravidade normal reproduz o FREEAIR de cada levantamento.

    Só dá para saber onde há ``GRA_OBS``. Critério: mediana de
    ``|FREEAIR − (GRA_OBS − γ)|`` ≤ ``tol_mgal``. Retorna uma linha por
    levantamento com ``normal`` (nome ou ``"desconhecida"``) e os desvios.
    """
    rows = []
    for survey, g in marine.groupby("SURVEY_ID"):
        g = g.dropna(subset=["GRA_OBS", "FREEAIR"])
        row = {"SURVEY_ID": survey, "n_gra_obs": len(g), "normal": "desconhecida"}
        if len(g):
            dev = {k: float(np.median(np.abs(g["FREEAIR"] - (g["GRA_OBS"] - f(g["LAT"])))))
                   for k, f in NORMAL_FORMULAS.items()}
            row.update({f"dev_{k}": v for k, v in dev.items()})
            best = min(dev, key=dev.get)
            if dev[best] <= tol_mgal:
                row["normal"] = best
        rows.append(row)
    return pd.DataFrame(rows)


def marine_observations(policy: str = "ok") -> tuple[pd.DataFrame, dict]:
    """Pontos marinhos da política escolhida, ar-livre em GRS80 quando possível."""
    offsets, xo = compute_offsets()
    # estimate_offsets devolve só o sigma; recalcula aqui do mesmo jeito (MAD dos
    # resíduos dos crossovers depois do ajuste)
    res = xo["d"] - (xo["s_i"].map(offsets["offset_mgal"]) - xo["s_j"].map(offsets["offset_mgal"]))
    sigma_x = max(1.4826 * float(np.median(np.abs(res - np.median(res)))), 0.5)

    m = select_marine(policy, offsets=offsets)
    normal = identify_normal_formula(m).set_index("SURVEY_ID")["normal"]
    m["normal"] = m["SURVEY_ID"].map(normal)
    fa = m["FREEAIR_CORR"].to_numpy(dtype=float).copy()
    for name, f in NORMAL_FORMULAS.items():
        sel = (m["normal"] == name).to_numpy()
        fa[sel] += f(m.loc[sel, "LAT"].to_numpy()) - gamma_grs80(m.loc[sel, "LAT"].to_numpy())

    sigma = np.full(len(m), sigma_x / np.sqrt(2))
    corr = m["OFFSET_APPLIED"].to_numpy() != 0
    sigma[corr] = np.sqrt(sigma[corr] ** 2 + sigma_x ** 2 / m.loc[corr, "N_XOVER"].to_numpy())
    obs = pd.DataFrame({
        "source": "marine", "survey": m["SURVEY_ID"].to_numpy(),
        "lat": m["LAT"].to_numpy(), "lon": m["LON"].to_numpy(), "fa": fa,
        "sigma": sigma, "normal": m["normal"].to_numpy(),
        "corrected": corr, "offset": m["OFFSET_APPLIED"].to_numpy(),
    })
    info = {"sigma_crossover": sigma_x,
            "normal_by_survey": normal.to_dict()}
    return obs, info


def terrestrial_observations(policy: str = "add") -> pd.DataFrame:
    t, _ = build_terrestrial(policy)
    return pd.DataFrame({
        "source": t["source"].to_numpy(),
        "survey": (t["source"] + ":" + t["station"].astype(str)).to_numpy(),
        "lat": t["lat"].to_numpy(), "lon": t["lon"].to_numpy(), "fa": t["fa_grs80"].to_numpy(),
        "sigma": t["sigma_mgal"].to_numpy(),        # NaN na CPRM: vem da pepita
        "normal": "GRS80", "corrected": False, "offset": 0.0,
    })


def block_reduce(obs: pd.DataFrame, cell_arcmin: float = CELL_ARCMIN) -> pd.DataFrame:
    """Mediana por (fonte, levantamento/estação, célula). No terrestre cada
    estação já é um "levantamento" próprio; o agrupamento por célula junta
    estações da mesma fonte na mesma célula com σ/√n."""
    o = obs.copy()
    o["ci"] = np.floor((o["lat"] + 90) * 60 / cell_arcmin).astype(int)
    o["cj"] = np.floor((o["lon"] + 180) * 60 / cell_arcmin).astype(int)
    land = o["source"] != "marine"
    o["group"] = np.where(land, o["source"], o["survey"])
    b = (o.groupby(["source", "group", "ci", "cj"], sort=False)
         .agg(lat=("lat", "mean"), lon=("lon", "mean"), r=("r", "median"), fa=("fa", "median"),
              sigma=("sigma", "median"), n=("r", "size"), survey=("survey", "first"),
              corrected=("corrected", "any"), offset=("offset", "first"))
         .reset_index())
    b = b.rename(columns={"group": "track"})
    b["domain"] = np.where(b["source"] == "marine", "sea", "land")
    is_land = b["domain"] == "land"
    b.loc[is_land, "sigma"] = b.loc[is_land, "sigma"] / np.sqrt(b.loc[is_land, "n"])
    return b.drop(columns=["ci", "cj"])


def build_observations(marine_policy: str = "ok", rgfb_policy: str = "add",
                       sandwell: Grid | None = None) -> tuple[pd.DataFrame, dict]:
    """Observações brutas (todas as fontes) com resíduo contra o Sandwell."""
    sandwell = sandwell or load_sandwell()
    mar, info = marine_observations(marine_policy)
    obs = pd.concat([mar, terrestrial_observations(rgfb_policy)], ignore_index=True)
    obs["sandwell"] = sandwell.interpolator()(obs[["lat", "lon"]].to_numpy())
    obs["r"] = obs["fa"] - obs["sandwell"]
    return obs, info


# --------------------------------------------------------------------------
# Covariância
# --------------------------------------------------------------------------

def matern32(h: np.ndarray, L: float | np.ndarray) -> np.ndarray:
    """Correlação Matérn ν=3/2 com comprimento L (= Markov de 2ª ordem)."""
    q = np.sqrt(3.0) * h / L
    return (1 + q) * np.exp(-q)


def empirical_variogram(b: pd.DataFrame, bins: np.ndarray = VARIO_BINS_KM,
                        exclude_same_track: bool = True) -> pd.DataFrame:
    """Semivariograma robusto (mediana/0,4549) dos resíduos ``r``.

    ``exclude_same_track``: descarta pares do mesmo cruzeiro (o erro ao longo
    da trilha é correlacionado e faria a pepita parecer ~0).
    """
    xy = _xy_km(b["lat"].to_numpy(), b["lon"].to_numpy())
    r = b["r"].to_numpy() - np.median(b["r"].to_numpy())
    pairs = cKDTree(xy).query_pairs(bins[-1], output_type="ndarray")
    if exclude_same_track:
        tr = b["track"].to_numpy()
        pairs = pairs[(tr[pairs[:, 0]] != tr[pairs[:, 1]]) | (b["domain"].to_numpy()[pairs[:, 0]] == "land")]
    h = np.hypot(*(xy[pairs[:, 0]] - xy[pairs[:, 1]]).T)
    g = 0.5 * (r[pairs[:, 0]] - r[pairs[:, 1]]) ** 2
    k = np.digitize(h, bins) - 1
    rows = []
    for i in range(len(bins) - 1):
        sel = k == i
        rows.append({"h": h[sel].mean() if sel.any() else 0.5 * (bins[i] + bins[i + 1]),
                     "gamma": np.median(g[sel]) / 0.4549 if sel.any() else np.nan,
                     "n": int(sel.sum())})
    return pd.DataFrame(rows)


@dataclass
class DomainCov:
    sill: float       # variância do sinal (mGal²)
    L: float          # km
    nugget: float     # mGal²

    @property
    def amp(self) -> float:
        return float(np.sqrt(self.sill))


def fit_variogram(v: pd.DataFrame, min_pairs: int = 30) -> DomainCov:
    """γ(h) = pepita + patamar·(1 − ρ(h)), mínimos quadrados ponderados
    (peso n/γ², Cressie 1985)."""
    v = v[(v["n"] >= min_pairs) & v["gamma"].notna()]
    h, g, n = v["h"].to_numpy(), v["gamma"].to_numpy(), v["n"].to_numpy()
    model = lambda h, nug, sill, L: nug + sill * (1 - matern32(h, L))  # noqa: E731
    p0 = [max(g[0] * 0.5, 0.1), g.max(), 10.0]
    popt, _ = curve_fit(model, h, g, p0=p0, sigma=g / np.sqrt(n),
                        bounds=([0, 0, 0.5], [np.inf, np.inf, 500]), maxfev=20000)
    return DomainCov(sill=float(popt[1]), L=float(popt[2]), nugget=float(popt[0]))


@dataclass
class CovModel:
    land: DomainCov
    sea: DomainCov
    variograms: dict = field(default_factory=dict)

    def params(self, is_land: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        amp = np.where(is_land, self.land.amp, self.sea.amp)
        L = np.where(is_land, self.land.L, self.sea.L)
        return amp, L

    def cov(self, h, amp_a, L_a, amp_b, L_b) -> np.ndarray:
        """Covariância não estacionária de Paciorek (2D, isotrópica por ponto)."""
        L2 = 0.5 * (L_a ** 2 + L_b ** 2)
        pref = (L_a * L_b) / L2
        return amp_a * amp_b * pref * matern32(h, np.sqrt(L2))

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({d: vars(getattr(self, d)) for d in ("land", "sea")}).T


def fit_covariance(b: pd.DataFrame) -> CovModel:
    """Covariância por domínio a partir dos resíduos decimados."""
    vl = empirical_variogram(b[b["domain"] == "land"])
    vs = empirical_variogram(b[b["domain"] == "sea"])
    cov = CovModel(fit_variogram(vl), fit_variogram(vs), {"land": vl, "sea": vs})
    for name in ("land", "sea"):
        d = getattr(cov, name)
        if d.sill < 0.05 * d.nugget:
            logger.warning("%s: sem sinal estruturado no resíduo (patamar %.2f ≪ pepita %.2f) "
                           "— a correção nesse domínio será ~0 e o produto fica no Sandwell",
                           name, d.sill, d.nugget)
    logger.info("covariância:\n%s", cov.table().round(2).to_string())
    return cov


def assign_noise(b: pd.DataFrame, cov: CovModel) -> pd.DataFrame:
    """σ faltante (CPRM) = √pepita de terra; RGFB com piso na mesma pepita."""
    b = b.copy()
    floor = np.sqrt(cov.land.nugget)
    cprm = b["source"] == "CPRM"
    b.loc[cprm, "sigma"] = floor / np.sqrt(b.loc[cprm, "n"])
    rg = b["source"] == "RGFB"
    b.loc[rg, "sigma"] = np.maximum(b.loc[rg, "sigma"], floor / np.sqrt(b.loc[rg, "n"]))
    if b["sigma"].isna().any():
        raise ValueError("observações sem σ depois de assign_noise")
    return b


# --------------------------------------------------------------------------
# Predição (LSC local)
# --------------------------------------------------------------------------

def lsc_predict(obs: pd.DataFrame, tgt_lat: np.ndarray, tgt_lon: np.ndarray,
                tgt_land: np.ndarray, cov: CovModel, k: int = K_NEIGHBORS,
                max_radius_km: float = MAX_RADIUS_KM, chunk: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Correção prevista (média zero) e desvio-padrão nos alvos.

    Para cada alvo usa os ``k`` vizinhos mais próximos a ≤ ``max_radius_km``.
    Alvos sem vizinho: correção 0 e erro = amplitude do sinal do domínio.
    """
    # observações de domínio sem sinal (patamar 0) têm peso nulo: fora da
    # busca de vizinhos, para não ocuparem as k vagas de quem informa
    o_amp, _ = cov.params((obs["domain"] == "land").to_numpy())
    obs = obs[o_amp > 0]
    oxy = _xy_km(obs["lat"].to_numpy(), obs["lon"].to_numpy())
    o_amp, o_L = cov.params((obs["domain"] == "land").to_numpy())
    o_r = obs["r"].to_numpy()
    o_nv = obs["sigma"].to_numpy() ** 2
    txy = _xy_km(tgt_lat, tgt_lon)
    t_amp, t_L = cov.params(tgt_land)

    n_obs = len(obs)
    k = min(k, n_obs)
    pred = np.zeros(len(txy))
    err = t_amp.astype(float).copy()
    if n_obs == 0:
        return pred, err
    tree = cKDTree(oxy)
    _, near = tree.query(txy, k=1, distance_upper_bound=max_radius_km)
    todo = np.where(near < n_obs)[0]
    for s in range(0, len(todo), chunk):
        ti = todo[s:s + chunk]
        dist, idx = tree.query(txy[ti], k=k, distance_upper_bound=max_radius_km)
        dist, idx = np.atleast_2d(dist), np.atleast_2d(idx)
        valid = idx < n_obs
        idx = np.where(valid, idx, 0)
        p = oxy[idx]                                            # (B,k,2)
        hh = np.linalg.norm(p[:, :, None, :] - p[:, None, :, :], axis=-1)
        C = cov.cov(hh, o_amp[idx][:, :, None], o_L[idx][:, :, None],
                    o_amp[idx][:, None, :], o_L[idx][:, None, :])
        C += np.eye(k)[None] * o_nv[idx][:, None, :]
        c = cov.cov(np.where(valid, dist, 0.0), o_amp[idx], o_L[idx],
                    t_amp[ti][:, None], t_L[ti][:, None])
        # vizinhos inexistentes: linha/coluna identidade e c = 0 → peso 0
        vv = valid[:, :, None] & valid[:, None, :]
        C = np.where(vv, C, np.eye(k)[None])
        c = np.where(valid, c, 0.0)
        w = np.linalg.solve(C, c[..., None])[..., 0]
        pred[ti] = (w * np.where(valid, o_r[idx], 0.0)).sum(1)
        err[ti] = np.sqrt(np.clip(t_amp[ti] ** 2 - (w * c).sum(1), 0, None))
    return pred, err


# --------------------------------------------------------------------------
# Validação cruzada
# --------------------------------------------------------------------------

def cv_folds(b: pd.DataFrame, block_deg: float = CV_BLOCK_DEG, n_folds: int = CV_FOLDS,
             seed: int = 42) -> np.ndarray:
    """Fold de cada observação: blocos espaciais de ``block_deg`` graus,
    distribuídos em ``n_folds`` grupos por um hash do id do bloco.

    O fold depende **só da posição** (não de quais outras observações existem),
    então treino e teste de políticas diferentes caem nos mesmos folds — sem
    isso a CV entre políticas vaza dado de teste para o treino.
    """
    bi = np.floor((b["lat"].to_numpy() + 90) / block_deg).astype(np.uint64)
    bj = np.floor((b["lon"].to_numpy() + 180) / block_deg).astype(np.uint64)
    x = bi * np.uint64(1_000_003) + bj + np.uint64(seed)
    # splitmix64
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x % np.uint64(n_folds)).astype(int)


def cross_validate(train: pd.DataFrame, test: pd.DataFrame, cov: CovModel,
                   block_deg: float = CV_BLOCK_DEG, n_folds: int = CV_FOLDS) -> pd.DataFrame:
    """Prevê cada observação de ``test`` sem os dados do seu bloco espacial.

    ``test`` pode ser diferente de ``train`` (ex.: o conjunto comum entre
    políticas, ou os cruzeiros corrigidos avaliados contra um treino que não
    os contém). Os folds dependem só da posição, então são iguais para
    qualquer ``train``.
    """
    ftr, fte = cv_folds(train, block_deg, n_folds), cv_folds(test, block_deg, n_folds)
    out = test.copy()
    out["pred"] = np.nan
    for f in range(n_folds):
        tr, te = train[ftr != f], np.where(fte == f)[0]
        pred, _ = lsc_predict(tr, test["lat"].to_numpy()[te], test["lon"].to_numpy()[te],
                              (test["domain"].to_numpy()[te] == "land"), cov)
        out.iloc[te, out.columns.get_loc("pred")] = pred
    out["resid_sandwell"] = out["r"]
    out["resid_fused"] = out["r"] - out["pred"]
    return out


def cv_summary(cv: pd.DataFrame, by: str = "source") -> pd.DataFrame:
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))  # noqa: E731
    g = cv.groupby(by)
    return pd.DataFrame({
        "n": g.size(),
        "rmse_sandwell": g["resid_sandwell"].apply(rms),
        "rmse_fused": g["resid_fused"].apply(rms),
        "mad_fused": g["resid_fused"].apply(lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))),
    }).assign(ganho=lambda d: 1 - d["rmse_fused"] / d["rmse_sandwell"])


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

@dataclass
class FusionResult:
    marine_policy: str
    rgfb_policy: str
    obs: pd.DataFrame           # decimadas, com σ e resíduo
    cov: CovModel
    info: dict
    grid: pd.DataFrame | None = None


def prepare(marine_policy: str = "ok", rgfb_policy: str = "add",
            sandwell: Grid | None = None, cov: CovModel | None = None) -> FusionResult:
    """Observações decimadas + covariância (ajustada, se não for dada)."""
    if marine_policy not in MARINE_POLICIES or rgfb_policy not in RGFB_POLICIES:
        raise ValueError("política inválida")
    obs, info = build_observations(marine_policy, rgfb_policy, sandwell)
    b = block_reduce(obs)
    cov = cov or fit_covariance(b)
    b = assign_noise(b, cov)
    logger.info("%s/%s: %d observações decimadas (%s)", marine_policy, rgfb_policy, len(b),
                b["source"].value_counts().to_dict())
    return FusionResult(marine_policy, rgfb_policy, b, cov, info)


def fuse_grid(res: FusionResult, sandwell: Grid, land: np.ndarray) -> FusionResult:
    lat2, lon2 = np.meshgrid(sandwell.lats, sandwell.lons, indexing="ij")
    corr, err = lsc_predict(res.obs, lat2.ravel(), lon2.ravel(), land.ravel(), res.cov)
    shape = lat2.shape
    res.grid = {"sandwell": sandwell.values, "correction": corr.reshape(shape),
                "fused": sandwell.values + corr.reshape(shape), "err": err.reshape(shape),
                "land": land.astype(np.int8)}
    return res


def write_netcdf(res: FusionResult, sandwell: Grid, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with netcdf_file(path, "w") as nc:
        nc.title = f"Hinge Line - fusao gravimetrica ({res.marine_policy}, RGFB {res.rgfb_policy})"
        nc.createDimension("lat", len(sandwell.lats))
        nc.createDimension("lon", len(sandwell.lons))
        for name, vals, units in (("lat", sandwell.lats, "degrees_north"),
                                  ("lon", sandwell.lons, "degrees_east")):
            v = nc.createVariable(name, "f8", (name,))
            v[:] = vals
            v.units = units
        for name, units in (("fused", "mGal"), ("sandwell", "mGal"), ("correction", "mGal"),
                            ("err", "mGal"), ("land", "1")):
            v = nc.createVariable(name, "i1" if name == "land" else "f4", ("lat", "lon"))
            v[:] = res.grid[name]
            v.units = units
    return path


def compare_policies(sandwell: Grid, cv_block_deg: float = CV_BLOCK_DEG
                     ) -> tuple[pd.DataFrame, pd.DataFrame, CovModel]:
    """CV das 4 combinações sobre um conjunto de teste comum.

    A covariância é ajustada uma vez, no baseline (``ok``/``add``), e usada em
    todas — assim a comparação mede só o efeito dos dados. O conjunto de teste
    comum é o baseline (marinho ``ok`` + CPRM + RGFB); os cruzeiros corrigidos
    são avaliados à parte, previstos a partir do baseline (validação
    independente da correção por crossover).
    """
    base = prepare("ok", "add", sandwell)
    cov = base.cov
    test = base.obs
    rows = []
    for mp in MARINE_POLICIES:
        for rp in RGFB_POLICIES:
            res = base if (mp, rp) == ("ok", "add") else prepare(mp, rp, sandwell, cov)
            cv = cross_validate(res.obs, test, cov, block_deg=cv_block_deg)
            s = cv_summary(cv).reset_index()
            s.insert(0, "rgfb", rp)
            s.insert(0, "marinho", mp)
            rows.append(s)
    table = pd.concat(rows, ignore_index=True)

    corr = prepare("ok+corrected", "add", sandwell, cov).obs
    corr = corr[corr["corrected"]].copy()
    # o baseline não contém esses cruzeiros: previsão direta, sem CV
    pred, _ = lsc_predict(base.obs, corr["lat"].to_numpy(), corr["lon"].to_numpy(),
                          np.zeros(len(corr), bool), cov)
    corr["resid_sandwell"] = corr["r"]                 # já corrigido
    corr["resid_fused"] = corr["r"] - pred             # corrigido − previsto
    corr["resid_uncorr"] = corr["r"] + corr["offset"] - pred
    has = np.isfinite(pred) & (np.abs(pred) > 0)
    cvc = cv_summary(corr[has], by="survey")
    cvc["rmse_sem_correcao"] = corr[has].groupby("survey")["resid_uncorr"].apply(
        lambda x: float(np.sqrt(np.mean(np.square(x)))))
    return table, cvc, cov


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--marine-policy", choices=MARINE_POLICIES, default="ok")
    p.add_argument("--rgfb-policy", choices=RGFB_POLICIES, default="add")
    p.add_argument("--compare", action="store_true", help="CV das 4 combinações de política")
    p.add_argument("--no-grid", action="store_true", help="não calcula a grade (só CV/covariância)")
    p.add_argument("--cv-block", type=float, default=CV_BLOCK_DEG, help="lado do bloco da CV (graus)")
    args = p.parse_args()

    sandwell = load_sandwell()
    pd.set_option("display.width", 160)
    if args.compare:
        table, cvc, cov = compare_policies(sandwell, args.cv_block)
        print("Covariância (ajustada no baseline ok/add):")
        print(cov.table().round(3).to_string())
        print("\nCV em blocos de %.2f° — conjunto de teste comum (baseline):" % args.cv_block)
        print(table.round(3).to_string(index=False))
        print("\nCruzeiros corrigidos por crossover vs. previsão só com o baseline:")
        print(cvc.round(3).to_string())
        out = PROCESSED_DIR / "fusion_cv_policies.csv"
        table.to_csv(out, index=False)
        print(f"-> {out}")
        return

    res = prepare(args.marine_policy, args.rgfb_policy, sandwell)
    print(res.cov.table().round(3).to_string())
    print("σ crossover: %.2f mGal; fórmula normal por levantamento: %s"
          % (res.info["sigma_crossover"], res.info["normal_by_survey"]))
    cv = cross_validate(res.obs, res.obs, res.cov, block_deg=args.cv_block)
    print(cv_summary(cv).round(3).to_string())
    tag = f"{args.marine_policy.replace('+', '_')}_{args.rgfb_policy}"
    res.obs.to_csv(PROCESSED_DIR / f"fusion_obs_{tag}.csv", index=False)
    if not args.no_grid:
        land = load_land_mask(sandwell.lats, sandwell.lons)
        fuse_grid(res, sandwell, land)
        out = write_netcdf(res, sandwell, PROCESSED_DIR / f"fused_{tag}.nc")
        c = res.grid["correction"]
        print("correção: |c|>0 em %.1f%% dos nós; min %.1f, max %.1f mGal"
              % (100 * np.mean(np.abs(c) > 1e-9), c.min(), c.max()))
        print(f"-> {out}")


if __name__ == "__main__":
    main()
