"""Separação regional-residual por critério estatístico objetivo.

Este é o núcleo metodológico do projeto: em vez de escolher visualmente o
grau da superfície de tendência que representa o "campo regional" (como foi
feito na dissertação de 2003), aqui o grau é escolhido por **validação
cruzada k-fold** — o grau que minimiza o erro de predição em dados que o
ajuste não viu é o escolhido, não o que minimiza o erro no próprio ajuste
(que sempre cai com mais parâmetros e levaria a "overfitting" o regional).

Método: ajusta uma superfície polinomial 2D de grau ``d`` em (x, y) → z por
mínimos quadrados (``x, y`` tipicamente longitude/latitude ou coordenadas
projetadas, ``z`` o campo gravimétrico). O "campo regional" é o valor
ajustado pela superfície; o "residual" é ``z - regional``. Isso é feito para
cada fonte isoladamente — não depende de fusão nem dos dados terrestres, só
precisa de uma grade/conjunto de pontos.

Uso típico (ver ``main`` / CLI): carregar os pontos do Sandwell (ou de
qualquer outra fonte), rodar ``choose_best_degree`` para ver a curva de erro
por grau, e ``separate`` para obter os campos regional e residual com o
grau escolhido.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


def _normalize(coord: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Centraliza e escala uma coordenada para evitar mal-condicionamento
    numérico em graus polinomiais altos (comum com lon/lat em graus)."""
    mean = coord.mean()
    scale = coord.std() or 1.0
    return (coord - mean) / scale, mean, scale


def design_matrix(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Monta a matriz de projeto de um polinômio 2D completo até ``degree``.

    Ex.: degree=2 -> termos 1, x, y, x^2, xy, y^2.
    """
    terms = []
    for total in range(degree + 1):
        for i in range(total + 1):
            j = total - i
            terms.append((x**i) * (y**j))
    return np.column_stack(terms)


def fit_trend(x: np.ndarray, y: np.ndarray, z: np.ndarray, degree: int) -> LinearRegression:
    """Ajusta a superfície de tendência de grau ``degree`` por mínimos quadrados."""
    xn, _, _ = _normalize(x)
    yn, _, _ = _normalize(y)
    X = design_matrix(xn, yn, degree)
    model = LinearRegression(fit_intercept=False)
    model.fit(X, z)
    return model


def predict_trend(model: LinearRegression, x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    xn, _, _ = _normalize(x)
    yn, _, _ = _normalize(y)
    X = design_matrix(xn, yn, degree)
    return model.predict(X)


def cv_rmse_for_degree(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    degree: int,
    k: int = 5,
    seed: int = 42,
) -> float:
    """RMSE médio em validação cruzada k-fold para um grau de polinômio.

    Isso é o coração do critério objetivo: mede o quão bem a superfície de
    grau ``degree`` prevê pontos que ela NÃO viu no ajuste — diferente do
    RMSE de treino, que só diminui (ou fica igual) conforme o grau aumenta.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    errors = []
    for train_idx, test_idx in kf.split(x):
        model = fit_trend(x[train_idx], y[train_idx], z[train_idx], degree)
        pred = predict_trend(model, x[test_idx], y[test_idx], degree)
        errors.append(np.sqrt(np.mean((z[test_idx] - pred) ** 2)))
    return float(np.mean(errors))


def aic_for_degree(x: np.ndarray, y: np.ndarray, z: np.ndarray, degree: int) -> float:
    """AIC do ajuste completo (todos os pontos) para o grau dado — informativo,
    complementa (não substitui) o critério de validação cruzada."""
    model = fit_trend(x, y, z, degree)
    pred = predict_trend(model, x, y, degree)
    n = len(z)
    p = model.coef_.shape[0]
    rss = np.sum((z - pred) ** 2)
    return n * np.log(rss / n) + 2 * p


def choose_best_degree(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    degrees: range = range(0, 7),
    k: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Roda CV e AIC para cada grau candidato e retorna uma tabela comparativa.

    O grau escolhido pelo critério objetivo é o de menor ``cv_rmse`` — a
    coluna ``aic`` é só para conferência (costuma concordar com o CV, mas
    pode divergir se o AIC for otimista em graus muito altos).
    """
    rows = []
    for d in degrees:
        cv = cv_rmse_for_degree(x, y, z, d, k=k, seed=seed)
        aic = aic_for_degree(x, y, z, d)
        rows.append({"degree": d, "cv_rmse": cv, "aic": aic})
        logger.info("grau %d: cv_rmse=%.4f, aic=%.2f", d, cv, aic)
    return pd.DataFrame(rows)


def separate(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    degree: int | None = None,
    degrees: range = range(0, 7),
    k: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int, pd.DataFrame]:
    """Separa campo regional/residual. Se ``degree`` for None, escolhe pelo
    critério objetivo (menor cv_rmse dentre ``degrees``).

    Retorna (regional, residual, grau_escolhido, tabela_de_comparacao).
    ``tabela_de_comparacao`` vem vazia se ``degree`` foi passado explicitamente.
    """
    comparison = pd.DataFrame()
    if degree is None:
        comparison = choose_best_degree(x, y, z, degrees=degrees, k=k, seed=seed)
        degree = int(comparison.loc[comparison["cv_rmse"].idxmin(), "degree"])
        logger.info("Grau escolhido pelo critério objetivo (menor cv_rmse): %d", degree)
    model = fit_trend(x, y, z, degree)
    regional = predict_trend(model, x, y, degree)
    residual = z - regional
    return regional, residual, degree, comparison


def _load_sandwell_xyz(path: Path) -> pd.DataFrame:
    """Lê um .xyz do Sandwell (sem cabeçalho: lon 0-360, lat, valor)."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["lon", "lat", "value"])
    df["lon"] = ((df["lon"] + 180) % 360) - 180
    return df


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help=".xyz do Sandwell (ou CSV com colunas lon,lat,value)")
    p.add_argument("--max-degree", type=int, default=6)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--out", type=Path, default=None, help="CSV de saída com lon,lat,value,regional,residual")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_arg_parser().parse_args()
    if args.input.suffix == ".xyz":
        df = _load_sandwell_xyz(args.input)
    else:
        df = pd.read_csv(args.input)
    x = df["lon"].to_numpy()
    y = df["lat"].to_numpy()
    z = df["value"].to_numpy()

    comparison = choose_best_degree(x, y, z, degrees=range(0, args.max_degree + 1), k=args.k)
    print(comparison.to_string(index=False))

    regional, residual, degree, _ = separate(x, y, z, degree=None, degrees=range(0, args.max_degree + 1), k=args.k)
    df["regional"] = regional
    df["residual"] = residual
    logger.info("Grau final escolhido: %d", degree)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        logger.info("Salvo %s", args.out)


if __name__ == "__main__":
    main()
