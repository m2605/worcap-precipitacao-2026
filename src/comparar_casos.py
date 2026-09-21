"""
Compara as 3 estrategias de tratar a precipitacao no periodo de teste
(ver docs/notas_tecnicas.md, secao 5):

  Caso 3: sem P nenhum (Modelo 1 atual)              -> Modelo A
  Caso 2: ancora estatica (tp_ultima_obs+lag_meses)  -> Modelo B, ancora fixa
  Caso 1: recursiva (realimenta a propria previsao)  -> Modelo B, recursivo

Modelos A e B sao treinados uma vez por janela (validacao "walk-forward":
treina so com dados anteriores a janela, nunca usa o futuro). Casos 1 e 2
reaproveitam o MESMO modelo B, diferindo so em como aplicam
p_ancora/lag_meses na validacao -- ver src/features.py.

3 janelas de 24 meses, nao sobrepostas, cobrindo os ultimos ~8 anos de
treino disponivel, para nao explodir o tempo de execucao.
"""

import gc
import json
import sys

import numpy as np
import xarray as xr
import lightgbm as lgb

from features import (
    construir_dataset,
    construir_dataset_ancora_treino,
    adicionar_ancora_janela_fixa,
    adicionar_ancora_passo_recursivo,
    climatologia_alvo,
)

DATA = "data"
SEED = 42
N_PIXELS_A = 8000
N_PIXELS_B = 3000
LAGS_TREINO_B = [1, 12, 24]

JANELAS = [
    dict(nome="2015-2016", treino_ate="2014-12-01", inicio="2015-01-01", fim="2016-12-01"),
    dict(nome="2017-2018", treino_ate="2016-12-01", inicio="2017-01-01", fim="2018-12-01"),
    dict(nome="2019-2020", treino_ate="2018-12-01", inicio="2019-01-01", fim="2020-12-01"),
]

PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": SEED,
    "verbose": -1,
}


def rmse(pred, true):
    mask = ~np.isnan(true)
    return float(np.sqrt(np.mean((pred[mask] - true[mask]) ** 2)))


def prever(model, X, num_iteration=None):
    return np.clip(model.predict(X, num_iteration=num_iteration), 0, None)


def treinar(X, y, feat_names, X_val=None, y_val=None):
    train_set = lgb.Dataset(X, label=y, feature_name=feat_names)
    kwargs = dict(num_boost_round=1500)
    callbacks = [lgb.log_evaluation(0)]
    if X_val is not None:
        val_set = lgb.Dataset(X_val, label=y_val, feature_name=feat_names, reference=train_set)
        kwargs["valid_sets"] = [val_set]
        kwargs["valid_names"] = ["holdout"]
        callbacks.append(lgb.early_stopping(50))
    model = lgb.train(PARAMS, train_set, callbacks=callbacks, **kwargs)
    return model


def rodar_janela(jan):
    rng = np.random.default_rng(SEED)
    print(f"\n=== Janela {jan['nome']} (treino ate {jan['treino_ate']}) ===", flush=True)

    alvo_treino = xr.open_dataset(f"{DATA}/treino_tp_alvo.nc").tp_alvo.sel(
        time=slice(None, jan["treino_ate"])
    )
    clim = climatologia_alvo(alvo_treino)
    n_pix_total = alvo_treino.sizes["lat"] * alvo_treino.sizes["lon"]
    del alvo_treino

    # ---- Modelo A (Caso 3: sem P) ----
    pix_a = rng.choice(n_pix_total, size=N_PIXELS_A, replace=False)
    print("  Construindo treino Modelo A...", flush=True)
    X_a, y_a, feat_a = construir_dataset(slice(None, jan["treino_ate"]), pix_a, clim)
    print(f"    X_a: {X_a.shape}", flush=True)
    modelo_a = treinar(X_a, y_a, feat_a)
    del X_a, y_a, pix_a
    gc.collect()

    print("  Validando Caso 3 na janela...", flush=True)
    X_val3, y_val3, _ = construir_dataset(slice(jan["inicio"], jan["fim"]), None, clim)
    pred3 = prever(modelo_a, X_val3, num_iteration=modelo_a.best_iteration)
    rmse_caso3 = rmse(pred3, y_val3)
    print(f"    RMSE Caso 3 (sem P): {rmse_caso3:.4f}", flush=True)
    del X_val3, y_val3, pred3, modelo_a
    gc.collect()

    # ---- Modelo B (Casos 1 e 2: com p_ancora + lag_meses) ----
    pix_b = rng.choice(n_pix_total, size=N_PIXELS_B, replace=False)
    print("  Construindo treino Modelo B (aumentado com lags)...", flush=True)
    X_b, y_b, feat_b = construir_dataset_ancora_treino(
        slice(None, jan["treino_ate"]), pix_b, clim, LAGS_TREINO_B
    )
    print(f"    X_b: {X_b.shape}", flush=True)
    modelo_b = treinar(X_b, y_b, feat_b)
    del X_b, y_b, pix_b
    gc.collect()

    p_ancora_inicial = xr.open_dataset(f"{DATA}/treino_tp.nc").tp.sel(
        time=jan["treino_ate"]
    ).values.astype(np.float32).reshape(-1)

    print("  Validando Caso 2 (ancora fixa) na janela...", flush=True)
    X_base_val, y_val2, feat_base = construir_dataset(slice(jan["inicio"], jan["fim"]), None, clim)
    X_val2 = adicionar_ancora_janela_fixa(X_base_val, len(p_ancora_inicial), p_ancora_inicial)
    pred2 = prever(modelo_b, X_val2, num_iteration=modelo_b.best_iteration)
    rmse_caso2 = rmse(pred2, y_val2)
    print(f"    RMSE Caso 2 (ancora estatica): {rmse_caso2:.4f}", flush=True)
    del X_base_val, X_val2, pred2
    gc.collect()

    print("  Validando Caso 1 (recursiva) na janela, mes a mes...", flush=True)
    meses_janela = xr.open_dataset(f"{DATA}/treino_tp_alvo.nc").tp_alvo.sel(
        time=slice(jan["inicio"], jan["fim"])
    ).time.values
    p_ancora_atual = p_ancora_inicial.copy()
    preds_recursivo, ys_recursivo = [], []
    for mes in meses_janela:
        X_mes_base, y_mes, _ = construir_dataset(slice(mes, mes), None, clim)
        X_mes = adicionar_ancora_passo_recursivo(X_mes_base, p_ancora_atual)
        pred_mes = prever(modelo_b, X_mes, num_iteration=modelo_b.best_iteration)
        preds_recursivo.append(pred_mes)
        ys_recursivo.append(y_mes)
        p_ancora_atual = pred_mes
        del X_mes_base, X_mes
    rmse_caso1 = rmse(np.concatenate(preds_recursivo), np.concatenate(ys_recursivo))
    print(f"    RMSE Caso 1 (recursiva): {rmse_caso1:.4f}", flush=True)

    return dict(
        janela=jan["nome"],
        caso1_recursiva=rmse_caso1,
        caso2_ancora=rmse_caso2,
        caso3_sem_p=rmse_caso3,
    )


if __name__ == "__main__":
    idx = int(sys.argv[1])
    resultado = rodar_janela(JANELAS[idx])
    with open(f"outputs/resultado_janela_{idx}.json", "w") as f:
        json.dump(resultado, f)
    print("\nRESULTADO_JSON:", json.dumps(resultado))
