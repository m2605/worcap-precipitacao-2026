"""
Modelo 1: LightGBM tabular, usando so as 9 variaveis atmosfericas (sempre
"frescas") + lat/lon + sazonalidade + climatologia como features. Ver
src/features.py para a justificativa de cada feature.

Validacao: mesmo esquema do baseline de climatologia (treina em 1940-2016,
mede RMSE contra 2017-2022 observado) para comparar diretamente com o
numero de referencia (2.3597 mm/dia).

Por limite de RAM (~4GB livres), o TREINO usa uma subamostra aleatoria de
pixels (mantendo todos os meses); a VALIDACAO usa a grade inteira, para
nao introduzir ruido extra na estimativa de erro.
"""

import numpy as np
import xarray as xr
import lightgbm as lgb

from features import construir_dataset, climatologia_alvo

DATA = "data"
SEED = 42
N_PIXELS_TREINO = 20000


def rmse(pred, true):
    mask = ~np.isnan(true)
    return float(np.sqrt(np.mean((pred[mask] - true[mask]) ** 2)))


def prever(model, X, num_iteration=None):
    """Previsao com clip em 0: precipitacao nunca e negativa."""
    return np.clip(model.predict(X, num_iteration=num_iteration), 0, None)


def main():
    alvo_treino = xr.open_dataset(f"{DATA}/treino_tp_alvo.nc").tp_alvo.sel(
        time=slice(None, "2016-12-01")
    )
    clim_treino = climatologia_alvo(alvo_treino)

    n_lat = alvo_treino.sizes["lat"]
    n_lon = alvo_treino.sizes["lon"]
    n_pix = n_lat * n_lon
    rng = np.random.default_rng(SEED)
    pixel_idx_treino = rng.choice(n_pix, size=N_PIXELS_TREINO, replace=False)

    print("Construindo features de treino (1940-2016, subamostra de pixels)...")
    X_train, y_train, feature_names = construir_dataset(
        time_slice=slice(None, "2016-12-01"),
        pixel_idx=pixel_idx_treino,
        clim=clim_treino,
    )
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")

    print("Construindo features de validacao (2017-2022, grade inteira)...")
    X_val, y_val, _ = construir_dataset(
        time_slice=slice("2017-01-01", "2022-11-01"),
        pixel_idx=None,
        clim=clim_treino,
    )
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_set = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_set)

    params = {
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

    print("Treinando LightGBM...")
    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[val_set],
        valid_names=["holdout"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )

    pred_val = prever(model, X_val, num_iteration=model.best_iteration)
    score = rmse(pred_val, y_val)
    print(f"\nRMSE Modelo 1 (LightGBM) no holdout 2017-2022: {score:.4f} mm/dia")
    print(f"RMSE climatologia no mesmo holdout: 2.3597 mm/dia (referencia anterior)")
    print(f"Melhor numero de arvores: {model.best_iteration}")

    print("\nImportancia das features (gain):")
    importancias = sorted(
        zip(feature_names, model.feature_importance(importance_type="gain")),
        key=lambda x: -x[1],
    )
    for nome, imp in importancias:
        print(f"  {nome:20s} {imp:12.1f}")

    model.save_model("models/lightgbm_modelo1.txt")
    print("\nModelo salvo em models/lightgbm_modelo1.txt")


if __name__ == "__main__":
    main()
