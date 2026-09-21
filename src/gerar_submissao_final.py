"""
Retreina o Modelo 1 (Caso 3, sem features de precipitacao -- venceu a
comparacao empirica da secao 12 das notas tecnicas) usando TODO o
historico disponivel (1940-2022), e aplica no periodo de teste real
(teste_features.nc, jan/2023-dez/2024) para gerar a submissao final.

Numero de arvores: fixo em 1481 (o best_iteration encontrado no
holdout original 2017-2022 do Modelo 1), ja que aqui nao ha mais dados
"de fora" para early stopping -- pratica padrao: usa o numero de
rodadas escolhido por validacao anterior para o treino final com tudo.
"""

import numpy as np
import pandas as pd
import xarray as xr
import lightgbm as lgb

from features import construir_dataset, construir_features_teste, climatologia_alvo

DATA = "data"
SEED = 42
N_PIXELS = 20000
NUM_BOOST_ROUND = 1481

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


def prever(model, X, num_iteration=None):
    return np.clip(model.predict(X, num_iteration=num_iteration), 0, None)


def main():
    print("Calculando climatologia com todo o historico (1940-2022)...", flush=True)
    alvo_completo = xr.open_dataset(f"{DATA}/treino_tp_alvo.nc").tp_alvo
    clim = climatologia_alvo(alvo_completo)
    n_pix_total = alvo_completo.sizes["lat"] * alvo_completo.sizes["lon"]
    del alvo_completo

    rng = np.random.default_rng(SEED)
    pix = rng.choice(n_pix_total, size=N_PIXELS, replace=False)

    print("Construindo features de treino (1940-2022, subamostra de pixels)...", flush=True)
    X_train, y_train, feature_names = construir_dataset(slice(None, "2022-11-01"), pix, clim)
    print(f"  X_train: {X_train.shape}", flush=True)

    print(f"Treinando LightGBM ({NUM_BOOST_ROUND} arvores, sem early stopping)...", flush=True)
    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    model = lgb.train(PARAMS, train_set, num_boost_round=NUM_BOOST_ROUND)
    model.save_model("models/lightgbm_final.txt")
    del X_train, y_train, train_set
    print("  Modelo salvo em models/lightgbm_final.txt", flush=True)

    print("Construindo features do periodo de teste real (2023-2024)...", flush=True)
    X_teste, blocos_tempo, (lat_flat, lon_flat) = construir_features_teste(clim)
    print(f"  X_teste: {X_teste.shape}", flush=True)

    print("Gerando previsoes...", flush=True)
    pred = prever(model, X_teste)

    n_pix = len(lat_flat)
    linhas = []
    for i, (ano, mes) in enumerate(blocos_tempo):
        bloco = pred[i * n_pix:(i + 1) * n_pix]
        linhas.append(pd.DataFrame({
            "ano": ano, "mes": mes,
            "lat": lat_flat.round(2), "lon": lon_flat.round(2),
            "tp_mm_day": bloco,
        }))
    df_pred = pd.concat(linhas, ignore_index=True)

    print("Montando CSV de submissao no formato/ordem do sample_submission...", flush=True)
    sample = pd.read_csv(f"{DATA}/sample_submission.csv")
    partes = sample["id"].str.split("_", expand=True)
    chave = pd.DataFrame({
        "ano": partes[0].astype(int), "mes": partes[1].astype(int),
        "lat": partes[2].astype(float).round(2), "lon": partes[3].astype(float).round(2),
    })
    merged = chave.merge(df_pred, on=["ano", "mes", "lat", "lon"], how="left")
    assert merged["tp_mm_day"].isna().sum() == 0, "Ha pontos sem previsao"

    sample["tp_mm_day"] = merged["tp_mm_day"].values
    sample.to_csv("submissions/modelo1_final.csv", index=False)
    print(f"\nSubmissao final salva: submissions/modelo1_final.csv ({len(sample)} linhas)", flush=True)
    print(sample["tp_mm_day"].describe())


if __name__ == "__main__":
    main()
