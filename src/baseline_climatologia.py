"""
Baseline de climatologia mensal: para cada ponto de grade e mes-calendario,
preve a media historica de precipitacao daquele mes.

Esta e a "regua a bater" citada no enunciado do desafio. Antes de gerar a
submissao, validamos o baseline com um holdout temporal (treina climatologia
em 1940-2016, mede RMSE contra 2017-2022 observado) para termos um numero
de referencia real, no mesmo tipo de metrica usada no leaderboard.
"""

import numpy as np
import pandas as pd
import xarray as xr

DATA = "data"


def rmse(pred, true):
    mask = ~np.isnan(true)
    return float(np.sqrt(np.mean((pred[mask] - true[mask]) ** 2)))


def climatologia_por_mes(tp: xr.DataArray) -> xr.DataArray:
    """Media historica de tp para cada mes-calendario (1-12), por ponto de grade."""
    return tp.groupby("time.month").mean("time")


def validar_holdout(tp: xr.DataArray, alvo: xr.DataArray, corte="2017-01-01"):
    treino = tp.sel(time=slice(None, "2016-12-01"))
    clim = climatologia_por_mes(treino)

    holdout_alvo = alvo.sel(time=slice(corte, "2022-12-01"))
    meses = holdout_alvo["time.month"].values

    pred = clim.sel(month=xr.DataArray(meses, dims="time")).values
    true = holdout_alvo.values

    return rmse(pred, true)


def gerar_submissao(tp: xr.DataArray, sample_path: str, out_path: str):
    clim = climatologia_por_mes(tp)  # dims: month, lat, lon

    sub = pd.read_csv(sample_path)
    partes = sub["id"].str.split("_", expand=True)
    ano = partes[0].astype(int)
    mes = partes[1].astype(int)
    lat = partes[2].astype(float)
    lon = partes[3].astype(float)

    clim_df = clim.to_dataframe(name="tp_clim").reset_index()
    clim_df["lat"] = clim_df["lat"].round(2)
    clim_df["lon"] = clim_df["lon"].round(2)

    chave = pd.DataFrame({"month": mes, "lat": lat.round(2), "lon": lon.round(2)})
    merged = chave.merge(clim_df, on=["month", "lat", "lon"], how="left")

    assert merged["tp_clim"].isna().sum() == 0, "Ponto de grade sem climatologia correspondente"

    sub["tp_mm_day"] = merged["tp_clim"].values
    sub.to_csv(out_path, index=False)
    return sub


if __name__ == "__main__":
    tp = xr.open_dataset(f"{DATA}/treino_tp.nc").tp
    alvo = xr.open_dataset(f"{DATA}/treino_tp_alvo.nc").tp_alvo

    score = validar_holdout(tp, alvo)
    print(f"RMSE climatologia (holdout 2017-2022, treinado em 1940-2016): {score:.4f} mm/dia")

    sub = gerar_submissao(tp, f"{DATA}/sample_submission.csv", "submissions/baseline_climatologia.csv")
    print(f"Submissao gerada: submissions/baseline_climatologia.csv ({len(sub)} linhas)")
