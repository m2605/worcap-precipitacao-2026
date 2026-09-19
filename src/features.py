"""
Construcao de features para o Modelo 1 (LightGBM).

Design (ver discussao no README/commit): usamos apenas as 9 variaveis
atmosfericas do mes M (sempre "frescas", disponiveis igualmente em treino
e teste) -- nao usamos precipitacao como feature de entrada, pois ela so
esta disponivel de forma confiavel para o primeiro dos 24 meses de teste.

Features por (tempo, lat, lon):
  - lat, lon
  - month_sin, month_cos       -> sazonalidade do mes-alvo (M+1)
  - climatologia                -> media historica de tp_alvo p/ aquele
                                    pixel e mes-calendario
  - 9 variaveis atmosfericas no pixel (mes M)
  - 9 variaveis atmosfericas, media numa janela 3x3 ao redor do pixel (mes M)
"""

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter

DATA = "data"

VARS_ATMOSFERICAS = [
    "t2", "cloud_cover", "surface_pressure", "shum_850",
    "rel_hum_850", "temperature_850", "geopotential_850", "u_850", "v_850",
]


def climatologia_alvo(alvo: xr.DataArray) -> xr.DataArray:
    """Media historica de tp_alvo por mes-calendario e pixel."""
    return alvo.groupby("time.month").mean("time", skipna=True)


def _neighborhood_mean(arr_txy: np.ndarray) -> np.ndarray:
    """Media numa janela 3x3 espacial, sem suavizar no tempo. arr: (time, lat, lon)."""
    return uniform_filter(arr_txy, size=(1, 3, 3), mode="nearest")


def construir_dataset(
    time_slice: slice,
    pixel_idx: np.ndarray | None,
    clim: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Monta X, y para o intervalo de tempo dado (aplicado ao mes M; o alvo
    e automaticamente M+1 via treino_tp_alvo.nc, ja deslocado).

    pixel_idx: indices (flatten lat*lon) a manter. None = todos os pixels.
    clim: climatologia de tp_alvo por mes-calendario e pixel (calculada
          previamente, so com dados de treino, para nao vazar informacao).
    """
    alvo_ds = xr.open_dataset(f"{DATA}/treino_tp_alvo.nc").tp_alvo.sel(time=time_slice)
    n_time, n_lat, n_lon = alvo_ds.shape
    n_pix = n_lat * n_lon

    lat = alvo_ds.lat.values
    lon = alvo_ds.lon.values
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    lat_flat = lat_grid.reshape(-1)
    lon_flat = lon_grid.reshape(-1)

    meses_alvo = (alvo_ds["time.month"].values.astype(np.int32) % 12) + 1  # mes de M+1
    month_sin = np.sin(2 * np.pi * meses_alvo / 12).astype(np.float32)
    month_cos = np.cos(2 * np.pi * meses_alvo / 12).astype(np.float32)

    y_full = alvo_ds.values.astype(np.float32).reshape(n_time, n_pix)

    clim_by_month = {int(m): clim.sel(month=int(m)).values.astype(np.float32).reshape(-1) for m in clim.month.values}

    if pixel_idx is None:
        pixel_idx = np.arange(n_pix)
    n_sub = len(pixel_idx)

    cols = {}
    cols["lat"] = np.tile(lat_flat[pixel_idx], n_time)
    cols["lon"] = np.tile(lon_flat[pixel_idx], n_time)
    cols["month_sin"] = np.repeat(month_sin, n_sub)
    cols["month_cos"] = np.repeat(month_cos, n_sub)
    cols["climatologia"] = np.concatenate([
        clim_by_month[int(m)][pixel_idx] for m in meses_alvo
    ])

    for var in VARS_ATMOSFERICAS:
        da = xr.open_dataset(f"{DATA}/treino_{var}.nc")[var].sel(time=time_slice)
        arr = da.values.astype(np.float32)  # (time, lat, lon)
        raw_flat = arr.reshape(n_time, n_pix)[:, pixel_idx].reshape(-1)

        arr_nb = _neighborhood_mean(arr)
        nb_flat = arr_nb.reshape(n_time, n_pix)[:, pixel_idx].reshape(-1)

        cols[var] = raw_flat
        cols[f"{var}_viz3x3"] = nb_flat
        del arr, arr_nb, da

    feature_names = list(cols.keys())
    X = np.column_stack([cols[k] for k in feature_names]).astype(np.float32)
    y = y_full[:, pixel_idx].reshape(-1)

    return X, y, feature_names
