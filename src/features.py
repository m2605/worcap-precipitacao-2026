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
    """
    Media historica de tp_alvo por mes-calendario ALVO (M+1) e pixel.

    O indice de tempo de tp_alvo e o mes de OBSERVACAO M (alvo[t] ja
    guarda o valor de M+1, para alinhar com os arquivos de features --
    ver construir_dataset). Por isso agrupamos pelo mes M+1, e nao pelo
    mes M do proprio indice: senao clim.sel(month=X), usado como feature
    para linhas cujo mes-alvo e X, acaba devolvendo a climatologia de
    X+1 em vez de X.
    """
    mes_alvo = ((alvo["time.month"] % 12) + 1).rename("month")
    return alvo.groupby(mes_alvo).mean("time", skipna=True)


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


def construir_features_teste(clim: xr.DataArray) -> tuple[np.ndarray, list, list]:
    """
    Monta X para o periodo de teste real (teste_features.nc), com o
    mesmo esquema de 23 features do Modelo 1 (Caso 3, sem P). As 9
    variaveis atmosfericas ja vem prontas nesse arquivo (o mes anterior
    a cada alvo); nao ha necessidade de fatiar por tempo.

    Retorna X, os ids (ano, mes) de cada bloco de tempo, e o array de
    lat/lon (para montar o CSV de submissao depois).
    """
    teste = xr.open_dataset(f"{DATA}/teste_features.nc")
    n_time, n_lat, n_lon = teste.t2.shape
    n_pix = n_lat * n_lon

    lat = teste.lat.values
    lon = teste.lon.values
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    lat_flat = lat_grid.reshape(-1)
    lon_flat = lon_grid.reshape(-1)

    meses_alvo = teste["time.month"].values.astype(np.int32)
    anos_alvo = teste["time.year"].values.astype(np.int32)
    month_sin = np.sin(2 * np.pi * meses_alvo / 12).astype(np.float32)
    month_cos = np.cos(2 * np.pi * meses_alvo / 12).astype(np.float32)

    clim_by_month = {int(m): clim.sel(month=int(m)).values.astype(np.float32).reshape(-1) for m in clim.month.values}

    cols = {}
    cols["lat"] = np.tile(lat_flat, n_time)
    cols["lon"] = np.tile(lon_flat, n_time)
    cols["month_sin"] = np.repeat(month_sin, n_pix)
    cols["month_cos"] = np.repeat(month_cos, n_pix)
    cols["climatologia"] = np.concatenate([clim_by_month[int(m)] for m in meses_alvo])

    for var in VARS_ATMOSFERICAS:
        arr = teste[var].values.astype(np.float32)  # (time, lat, lon)
        raw_flat = arr.reshape(n_time, n_pix).reshape(-1)
        arr_nb = _neighborhood_mean(arr)
        nb_flat = arr_nb.reshape(n_time, n_pix).reshape(-1)
        cols[var] = raw_flat
        cols[f"{var}_viz3x3"] = nb_flat
        del arr, arr_nb

    feature_names = list(cols.keys())
    X = np.column_stack([cols[k] for k in feature_names]).astype(np.float32)

    return X, list(zip(anos_alvo, meses_alvo)), (lat_flat, lon_flat)


def construir_dataset_ancora_treino(
    time_slice: slice,
    pixel_idx: np.ndarray,
    clim: xr.DataArray,
    lags: list[int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Treino do "Modelo B": features base (construir_dataset) + p_ancora e
    lag_meses. Para cada mes-alvo, replica uma linha por valor em `lags`,
    usando tp observado `lag` meses antes do mes-alvo como p_ancora. Isso
    ensina o modelo a modular a confianca no p_ancora conforme lag_meses
    cresce -- a mesma situacao real do periodo de teste (Casos 1 e 2
    reaproveitam este mesmo modelo, diferindo so em como aplicam
    p_ancora/lag_meses na validacao/teste).
    """
    X_base, y, feat_base = construir_dataset(time_slice, pixel_idx, clim)
    n_sub = len(pixel_idx)
    n_time = X_base.shape[0] // n_sub

    tp_full = xr.open_dataset(f"{DATA}/treino_tp.nc").tp
    tp_values = tp_full.values.astype(np.float32).reshape(tp_full.shape[0], -1)
    alvo_time = xr.open_dataset(f"{DATA}/treino_tp_alvo.nc").tp_alvo.sel(time=time_slice).time.values
    start_idx = tp_full.get_index("time").get_loc(alvo_time[0])

    # descobre o total de linhas antes, para pre-alocar (evita o pico de
    # memoria de acumular uma lista e depois concatenar, que dobra o pico)
    validos_por_lag = []
    total_linhas = 0
    for L in lags:
        i_local = np.arange(n_time)
        anchor_abs = start_idx + i_local + 1 - L
        i_validos = i_local[anchor_abs >= 0]
        validos_por_lag.append((L, i_validos, anchor_abs[anchor_abs >= 0]))
        total_linhas += len(i_validos) * n_sub

    n_feat_total = X_base.shape[1] + 2
    X = np.empty((total_linhas, n_feat_total), dtype=np.float32)
    y_out = np.empty(total_linhas, dtype=np.float32)

    pos = 0
    for L, i_validos, anchor_abs_validos in validos_por_lag:
        if len(i_validos) == 0:
            continue
        n_bloco = len(i_validos) * n_sub
        p_ancora = tp_values[anchor_abs_validos][:, pixel_idx].reshape(-1)
        linhas = np.concatenate([np.arange(i * n_sub, (i + 1) * n_sub) for i in i_validos])

        X[pos:pos + n_bloco, :-2] = X_base[linhas]
        X[pos:pos + n_bloco, -2] = p_ancora
        X[pos:pos + n_bloco, -1] = L
        y_out[pos:pos + n_bloco] = y[linhas]
        pos += n_bloco

    feature_names = feat_base + ["p_ancora", "lag_meses"]
    return X, y_out, feature_names


def adicionar_ancora_janela_fixa(
    X_base: np.ndarray,
    n_pix: int,
    p_ancora_fixo: np.ndarray,
) -> np.ndarray:
    """
    Caso 2 (ancora estatica decaindo): anexa p_ancora fixo (repetido em
    todos os meses da janela) e lag_meses crescente (1, 2, 3, ...).
    """
    n_time = X_base.shape[0] // n_pix
    lag_meses = np.repeat(np.arange(1, n_time + 1, dtype=np.float32), n_pix)
    p_ancora_col = np.tile(p_ancora_fixo, n_time)
    return np.column_stack([X_base, p_ancora_col, lag_meses]).astype(np.float32)


def adicionar_ancora_passo_recursivo(
    X_base: np.ndarray,
    p_ancora_atual: np.ndarray,
) -> np.ndarray:
    """
    Caso 1 (recursiva): um unico mes por chamada. lag_meses sempre 1,
    porque do ponto de vista do modelo o "ultimo observado" e sempre a
    previsao do passo anterior, tratada como se fosse fresca.
    """
    lag_meses = np.ones(X_base.shape[0], dtype=np.float32)
    return np.column_stack([X_base, p_ancora_atual, lag_meses]).astype(np.float32)
