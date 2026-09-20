"""Gera um mapa rapido de cada variavel para um mes de exemplo, so para
inspecao visual dos dados (nao faz parte do pipeline de modelagem)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

DATA = "data"
MES = "2022-12-01"

VARS = [
    ("tp", "treino_tp.nc", "Precipitacao (mm/dia)"),
    ("t2", "treino_t2.nc", "Temperatura 2m (K)"),
    ("cloud_cover", "treino_cloud_cover.nc", "Cobertura de nuvens (0-1)"),
    ("surface_pressure", "treino_surface_pressure.nc", "Pressao superficie (Pa)"),
    ("shum_850", "treino_shum_850.nc", "Umidade especifica 850hPa (kg/kg)"),
    ("rel_hum_850", "treino_rel_hum_850.nc", "Umidade relativa 850hPa (%)"),
    ("temperature_850", "treino_temperature_850.nc", "Temperatura 850hPa (K)"),
    ("geopotential_850", "treino_geopotential_850.nc", "Geopotencial 850hPa (m2/s2)"),
    ("u_850", "treino_u_850.nc", "Vento zonal 850hPa (m/s)"),
    ("v_850", "treino_v_850.nc", "Vento meridional 850hPa (m/s)"),
]

fig, axes = plt.subplots(2, 5, figsize=(22, 10))

for ax, (var, fname, titulo) in zip(axes.flat, VARS):
    da = xr.open_dataset(f"{DATA}/{fname}")[var].sel(time=MES)
    cmap = "Blues" if var == "tp" else "RdBu_r" if var in ("u_850", "v_850") else "viridis"
    im = da.plot(ax=ax, cmap=cmap, add_colorbar=True, cbar_kwargs={"shrink": 0.7})
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")

fig.suptitle(f"Cara dos dados — mes de {MES} (ERA5, America do Sul)", fontsize=14)
fig.tight_layout()
fig.savefig("outputs/cara_dos_dados.png", dpi=110)
print("Salvo em outputs/cara_dos_dados.png")
