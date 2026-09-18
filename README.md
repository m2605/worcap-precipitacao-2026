# WORCAP — Previsão Climática de Precipitação sobre a América do Sul

Desafio de dados do Workshop de Computação Aplicada (WORCAP/INPE). O objetivo é
prever a precipitação média (mm/dia) do mês seguinte em cada ponto de grade da
América do Sul, a partir do estado atmosférico do mês atual (reanálise ERA5).

Competição no Kaggle: https://www.kaggle.com/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul

## Estrutura do repositório

```
data/         dados brutos do Kaggle (.nc, .csv) — NÃO versionado, baixar manualmente
src/          código-fonte (pré-processamento, features, modelos, submissão)
notebooks/    exploração e experimentos
models/       checkpoints/modelos treinados — NÃO versionado
outputs/      previsões intermediárias, gráficos, métricas — NÃO versionado
submissions/  arquivos de submissão gerados — NÃO versionado por padrão
```

## Dados

Baixe os arquivos da aba "Data" da competição no Kaggle e coloque em `data/`:

- `treino_tp.nc` — precipitação mensal observada (1940–2022), variável alvo base
- `treino_tp_alvo.nc` — precipitação do mês M+1 (alvo de treino, já deslocado)
- `treino_<var>.nc` — 9 variáveis atmosféricas adicionais (t2m, nuvens, pressão,
  umidade específica/relativa, temperatura/geopotencial/vento em 850 hPa)
- `teste_features.nc` — features do período avaliado (jan/2023–dez/2024)
- `sample_submission.csv` — formato esperado da submissão

## Tarefa

Dado o estado atmosférico do mês M (10 variáveis, grade 301×261, 0.25°),
prever a precipitação média do mês M+1 em cada um dos 78.561 pontos de grade.

- Treino: jan/1940 – dez/2022
- Avaliação: jan/2023 – dez/2024 (2023 = leaderboard público, 2024 = privado/final)
- Métrica: RMSE (mm/dia) sobre todos os pontos de grade e meses de teste
- Baseline a superar: climatologia mensal (média histórica por mês-calendário)

## Como rodar

```bash
pip install -r requirements.txt
# baixar dados via Kaggle API (requer kaggle.json em ~/.kaggle/)
kaggle competitions download -c previsao-climatica-de-precipitacao-sobre-a-america-do-sul -p data/
```

## Status

Em andamento — ver commits para progresso.
