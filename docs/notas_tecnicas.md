# Notas técnicas — WORCAP: Previsão de Precipitação na América do Sul

Documento de referência com as decisões e discussões técnicas do projeto,
para consulta e apresentação. Atualizado conforme o trabalho avança.

## 1. O desafio, em uma frase

Dado o estado atmosférico observado no mês M (9 variáveis), prever a
precipitação média (mm/dia) do mês M+1, para cada um dos 78.561 pontos de
grade (grade regular 0,25°) da América do Sul. Métrica: RMSE contra a
reanálise ERA5.

## 2. Estrutura dos dados

| Arquivo | Conteúdo |
|---|---|
| `treino_tp.nc` | Precipitação observada, jan/1940–dez/2022 (996 meses) |
| `treino_tp_alvo.nc` | A mesma série, mas cada posição já contém o valor do **mês seguinte** (deslocamento já aplicado). Último mês é NaN (apontaria pra jan/2023, fora do treino) |
| `treino_{var}.nc` (9 arquivos) | t2, cloud_cover, surface_pressure, shum_850, rel_hum_850, temperature_850, geopotential_850, u_850, v_850 — mesmo período/grade, mês da própria observação |
| `teste_features.nc` | Cobre jan/2023–dez/2024 (24 meses). Eixo `time` = mês-alvo; os campos guardados são do mês **anterior** (`time_origem`). Contém `tp_alvo` (NaN, é o que se pede), `tp_ultima_obs` (precipitação real de dez/2022, fixa, repetida nas 24 linhas) e `lag_meses` (1 a 24, quão "velha" está essa âncora) |
| `sample_submission.csv` | 1.885.464 linhas = 24 meses × 78.561 pontos de grade, formato `id,tp_mm_day` |

**Verificado nos dados reais** (não é suposição):
- Nenhum NaN nas 9 variáveis atmosféricas de treino; faixas de valores fisicamente plausíveis.
- `tp_ultima_obs` é idêntico nas 24 linhas de teste e bate exatamente com `treino_tp` de dez/2022.
- As 9 variáveis atmosféricas em `teste_features.nc` são sempre o mês **real e fresco** anterior ao alvo (`time_origem` confere com o mês anterior real, mês a mês) — diferente da precipitação, que só é "fresca" no primeiro mês de teste.

## 3. O "buraco" central do problema

- **Temos completo**: tudo de 1940 a dez/2022, inclusive precipitação.
- **Temos parcialmente em 2023-2024**: as 9 variáveis atmosféricas do mês anterior a cada alvo (dado real do ERA5, sempre disponível).
- **Não temos**: a precipitação real de jan/2023 a dez/2024 — é o que devemos prever. A única pista de precipitação nesse período é a âncora fixa de dez/2022, com defasagem crescente.

## 4. A tarefa como problema de aprendizado supervisionado

Reformulação padrão de série temporal em (X, Y): pensando em termos
estatísticos, `P_i → P_{i+1}` com covariáveis `Z_i` é um modelo tipo
ARX(1) (autorregressivo de ordem 1 com exógenas). O deslocamento em si
(usar o mês `i` pra prever o mês `i+1`) **não é, por si só, uma técnica
recursiva** — é só a forma padrão de transformar sequência temporal em
pares de treino supervisionado ("windowing"). Isso vale tanto se
usarmos `P_i` como entrada quanto se usarmos só `Z_i`.

**Por que não dá pra treinar `Z_i → P_i` (mesmo mês, sem deslocamento)?**
Porque o arquivo de teste nunca fornece o `Z` do próprio mês-alvo — só o
do mês anterior (`time_origem`). Não é uma regra escrita, é uma
impossibilidade estrutural: o dado que esse esquema precisaria não é
distribuído. Além disso, fisicamente seria uma tarefa diferente
(diagnóstico do mesmo mês, não previsão do futuro).

## 5. Onde a recursividade entra de verdade

A recursividade não está no deslocamento — está em **o que usar no
lugar de `P` durante os 24 meses de teste**, já que o valor real deixa
de existir a partir de jan/2023. Três caminhos possíveis, nenhum
"proibido" pelas regras, cada um com trade-offs:

1. **Recursiva (iterada)**: usar a própria previsão do mês anterior como
   se fosse o `P` real, encadeando os 24 meses. Risco: erro acumula a
   cada passo (documentado na literatura de forecasting).
2. **Âncora estática decaindo**: usar sempre `tp_ultima_obs` (dez/2022)
   + `lag_meses` como features — é o que o próprio arquivo de teste já
   entrega pronto, sem precisar implementar recursão.
3. **Sem `P` nenhum**: usar só as variáveis sempre-frescas (`Z`) e
   estáticas (clima, localização, sazonalidade). Evita todo o problema,
   ao custo de não aproveitar nenhum sinal de persistência da chuva.

O exemplo de código do próprio enunciado da competição usa a opção com
`P` (sugerindo caminho 1 ou 2) — a escolha de ir pela opção 3 no
Modelo 1 foi uma decisão de projeto, não uma exigência da competição.
**Os três caminhos são testáveis e comparáveis** no mesmo esquema de
validação — próximo item do roadmap.

## 6. Como decidimos entre modelos, sem ter os dados reais de teste

Nunca teremos os valores reais de 2023-2024 (só o Kaggle tem). Para
comparar abordagens:

1. Fingimos que 2017-2022 é "futuro desconhecido": treinamos só com
   1940-2016, prevemos 2017-2022, e comparamos com o valor real (que
   temos, pois é dado de treino distribuído). Isso é um **proxy honesto**
   do teste real.
2. Escolhido o melhor design nesse holdout, **retreinamos usando todo o
   histórico disponível (1940-2022)** para gerar o modelo final — nessa
   etapa não precisamos mais "esconder" nada.
3. O RMSE do Kaggle (2023 alimenta o leaderboard público durante a
   competição; 2024 decide a classificação final, revelado só no
   encerramento) é a validação real, fora do nosso controle.

Validação é por **corte temporal**, não k-fold aleatório: os meses são
autocorrelacionados, então embaralhar aleatoriamente vazaria informação
do futuro pro passado.

## 7. Modelo 1 — especificação atual

- **Algoritmo**: LightGBM (gradient boosting de árvores), objetivo de
  regressão (L2), alinhado com a métrica RMSE da competição.
- **Features (23)**: lat, lon, seno/cosseno do mês-alvo, climatologia do
  pixel/mês (calculada só com dados de treino), e as 9 variáveis
  atmosféricas do mês M — cada uma em versão bruta e em média espacial
  numa janela 3×3 ao redor do pixel.
- **Sem features de precipitação** (ver seção 5, opção 3).
- **Validação**: holdout temporal, treino 1940-2016, validação 2017-2022.
- **Treino em subamostra**: 20.000 de 78.561 pixels (limite de RAM da
  máquina; ~4GB livres de 15GB totais), todos os meses. Validação usa a
  grade inteira.
- **Correção aplicada**: previsões negativas (fisicamente impossíveis)
  são limitadas em zero — 0,73% dos casos no holdout, efeito pequeno
  no RMSE (1,91030 → 1,91024) mas sempre aplicado.

## 8. Resultados até agora

| Modelo | RMSE holdout (2017-2022) |
|---|---|
| Climatologia (média histórica por pixel/mês) | 2,3597 mm/dia |
| Modelo 1 (LightGBM, sem features de precipitação) | **1,9102 mm/dia** |

Melhora relativa de ~19% sobre a climatologia, cumprindo o critério
explícito da competição ("um bom modelo precisa superar a média
histórica de forma consistente").

## 9. Limitações conhecidas (honestas)

1. Holdout único — sem intervalo de confiança da estimativa de RMSE.
2. Hiperparâmetros não otimizados sistematicamente.
3. Treino em subamostra de pixels — efeito da grade inteira não testado.
4. Sem transformação da resposta (`log1p`) apesar da assimetria da chuva.
5. Importância de features só via "gain" do LightGBM (viés conhecido
   pra variáveis contínuas) — falta permutation importance/SHAP.
6. Autocorrelação espaço-temporal entre linhas não é tratada
   formalmente na estimativa de incerteza do RMSE.

## 10. Roadmap

1. Comparar formalmente as 3 estratégias da seção 5 (sem P / âncora
   estática / recursiva) no mesmo holdout.
2. Validação mais robusta (múltiplas janelas temporais).
3. Busca de hiperparâmetros.
4. Testar `log1p` no alvo.
5. SHAP / permutation importance.
6. Testar grade inteira (sem subamostragem) em máquina com mais RAM.
7. Dados externos permitidos (sem vazamento): índice ENSO (ONI/Niño 3.4),
   elevação do terreno, distância à costa.
8. Modelo 2 (CNN/U-Net) como aposta de upside, se sobrar tempo.

## 11. Prazos e submissões

- Encerramento da competição: 23/09/2026.
- Limite: 5 submissões/dia por time.
- O leaderboard mostra sempre a **melhor** submissão do time, não a
  mais recente — uma submissão pior não piora a posição, só "gasta"
  uma tentativa do dia.

## 12. Metodologia para comparar os 3 casos (recursiva / âncora / sem P)

**Hiperparâmetros fixos entre os casos.** Para isolar o efeito da
estratégia de tratamento do `P` (a variável que estamos de fato
comparando), os três casos usam os mesmos hiperparâmetros do Modelo 1
atual na primeira rodada — é um experimento controlado: muda-se uma
coisa por vez. Ajuste fino de hiperparâmetros fica para depois, só no
design vencedor (roadmap item 3).

**Qual caso a competição sugere de fato?** O exemplo de código do
enunciado usa `P` como entrada, mas isso por si só não distingue entre
Caso 1 (recursiva) e Caso 2 (âncora estática). O indício mais forte
está na estrutura do `teste_features.nc`: a organização incluiu um
campo `lag_meses` explícito junto com `tp_ultima_obs` fixo. Recursão
pura não precisaria de um contador de defasagem — ela simplesmente
realimenta a última previsão. A presença de `lag_meses` sugere que o
**Caso 2 é o caminho estruturalmente suportado** pelos arquivos
distribuídos; o Caso 1 (recursiva) é possível, mas exigiria construir
o loop de propagação por conta própria, sem apoio direto dos dados.

**Como testar os três de forma justa.** Os Casos 2 e 3 são previsão
"de tiro único" (o modelo prevê cada mês do holdout de uma vez, com
todas as features já prontas). O Caso 1 exige simulação iterativa de
verdade: prever o mês 1, usar essa previsão como entrada do mês 2, e
assim por diante. Para comparar com o teste real (sempre um horizonte
de 24 meses), o correto é simular a recursiva em **janelas rolantes de
24 meses** dentro do holdout 2017-2022 (múltiplos pontos de partida),
em vez de uma cadeia contínua de 71 meses — isso mede como o erro
acumula num horizonte do mesmo tamanho do real. Essa infraestrutura de
janelas rolantes também resolve o item 2 do roadmap (validação mais
robusta), então os dois trabalhos foram unificados.

**Estimativa de esforço** (registrada em 19/09/2026): ~1-2h de
implementação, ~1-3h de execução total (a maior parte roda em segundo
plano). O Caso 1 é o mais caro computacionalmente, por precisar de
previsão passo a passo em vez de em lote.
