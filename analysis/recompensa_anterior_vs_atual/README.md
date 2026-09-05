# Comparação entre a arquitetura de recompensa anterior e a atual

Por que as execuções `double-six` que chegavam a ~66% contra o oponente
aleatório passaram a parar em ~65% depois do redesenho de recompensa
introduzido em `895d512` ("reward shaping changes and bugs correction").

**A resposta é a recompensa.** A execução `default_lookup` é a comparação
controlada que faltava: recompensa atual com `lr = 0.001`, os mesmos buckets,
a mesma semente e os mesmos pesos supervisionados de `bucket_heuristic_recent`.
Chegou a 65,335% contra 66,067%, ou seja **−0,732 pp** com a taxa de
aprendizado igualada. Baixar a lr recuperou parte da diferença — +0,232 pp
sobre `default_lr016_lookup`, que é a comparação com baseline igual — mas os
0,732 pp restantes não têm mais a lr como explicação.

O efeito foi reproduzido em outra máquina: `rick_heuristic`, treinada em outro
computador com outro binário supervisionado e `lr = 0.002`, chegou a 65,397%
contra 66,375% de `bucket_heuristic` — **−0,978 pp** na mesma condição de
oponentes. Mesma direção, praticamente o mesmo tamanho.

**E a correção proposta foi executada.** A análise previa que
`reward_eta ≈ 0,112` devolveria à metade local o peso relativo que ela tinha
sob a recompensa anterior. Duas execuções com `reward_eta = 0,115` — uma aqui,
outra na máquina do orientador — confirmam a previsão em três medidas
independentes: a razão local/terminal registrada ao vivo caiu de 2,45x para
**0,285x**, contra os 0,30x da recompensa anterior; a correlação por iteração
entre recompensa e vitória subiu de +0,71–+0,77 para **+0,89**, de volta à
faixa antiga; e no orçamento comum de 2,7 M de partidas elas ficaram **+0,600
pp** e **+0,718 pp** acima das execuções gêmeas que mantiveram `eta = 0,5`.
Ressalva: as quatro treinam com o bucket `random`, o mesmo oponente do
diagnóstico, então o contraste entre elas é limpo mas o nível absoluto não é
comparável com o bloco da recompensa anterior. O experimento que falta é o
mesmo `eta = 0,115` com buckets `heuristic,recent`.

O mesmo experimento **corrige** uma leitura anterior: a entropia que voltava a
subir no meio do treino e a pressão sobre a região de confiança do PPO eram a
taxa de aprendizado, não a recompensa. Com `lr = 0.001` a entropia volta a cair
de forma monótona e a KL mediana volta a 0,0024. A lr explica a dinâmica, a
recompensa explica o resultado.

A análise separa quatro perguntas que costumam ser misturadas:

1. **O resultado caiu?** Comparação das curvas do diagnóstico periódico de
   todas as execuções canônicas `forever` em `double-six`, agrupadas pela
   arquitetura de recompensa que cada uma usou. A arquitetura é decidida pela
   presença dos pesos do redesenho em `locked_arguments`, não pela data.
2. **Foi a recompensa ou a taxa de aprendizado?** O par controlado
   `bucket_heuristic_recent` / `default_lookup` mantém tudo fixo menos a
   recompensa e o baseline de vantagem, e é lido tanto no pico quanto em
   volumes de partidas iguais. As execuções da tabela têm idades muito
   diferentes — de 2,7 M a 28,4 M de partidas — e por isso toda comparação de
   resultado é feita também no **horizonte comum**, o maior volume que todas
   atravessaram, além do melhor de cada vida inteira.
3. **A recompensa explica a queda?** As duas funções de recompensa são
   **recomputadas sobre o mesmo conjunto de 422.055 decisões reais**, de modo
   que sejam comparadas nas mesmas trajetórias em vez de através das execuções
   que cada uma otimizou. `gamma_f`, `gamma_i` e o modo de distância são
   idênticos em todas as execuções e o script aborta se alguma divergir neles.
   `reward_eta` também era parte dessa invariante, até as execuções corrigidas
   o moverem de propósito; hoje ele é uma coluna do resumo, e as comparações
   **entre arquiteturas** continuam restritas às execuções que o mantiveram em
   0,5. As execuções da arquitetura atual confirmam o resultado sozinhas, pelas
   colunas `terminal_abs_mean`/`local_abs_mean` da versão 8 do esquema de
   métricas.
4. **`reward_eta` corrige?** As execuções de `ETA_PAIRS` seguram tudo — GPU,
   binário supervisionado, semente, lr, baseline e buckets — e mexem só em
   `reward_eta`. É a única comparação da análise em que a recompensa atual
   aparece dos dois lados.

As duas arquiteturas, com os valores que as execuções realmente usaram:

| | anterior | atual |
|---|---|---|
| Utilidade terminal | `±1 − 0.05 · pontos_na_mão_do_aprendiz` | `±1` (mão vazia) ou `±m(Δp)` (bloqueio) |
| Faixa da utilidade terminal | `[−5.05, +1]` observada | `[−1, +1]` por construção |
| Evento de saque | `±0.2` | `±1.0` |
| Evento de passe | `±0.1` | `±1.0` |
| `gamma_f` / `gamma_i` / `reward_eta` | 0.95 / 0.90 / 0.5 | 0.95 / 0.90 / 0.5, e 0.115 nas duas execuções corrigidas |

Fontes de dados, todas apenas lidas:

- `models/rl/domino_rl_forever_seed42_run*/run_compact_diagnostics/rl_vs_random_progress.csv`
  e `run_config.json` — a curva do diagnóstico periódico contra o oponente
  aleatório (100.000 partidas fixas, mesma semente em todas as execuções) e a
  configuração imutável de cada execução.
- `models/rl/modelos_rick/*/` — quatro execuções treinadas em outras máquinas e
  recebidas como o pacote de troca, sem o modelo. Todas trazem a curva e a
  configuração, e por isso entram em todas as comparações de resultado. Três
  vieram apenas com os quatro arquivos, sem `training_metrics.jsonl`, e por
  isso ficam de fora das figuras `07` e `09` e das colunas de dinâmica de
  `resumo_execucoes.csv`; `20260904-155_...eta0115` veio com o registro por
  iteração junto e entra em tudo.
- `models/rl/domino_rl_forever_seed42_run*/training_metrics.jsonl` — uma linha
  por iteração de PPO, com `reward_mean`, `good_pct`, `batch_win_rate`,
  `entropy`, `final_approx_kl` e `final_clip_fraction`. As execuções gravadas
  sob a versão 8 do esquema trazem também `terminal_abs_mean` e
  `local_abs_mean`, que são `E|(1−η)·G_T|` e `E|η·G_I|` como o rollout os
  creditou; execuções da versão 7 deixam essas colunas em branco em vez de
  receberem um valor inventado.
- `analysis/reward_lookup_table/derived/double-six_reward_lookup_samples.json.gz`
  — 422.055 decisões reais de 100.000 partidas `double-six`, com a
  decomposição terminal em componentes unitários, os eventos locais futuros e
  as distâncias de cada um. Nenhum peso, `gamma` ou `eta` foi aplicado a esse
  corpus, que é exatamente o que permite aplicar as duas recompensas a ele.
- `training/rl/reward_model.py` — `blocked_reward_magnitude` é importada em
  vez de reimplementada, para que a curva de `m(Δp)` na figura 3 não possa
  divergir do que o treino usa.

Comando:

```bash
/home/diego/CCO/amb_virtual/bin/python analysis/recompensa_anterior_vs_atual/analyze.py
```

O script regenera, neste diretório, `REPORT.md`, `analysis_summary.json`,
quatro CSVs e as nove figuras:

| Figura | Conteúdo |
|---|---|
| `01_taxa_vitoria_por_partidas.png` | Curvas de vitória contra o aleatório por volume de partidas, azul para a recompensa anterior, vermelho para a atual e verde para a atual com `eta` corrigido, com o pico de cada execução |
| `02_taxa_vitoria_por_tempo.png` | As mesmas curvas por tempo de parede de RL |
| `03_utilidade_terminal.png` | A utilidade terminal das quatro classes de desfecho sob cada arquitetura, em função dos pontos na mão e da margem de bloqueio |
| `04_retorno_por_decisao.png` | Distribuição do retorno por decisão nas mesmas 422.055 decisões, separada entre partidas ganhas e perdidas |
| `05_equilibrio_terminal_local.png` | Magnitude média de cada metade do retorno e a razão local/terminal resultante |
| `06_alinhamento_recompensa_resultado.png` | Correlação entre retorno e vitória, e as decisões cujo sinal contradiz o resultado |
| `07_dinamica_de_treino.png` | Recompensa média, decisões positivas, entropia e fração recortada ao longo das iterações |
| `08_resumo_resultados.png` | Melhor resultado de cada execução, com IC 95%, no horizonte comum de partidas (barra cheia) e na execução inteira (barra clara) |
| `09_mistura_ao_vivo.png` | O equilíbrio terminal/local que cada execução da arquitetura atual registrou durante o próprio treino, com os valores recomputados sobre o corpus como referência |

E os CSVs:

| Arquivo | Conteúdo |
|---|---|
| `resumo_execucoes.csv` | Uma linha por execução: configuração, melhor e último resultado, correlação recompensa/vitória, entropia e pressão do PPO |
| `curvas_vitoria.csv` | Todos os pontos de diagnóstico de todas as execuções |
| `desfechos_terminais.csv` | Utilidade terminal média, mínima e máxima por classe de desfecho sob cada arquitetura |
| `retorno_por_decisao.csv` | Percentis do retorno e de cada uma de suas metades sob cada arquitetura |

As execuções comparadas, com a arquitetura de recompensa de cada uma:

| Execução | Arq. | lr | Baseline | Buckets |
|---|---|---|---|---|
| `bucket_heuristic` | anterior | 0.001 | batch-mean | heuristic |
| `bucket_heuristic_recent` | anterior | 0.001 | batch-mean | heuristic,recent |
| `baseline_zero` | anterior | 0.001 | zero | heuristic,recent |
| `bucket_all` | anterior | 0.001 | batch-mean | os seis |
| `d6_maxwr_lr032` | atual | 0.032 | batch-mean | heuristic |
| `d6_maxwr_lr016` | atual | 0.016 | batch-mean | heuristic (interrompida) |
| `default_lr032` | atual | 0.032 | batch-mean | heuristic,recent |
| `default_lr016_lookup` | atual | 0.016 | lookup-table | heuristic,recent |
| `default_lookup` | atual | **0.001** | lookup-table | heuristic,recent |
| `rick_heuristic` | atual | 0.002 | batch-mean | heuristic (outra máquina) |
| `rick_random_desktop` | atual | 0.002 | batch-mean | random (outra máquina) |
| `rick_random_notebook` | atual | 0.002 | batch-mean | random (outra máquina) |
| `d6_random_eta0115` | atual, `eta = 0.115` | 0.002 | batch-mean | random |
| `rick_random_eta0115` | atual, `eta = 0.115` | 0.002 | batch-mean | random (outra máquina) |

As quatro execuções com o bucket `random` treinam contra o mesmo oponente que o
diagnóstico mede. Os números delas são reais, mas medem outra coisa, e por isso
ficam marcados com `*` na figura `08` e fora do ordenamento entre arquiteturas.
O contraste **entre** elas continua válido, porque as duas pontas carregam o
mesmo viés: é sobre esse par que o teste de `reward_eta` é lido.

Para acrescentar uma execução nova, basta incluí-la em `RUNS` e dar-lhe uma cor
em `RUN_COLORS` — e, se ela isolar `reward_eta`, um par em `ETA_PAIRS`; a arquitetura é inferida da própria configuração gravada. O
caminho pode ser o diretório de uma execução local ou o pacote de troca
desempacotado — `analyze.py` reconhece os dois formatos e trata
`training_metrics.jsonl` como opcional, aceitando-o também comprimido.

## Trocar execuções com outra máquina

Para uma execução treinada em outro computador entrar inteira nesta análise,
bastam os quatro arquivos de `run_compact_diagnostics/` mais o
`training_metrics.jsonl`. Este último fica um nível acima, na raiz do diretório
da execução, e por isso não vem junto quando se copia só a pasta de
diagnósticos — sem ele a execução entra nas comparações de resultado, mas fica
de fora das figuras `07` e `09` e das colunas de dinâmica.

O registro por iteração é gravado e sincronizado a cada iteração de PPO, então
está completo em disco mesmo numa execução ainda em andamento. `analyze.py` o
aceita comprimido (`.gz`) e ignora uma última linha incompleta, de modo que uma
cópia feita durante o treino também serve.

Uma execução ainda em andamento avança entre duas gerações do relatório; o
`REPORT.md` traz no topo o instante em que foi gerado.

Nenhum diretório de execução, modelo ou dataset é escrito ou modificado.
