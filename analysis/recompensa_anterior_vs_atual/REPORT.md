# Recompensa anterior e recompensa atual: o que mudou e o que o resultado mostra

*Gerado em 2026-09-03 12:57 -0300 a partir do estado corrente dos diretórios de execução. Uma execução ainda em andamento avança entre duas gerações deste relatório.*

## Conclusão

A queda é real e maior que o ruído do painel de diagnóstico. A melhor execução sob a recompensa anterior chegou a **66.375%** contra o oponente aleatório (`bucket_heuristic`); a melhor sob a recompensa atual chegou a **65.203%** (`d6_maxwr_lr032`). A diferença de **1.172 pontos percentuais** é cerca de 4.0x a meia-largura do intervalo de 95% do painel de 100,000 partidas (±0.29 pp), então não é flutuação do diagnóstico.

A causa provável não é o novo formato da utilidade terminal, e sim o **deslocamento do equilíbrio entre a metade terminal e a metade local** que o novo formato provocou sem que `reward_eta` mudasse. Recomputando as duas recompensas sobre as **mesmas 422,055 decisões reais**, a metade local passou de 0.30x para 2.38x a magnitude da metade terminal — um fator de 8.0 no peso efetivo do termo de moldagem, com `reward_eta` fixo em 0,5 nas duas.

A consequência mensurável é que a recompensa deixou de informar sobre o resultado da partida com a mesma nitidez: a correlação entre o retorno de uma decisão e vencer caiu de **+0.832** para **+0.591**, e a fração de decisões tomadas em **partidas perdidas** que ainda assim recebem retorno positivo subiu de **4.82%** para **35.74%**.

## O que exatamente mudou

As duas arquiteturas compartilham a mesma estrutura de dois termos, os mesmos descontos e o mesmo ponto de mistura:

```
G(t) = (1 - eta) * gamma_f^k * U_T  +  eta * G_local(t)
gamma_f = 0.95   gamma_i = 0.9   eta = 0.5
```

Os valores de `gamma_f`, `gamma_i`, `reward_eta` e o modo de distância `turn-turn` são idênticos em todas as execuções comparadas aqui, então saem da comparação. O que mudou são as duas funções que produzem `U_T` e `G_local`.

**Utilidade terminal.** A anterior era um resultado binário menos uma penalidade proporcional aos pontos que o aprendiz ainda tinha na mão:

```
U_T_anterior = (+1 se venceu, -1 se perdeu) - 0.05 * pontos_na_mao_do_aprendiz
```

A atual decompõe o desfecho em duas classes mutuamente exclusivas e normaliza tudo para `[-1, +1]`:

```
U_T_atual = +/-1                        (fim por mão vazia)
U_T_atual = +/-m(dp),  m(dp) = 0.1 + 0.9 * min(dp / (2*max_pip), 1)   (bloqueio)
```

**Eventos locais.** A anterior valia `+/-0.2` para um saque e `+/-0.1` para um passe. A atual normaliza os dois para `+/-1.0` e expressa a importância relativa só pela razão entre `a_D` e `a_P`; com os pesos padrão iguais a 1, ambas as escalas resolvem para 1,0. Um saque passou a valer **5x** mais e um passe **10x** mais, em unidades da mesma utilidade terminal.

## Efeito sobre cada classe de desfecho

Medido nas mesmas decisões, sem desconto temporal:

| Desfecho | % das decisões | Pontos na mão | U anterior (média) | U anterior (mín) | U atual (média) |
|---|---|---|---|---|---|
| mão vazia — vitória | 47.89% | 0.0 | +1.000 | +1.000 | +1.000 |
| mão vazia — derrota | 33.42% | 11.3 | -1.566 | -4.350 | -1.000 |
| bloqueio — vitória | 9.89% | 13.8 | +0.311 | -1.800 | +0.673 |
| bloqueio — derrota | 8.80% | 22.9 | -2.146 | -5.050 | -0.653 |

Três leituras saem daqui:

1. **A recompensa anterior era fortemente avessa à derrota.** Uma derrota custava em média entre 1,57 e 2,15, enquanto uma vitória rendia entre 0,31 e 1,00. A assimetria não era um parâmetro escolhido: vinha da penalidade de pontos, que era subtraída em todo desfecho e não tinha piso.
2. **A recompensa anterior pagava por descartar peças pesadas.** Cada ponto restante na mão custava 0,05 independentemente de como a partida terminasse. Contra um oponente aleatório, esvaziar a mão rápido e não ficar com peças altas é exatamente a política que mais vence, então essa penalidade empurrava na direção certa para o diagnóstico usado.
3. **Vencer por bloqueio segurando peças pesadas podia ser punido.** O mínimo de `U anterior` numa vitória por bloqueio é negativo: o aprendiz vencia a partida e ainda assim recebia retorno negativo. A arquitetura atual corrige isso — e essa correção é uma melhora real de coerência, não um defeito.

## O equilíbrio entre as duas metades

É aqui que está o problema. `reward_eta` = 0,5 diz que as duas metades pesam igual, mas só controla o coeficiente, não a magnitude do que multiplica:

| Arquitetura | magnitude média da metade terminal | magnitude média da metade local | razão local/terminal |
|---|---|---|---|
| anterior | 0.5557 | 0.1658 | 0.30x |
| atual | 0.4346 | 1.0329 | 2.38x |

Sob a recompensa anterior a moldagem local era um termo acessório, com 30% da magnitude do sinal terminal. Sob a atual ela é o termo **dominante**, com 2.38 vezes a magnitude do sinal terminal. Duas mudanças somaram-se nessa direção: os eventos locais ficaram 5 a 10 vezes maiores, e a utilidade terminal ficou *menor* em módulo (0.556 para 0.435) porque perdeu a cauda de penalidade de pontos e porque as vitórias por bloqueio passaram a valer `m(dp)` em vez de aproximadamente 1.

`G_local` também não é renormalizado para `[-1, 1]` — isso é deliberado e está documentado em `training/rl/README.md` — então a soma geométrica de eventos alcança +5.57 e -4.70 nas caudas, contra +0.85 e -0.73 antes.

## Alinhamento entre recompensa e resultado

| Métrica | anterior | atual |
|---|---|---|
| correlação entre G(t) e vencer | +0.8323 | +0.5913 |
| G(t) > 0 em partidas perdidas | 4.82% | 35.74% |
| G(t) < 0 em partidas ganhas | 5.33% | 11.81% |
| discordância de sinal | 5.12% | 21.91% |
| média de G(t) | +0.0191 | +0.3086 |
| desvio de G(t) | 0.3826 | 0.7468 |

Mais de um terço das decisões tomadas em partidas que o agente **perdeu** recebem retorno positivo sob a recompensa atual. Para uma política de gradiente isso não é ruído neutro: essas decisões são reforçadas. O termo local é quase sempre positivo porque a política supervisionada inicial já força o oponente a sacar e passar com frequência, e esse crédito agora chega em escala comparável à do próprio resultado da partida.

O mesmo efeito aparece nos dados de treino ao vivo, sem nenhuma recomputação: a correlação entre `reward_mean` e `batch_win_rate` por iteração cai de +0.866 e +0.900 nas execuções antigas para +0.706 e +0.771 nas novas, e a `reward_mean` mediana sai de aproximadamente zero para cerca de +0.270 com a taxa de vitória em lote parada em ~50,8%.

## Verificação direta na iteração 1

As execuções `bucket_heuristic_recent` (anterior) e `default_lr032` (atual) partem dos mesmos pesos supervisionados com a mesma semente. Na primeira iteração elas jogam **as mesmas partidas**: 8.120 decisões e a mesma contagem de eventos `[6910, 4288, 6934, 4244]` nas duas. É um A/B exato da função de recompensa sobre trajetórias idênticas:

| Iteração 1 | bucket_heuristic_recent (anterior) | default_lr032 (atual) |
|---|---|---|
| recompensa média | -0.02323 | +0.22665 |
| desvio | 0.38669 | 0.74975 |
| mínimo / máximo | -1.477 / +0.643 | -2.232 / +2.463 |
| decisões com retorno positivo | 52.62% | 61.72% |
| taxa de vitória do lote | 50.75% | 50.75% |

A taxa de vitória é a mesma porque as partidas são as mesmas. A recompensa não é.

## Resultados por execução

| Execução | Arq. | lr | Baseline | Buckets | Partidas | Melhor | Partidas até o melhor | Final | Em 12.7 M |
|---|---|---|---|---|---|---|---|---|---|
| bucket_heuristic | anterior | 0.001 | nenhum | heuristic | 25.3 M | **66.375%** | 21.6 M | 66.086% | 65.973% |
| bucket_heuristic_recent | anterior | 0.001 | nenhum | heuristic,recent | 18.2 M | **66.067%** | 13.1 M | 65.771% | 65.939% |
| baseline_zero | anterior | 0.001 | zero | heuristic,recent | 18.0 M | **66.238%** | 16.6 M | 65.855% | 65.683% |
| bucket_all | anterior | 0.001 | nenhum | heuristic,recent,medium_term,historical_uniform,champion_vs_heuristic,champion_vs_learner | 12.7 M | **65.861%** | 9.3 M | 65.793% | 65.793% |
| d6_maxwr_lr032 | atual | 0.032 | nenhum | heuristic | 28.4 M | **65.203%** | 27.8 M | 64.689% | 64.443% |
| d6_maxwr_lr016 | atual | 0.016 | nenhum | heuristic | 0.7 M | 63.603% | 0.5 M | 63.595% | — |
| default_lr032 | atual | 0.032 | nenhum | heuristic,recent | 20.6 M | **64.939%** | 12.0 M | 64.567% | 64.470% |
| default_lr016_lookup | atual | 0.016 | lookup-table | heuristic,recent | 15.4 M | **65.103%** | 15.4 M | 65.103% | 64.483% |

Todas as execuções partem da mesma política supervisionada, que vale 62.631% contra o aleatório. A execução `d6_maxwr_lr016` foi interrompida com 0,7 M de partidas e não entra em nenhuma comparação.

### Pares com o mesmo conjunto de oponentes

| Buckets | anterior | atual | Diferença |
|---|---|---|---|
| heuristic | bucket_heuristic: 66.375% (lr=0.001) | d6_maxwr_lr032: 65.203% (lr=0.032) | **-1.172 pp** |
| heuristic,recent | bucket_heuristic_recent: 66.067% (lr=0.001) | default_lr032: 64.939% (lr=0.032) | **-1.128 pp** |

## Fatores de confusão

Esta comparação **não é um experimento controlado da recompensa sozinha**, e vale registrar exatamente onde ela é frágil:

1. **A taxa de aprendizado difere.** As execuções antigas usaram lr = 0,001; as novas, 0,016 e 0,032. A grade `analysis/analise_lr_KL` mostrou, em `double-three` e sob a recompensa anterior, que taxas mais altas produziram jogadores **melhores** — o que faz a lr trabalhar contra a hipótese de que ela explique a queda, mas não a elimina, porque a grade foi feita em outro ruleset e sob a outra recompensa.
2. **O aumento de escala da recompensa muda o passo efetivo.** O desvio do retorno praticamente dobrou, o que multiplica o gradiente antes mesmo da lr. As novas execuções têm KL mediana ~4x maior e fração de recorte ~3,5x maior. Parte do efeito observado pode ser essa mudança de passo, não a forma da recompensa.
3. **Uma execução por configuração.** Não há repetição com sementes diferentes; a variação entre execuções não está medida.
4. **O corpus de recomputação vem de uma política só** — o checkpoint `double six 66p local.npz`, treinado sob a recompensa anterior, jogando contra o heurístico. As proporções de desfecho refletem essa política. As conclusões sobre *forma* e *escala* das duas funções não dependem disso; as proporções por classe de desfecho, sim.

## O experimento que decide

Uma única execução resolve a ambiguidade: recompensa atual, **lr = 0,001**, buckets `heuristic,recent`, semente 42, mesmos pesos supervisionados — ou seja, `bucket_heuristic_recent` com a única diferença sendo a recompensa.

```bash
python -u -m training.pipeline forever \
    --learning-rate 0.001 \
    --opponent-buckets heuristic,recent \
    --run-name recompensa_atual_lr001
```

Se essa execução ficar perto de 66%, a recompensa não é a causa e o problema está na taxa de aprendizado combinada com a nova escala. Se ficar perto de 65%, a recompensa é a causa.

Duas observações sobre como corrigir o desequilíbrio, caso ele se confirme como a causa:

- **Reduzir `reward_eta`.** O equilíbrio efetivo entre as metades é `eta * |G_local| / ((1 - eta) * |G_terminal|)`. Para recuperar com as magnitudes atuais o equilíbrio de 0.30x que a recompensa anterior tinha com eta = 0,5, seria preciso **reward_eta ≈ 0.112** — cerca de 11% em vez de 50%.
- **Reduzir `immediate_draw_weight` e `immediate_pass_weight` juntos** não funciona: a normalização por `max(a_D, a_P)` divide o par pelo seu maior membro, então só a *razão* entre eles é ajustável. A escala absoluta do termo local só se move por `reward_eta`. Isso é uma propriedade da arquitetura atual que vale registrar: **não existe hoje um controle direto da magnitude local**.

## Reprodução

```bash
/home/diego/CCO/amb_virtual/bin/python analysis/recompensa_anterior_vs_atual/analyze.py
```

O script relê os diretórios de execução e o corpus derivado e regenera as figuras, os CSVs, `analysis_summary.json` e este relatório. Nenhum diretório de execução, modelo ou dataset é escrito ou modificado.
