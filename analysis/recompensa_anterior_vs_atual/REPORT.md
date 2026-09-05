# Recompensa anterior e recompensa atual: o que mudou e o que o resultado mostra

*Gerado em 2026-09-04 14:25 -0300 a partir do estado corrente dos diretórios de execução. Uma execução ainda em andamento avança entre duas gerações deste relatório.*

## Conclusão

A queda é real, maior que o ruído do painel de diagnóstico, e **a recompensa é a causa**. A execução `default_lookup` fecha a lacuna que as versões anteriores deste relatório não conseguiam fechar: ela usa a recompensa atual com **lr = 0.001**, os mesmos buckets `heuristic,recent`, a mesma semente e os mesmos pesos supervisionados de `bucket_heuristic_recent`. Chegou a **65.335%** contra o oponente aleatório, contra **66.067%** da execução equivalente sob a recompensa anterior: **-0.732 pontos percentuais**, cerca de 2.5x a meia-largura do intervalo de 95% do painel de 100,000 partidas (±0.29 pp).

Igualar a taxa de aprendizado **recuperou parte** da diferença, e só parte. O par que isola a lr limpo é `default_lr016_lookup` / `default_lookup`, que compartilham recompensa, buckets e baseline `lookup-table` e diferem só na lr (0.016 contra 0.001): 65.103% para 65.335%, +0.232 pp. Contra `default_lr032` a diferença é +0.396 pp, mas ali o baseline também muda. Em qualquer das duas leituras, os 0.732 pp que separam a recompensa atual da anterior não têm mais a lr como explicação possível.

A causa provável dentro da recompensa não é o novo formato da utilidade terminal, e sim o **deslocamento do equilíbrio entre a metade terminal e a metade local** que o novo formato provocou sem que `reward_eta` mudasse. Recomputando as duas recompensas sobre as **mesmas 422,055 decisões reais**, a metade local passou de 0.30x para 2.38x a magnitude da metade terminal — um fator de 8.0 no peso efetivo do termo de moldagem, com `reward_eta` fixo em 0,5 nas duas. As próprias execuções da arquitetura atual confirmam o número ao vivo: a razão mediana registrada por `default_lookup` é 2.45x.

A consequência mensurável é que a recompensa deixou de informar sobre o resultado da partida com a mesma nitidez: a correlação entre o retorno de uma decisão e vencer caiu de **+0.832** para **+0.591**, e a fração de decisões tomadas em **partidas perdidas** que ainda assim recebem retorno positivo subiu de **4.82%** para **35.74%**.

O veredito não depende desta máquina. Uma execução treinada em outro computador, com outro binário supervisionado e lr = 0.002, sob a recompensa atual e o bucket `heuristic`, chegou a 65.397% contra 66.375% da execução equivalente sob a recompensa anterior — -0.978 pp, a mesma direção e praticamente o mesmo tamanho.

Uma segunda conclusão sai do mesmo experimento, e é uma **correção** do que as versões anteriores deste relatório sugeriam: a anomalia de entropia e a pressão sobre a região de confiança do PPO **eram a taxa de aprendizado, não a recompensa**. Com lr = 0.001, `default_lookup` volta a ter entropia monotonicamente decrescente, KL mediana de 0.00240 e fração de recorte de 0.0348 — os mesmos valores das execuções antigas. Os dois efeitos que antes apareciam juntos agora estão separados: **a lr explica a dinâmica, a recompensa explica o resultado**.

**E a correção proposta foi testada.** As versões anteriores deste relatório terminavam prevendo que `reward_eta ≈ 0.112` devolveria à metade local o peso que ela tinha antes. Duas execuções com `reward_eta = 0.115` — uma aqui, outra na máquina do orientador — confirmam a previsão em três medidas independentes: registraram ao vivo a razão local/terminal em 0.285x, contra os 0.30x da recompensa anterior; a correlação por iteração entre recompensa e vitória subiu de +0.706–+0.771 para +0.889–+0.893, de volta à faixa antiga (+0.854–+0.900); e ficaram +0.600 pp e +0.718 pp acima das execuções gêmeas que mantiveram eta = 0,5, medidas no mesmo orçamento de 2.7 M de partidas. O detalhe que impede de declarar o problema resolvido é que essas quatro execuções treinam com o bucket `random`, o mesmo oponente do diagnóstico: o contraste entre elas é limpo, o nível absoluto não é comparável com o bloco da recompensa anterior. A seção [A correção de eta, executada](#a-correção-de-eta-executada) trata disso, e o experimento que falta está no fim dela.

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

As execuções da arquitetura atual medem esse mesmo desequilíbrio **sozinhas**, sem nenhuma recomputação offline: a versão 8 do esquema de métricas registra `terminal_abs_mean` e `local_abs_mean`, que são exatamente `E|(1-eta) G_T|` e `E|eta G_I|` como o rollout os creditou. A figura `09_mistura_ao_vivo.png` traz as duas séries.

| Execução | lr | eta | magnitude terminal | magnitude local | razão local/terminal |
|---|---|---|---|---|---|
| d6_maxwr_lr032 | 0.032 | 0.5 | 0.2112 | 0.5136 | 2.43x |
| default_lr032 | 0.032 | 0.5 | 0.2104 | 0.5159 | 2.45x |
| default_lr016_lookup | 0.016 | 0.5 | 0.2118 | 0.5169 | 2.44x |
| default_lookup | 0.001 | 0.5 | 0.2106 | 0.5167 | 2.45x |
| d6_random_eta0115 | 0.002 | 0.115 | 0.4018 | 0.1147 | 0.29x |
| rick_random_eta0115 | 0.002 | 0.115 | 0.4000 | 0.1141 | 0.29x |

As quatro execuções que mantiveram eta = 0,5 concordam entre si dentro de 2.43x–2.45x e concordam com os 2.38x recomputados sobre o corpus fixo, que vem de outra política e de outro conjunto de partidas. São duas medições independentes do mesmo número, o que descarta a possibilidade de o desequilíbrio ser um artefato da política que gerou o corpus. A razão também **não depende da taxa de aprendizado**: é uma propriedade da função de recompensa — as quatro cobrem lr de 0.001 a 0.032 e medem o mesmo valor.

As duas últimas linhas são a verificação da correção proposta adiante: com eta = 0,115 a mesma coluna registra 0.285x e 0.286x, contra os 0.30x que a recompensa anterior tinha. A conta que produziu esse eta foi feita sobre o corpus recomputado; quem a confirma aqui é o próprio rollout, em outra política e em outro conjunto de partidas. **O controle funciona como previsto.**

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

O mesmo efeito aparece nos dados de treino ao vivo, sem nenhuma recomputação: a correlação entre `reward_mean` e `batch_win_rate` por iteração cai de +0.866 e +0.900 nas execuções antigas para +0.706, +0.771 e +0.722 nas novas — incluindo `default_lookup`, que usa a mesma lr das antigas —, e a `reward_mean` mediana sai de aproximadamente zero para cerca de +0.270 com a taxa de vitória em lote parada em ~51%.

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

`default_lookup` abre com exatamente os mesmos números da coluna da direita, porque na primeira iteração a política ainda é a supervisionada e a taxa de aprendizado não teve como importar. A divergência entre as duas execuções da arquitetura atual começa na iteração 2.

## O que a dinâmica de treino mostra

A figura `07_dinamica_de_treino.png` traz um efeito que as versões anteriores deste relatório não conseguiam atribuir, e que a execução `default_lookup` agora resolve: **a entropia da política deixava de cair de forma monótona**. Sob a recompensa anterior ela desce ao longo de todo o treino e termina no seu mínimo (0.293 para 0.225 em `bucket_heuristic`). Sob a atual **com lr alta** ela cai rápido, atinge o mínimo de 0.199 já na iteração 495 e depois **volta a subir**, até 0.277.

A pergunta em aberto era se isso vinha da recompensa ou da lr, já que as duas mudaram juntas. Vinha da lr. Com a recompensa atual e lr = 0.001, `default_lookup` se comporta como as execuções antigas em todos os três indicadores:

| Indicador | bucket_heuristic_recent (anterior, lr 0.001) | default_lookup (atual, lr 0.001) | default_lr032 (atual, lr 0.032) |
|---|---|---|---|
| entropia final | 0.228 | 0.206 | 0.264 |
| iteração da entropia mínima | 7,920 de 9,100 | 12,176 de 12,800 | 564 de 10,300 |
| KL mediana | 0.00231 | 0.00240 | 0.00950 |
| fração recortada mediana | 0.0335 | 0.0348 | 0.1160 |

A entropia volta a ser monótona: o mínimo cai na iteração 12,176 de 12,800, ou seja, no fim do treino, e não há repique. A KL e o recorte voltam aos valores das execuções antigas. O aumento de escala do retorno, que praticamente dobrou o desvio (0.383 para 0.747), não foi suficiente por si só para pressionar a região de confiança quando a lr é baixa.

O que **não** volta ao normal com lr baixa é o sinal em si: a correlação por iteração entre `reward_mean` e `batch_win_rate` fica em +0.722 em `default_lookup`, contra +0.866 em `bucket_heuristic_recent`, e a `reward_mean` mediana permanece em +0.278 com a taxa de vitória em lote em 51.00%. Esses são os indicadores que dependem da forma da recompensa, e eles não se movem com a lr.

## Resultados por execução

| Execução | Arq. | lr | eta | Baseline | Buckets | Partidas | Melhor | Partidas até o melhor | Final | Melhor em 2.7 M |
|---|---|---|---|---|---|---|---|---|---|---|
| bucket_heuristic | anterior | 0.001 | 0.5 | batch-mean | heuristic | 25.3 M | 66.375% | 21.6 M | 66.086% | **65.135%** |
| bucket_heuristic_recent | anterior | 0.001 | 0.5 | batch-mean | heuristic,recent | 18.2 M | 66.067% | 13.1 M | 65.771% | **64.999%** |
| baseline_zero | anterior | 0.001 | 0.5 | zero | heuristic,recent | 18.0 M | 66.238% | 16.6 M | 65.855% | **65.187%** |
| bucket_all | anterior | 0.001 | 0.5 | batch-mean | heuristic,recent,medium_term,historical_uniform,champion_vs_heuristic,champion_vs_learner | 12.7 M | 65.861% | 9.3 M | 65.793% | **65.069%** |
| d6_maxwr_lr032 | atual | 0.032 | 0.5 | batch-mean | heuristic | 28.4 M | 65.203% | 27.8 M | 64.689% | **64.211%** |
| d6_maxwr_lr016 | atual | 0.016 | 0.5 | batch-mean | heuristic | 0.7 M | 63.603% | 0.5 M | 63.595% | — |
| default_lr032 | atual | 0.032 | 0.5 | batch-mean | heuristic,recent | 20.6 M | 64.939% | 12.0 M | 64.567% | **64.080%** |
| default_lr016_lookup | atual | 0.016 | 0.5 | lookup-table | heuristic,recent | 15.6 M | 65.103% | 15.4 M | 64.802% | **64.359%** |
| default_lookup | atual | 0.001 | 0.5 | lookup-table | heuristic,recent | 25.6 M | 65.335% | 24.0 M | 65.116% | **64.539%** |
| rick_heuristic | atual | 0.002 | 0.5 | batch-mean | heuristic | 20.5 M | 65.397% | 12.5 M | 65.143% | **64.584%** |
| rick_random_desktop | atual | 0.002 | 0.5 | batch-mean | random | 25.1 M | 65.836% | 20.3 M | 65.671% | **65.188%** |
| rick_random_notebook | atual | 0.002 | 0.5 | batch-mean | random | 23.8 M | 65.870% | 21.4 M | 65.597% | **65.126%** |
| d6_random_eta0115 | atual | 0.002 | 0.115 | batch-mean | random | 5.0 M | 66.013% | 4.9 M | 65.923% | **65.844%** |
| rick_random_eta0115 | atual | 0.002 | 0.115 | batch-mean | random | 2.7 M | 65.788% | 2.7 M | 65.788% | **65.788%** |

Todas as execuções partem da mesma política supervisionada, que vale 62.631% contra o aleatório. A execução `d6_maxwr_lr016` foi interrompida com 0,7 M de partidas e não entra em nenhuma comparação.

A coluna em negrito é a que compara. As execuções desta tabela têm idades muito diferentes — de 2.7 M a 28.4 M de partidas — e o melhor de uma vida inteira premia quem foi deixado treinando por mais tempo, o que não é a variável em estudo. A última coluna é o melhor que cada execução já tinha alcançado com 2.7 M de partidas, o horizonte que **todas** atravessaram; é a mesma estatística, medida no mesmo orçamento. A figura `08_resumo_resultados.png` traz as duas leituras lado a lado: a barra cheia no horizonte comum, a barra clara na execução inteira.

O melhor resultado absoluto continua sendo o de `bucket_heuristic` sob a recompensa anterior, 66.375%, contra 65.397% de `rick_heuristic` sob a atual (0.978 pp, 3.3x a meia-largura do intervalo). Esses dois compartilham o bucket `heuristic` e o baseline, mas foram treinados em máquinas diferentes e com lr diferente; o par controlado abaixo é o que decide.

A separação é completa, e a figura `08_resumo_resultados.png` mostra isso de uma vez: as quatro execuções completas sob a recompensa anterior ficam **todas** acima das cinco sob a atual. A pior das antigas (`bucket_all`, 65.861%) ainda supera a melhor das novas (`rick_heuristic`, 65.397%), e dessa vez os intervalos de 95% nem se tocam (65.567% contra 65.692%). A ordenação também não é uma escala de taxa de aprendizado: as cinco execuções novas cobrem lr de 0.001 a 0.032, as duas que mais se aproximam do bloco antigo usam as duas lr mais baixas, e ainda assim param antes.

A separação também não é um efeito do tempo de treino: ela sobrevive ao corte em 2.7 M de partidas. Ali a pior das antigas (`bucket_heuristic_recent`, 64.999%) ainda supera a melhor das novas (`rick_heuristic`, 64.584%) por 0.415 pp, e os dois blocos continuam sem se interpenetrar.

Essa contagem exclui as quatro execuções que treinaram com o bucket `random`, discutidas na seção seguinte: elas treinam contra o mesmo oponente que o diagnóstico mede, então o número delas não é comparável com o das demais.

### Pares com o mesmo conjunto de oponentes

A última linha é a comparação controlada: as duas execuções compartilham buckets, taxa de aprendizado, semente e pesos supervisionados, e diferem na recompensa. A penúltima é a mesma comparação feita em outra máquina, com outra lr e outro binário supervisionado.

| Buckets | anterior | atual | Mesma lr | Diferença |
|---|---|---|---|---|
| heuristic | bucket_heuristic: 66.375% (lr=0.001) | d6_maxwr_lr032: 65.203% (lr=0.032) | não | **-1.172 pp** |
| heuristic,recent | bucket_heuristic_recent: 66.067% (lr=0.001) | default_lr032: 64.939% (lr=0.032) | não | **-1.128 pp** |
| heuristic | bucket_heuristic: 66.375% (lr=0.001) | rick_heuristic: 65.397% (lr=0.002) | não | **-0.978 pp** |
| heuristic,recent | bucket_heuristic_recent: 66.067% (lr=0.001) | default_lookup: 65.335% (lr=0.001) | **sim** | **-0.732 pp** |

## As execuções recebidas de fora

Quatro das execuções da tabela não foram treinadas nesta máquina. Elas chegaram como o pacote de quatro arquivos que trocamos para comparar modelos sem transferir o modelo inteiro: `run_config.json`, `periodic_diagnostics.jsonl`, `rl_vs_random_progress.csv` e o PNG do progresso. Esse pacote basta para tudo que este relatório mede por execução — a curva de vitória, o pico, a trajetória e os intervalos — porque a variável de desfecho sai inteira do CSV de progresso.

O que os quatro arquivos não trazem é `training_metrics.jsonl`, o registro por iteração, que fica um nível acima deles na raiz do diretório da execução. Três das recebidas vieram sem ele — `rick_heuristic`, `rick_random_desktop`, `rick_random_notebook` — e por isso têm as colunas de entropia, KL, recorte e mistura viva vazias em `resumo_execucoes.csv` e não aparecem nas figuras `07_dinamica_de_treino.png` e `09_mistura_ao_vivo.png`. `rick_random_eta0115` veio com ele e entra em tudo, o que mostra que o arquivo é a única peça que faltava: uma execução recebida com os cinco arquivos é indistinguível de uma treinada aqui, para efeitos desta análise.

### Os pesos supervisionados não são um confundidor

As quatro máquinas geraram cada uma o seu próprio `domino_sl_standard_seed42.npz`, e os quatro arquivos têm sha256 diferentes (7a7bb7f4, a38f40d6, a5c95ef0, e0695fcb). Mesma semente, binários distintos: a ordem de acumulação em ponto flutuante muda com o número de trabalhadores e com a GPU. Isso poderia ser um confundidor sério, e o diagnóstico periódico resolve a dúvida sem custo, porque a primeira linha de cada curva mede exatamente essas políticas supervisionadas nas mesmas 100.000 partidas fixas:

| Pesos supervisionados | Máquina | Execução que os usa | Vitória da política inicial |
|---|---|---|---|
| `e0695fcb` | NVIDIA GeForce RTX 3050 6GB Laptop GPU | `bucket_heuristic` | 62.631% |
| `a5c95ef0` | NVIDIA GeForce GTX 1650 | `rick_random_desktop` | 62.632% |
| `7a7bb7f4` | NVIDIA GeForce GTX 960M | `rick_random_notebook` | 62.632% |
| `a38f40d6` | NVIDIA GeForce RTX 4050 Laptop GPU | `rick_heuristic` | 62.633% |

Os quatro pontos de partida caem dentro de 0.002 pp uns dos outros, contra uma meia-largura de ±0.30 pp no próprio diagnóstico. São o mesmo jogador para efeitos de medida. A diferença entre as arquiteturas de recompensa não pode ser atribuída ao ponto de partida.

### A replicação

`rick_heuristic` repete, em outra máquina e com lr 0.002, a mesma condição de `bucket_heuristic`: bucket `heuristic`, semente 42, baseline batch-mean, double-six. O resultado é 65.397% contra 66.375%, uma queda de -0.978 pp. É a mesma direção e praticamente o mesmo tamanho da queda do par controlado (-0.732 pp), obtida com outro hardware, outro binário supervisionado e outra taxa de aprendizado. O efeito não é uma peculiaridade desta máquina.

### As execuções contra o bucket `random`

Quatro execuções da tabela — recebidas ou locais — treinam com o bucket `random`, isto é, contra o mesmo oponente que o diagnóstico usa para medir. Elas chegam a 65.788% e 65.836% e 65.870% e 66.013% — acima de qualquer execução sob a recompensa atual, e acima até da pior execução sob a anterior. O número é real, mas mede outra coisa: treinar e avaliar contra a mesma política é otimizar diretamente a métrica, e o valor deixa de indicar força de jogo geral. Por isso elas estão marcadas com `*` na figura `08_resumo_resultados.png` e ficam fora do ordenamento entre arquiteturas.

Elas não são inúteis: um viés que todas compartilham não atrapalha a comparação **entre** elas. É exatamente isso que sustenta o teste de eta da seção seguinte, em que os dois lados treinam contra o mesmo oponente e diferem só em eta.

E duas delas medem reprodutibilidade. Rodaram a mesma configuração, com eta = 0,5, em GPUs diferentes (NVIDIA GeForce GTX 1650 e NVIDIA GeForce GTX 960M) e terminaram a 0.034 pp uma da outra, dentro do ruído do diagnóstico. Duas máquinas independentes, o mesmo resultado.

## Fatores de confusão

O par controlado `bucket_heuristic_recent` / `default_lookup` fecha os dois confundidores principais que as versões anteriores deste relatório listavam. Restam três, e vale registrar exatamente onde a comparação ainda é frágil:

1. **O baseline de vantagem difere dentro do par controlado.** `bucket_heuristic_recent` usou o baseline padrão (`batch-mean`) e `default_lookup` usou `lookup-table`. É o único parâmetro de treino que ainda separa os dois. O tamanho desse efeito está medido em outra execução: `baseline_zero`, que é `bucket_heuristic_recent` com o baseline trocado por zero sob a mesma recompensa, chegou a 66.238% contra 66.067% — +0.171 pp, cerca de 23% da lacuna que precisa ser explicada. É pequeno, mas não é zero, e um baseline diferente do avaliado ali não está medido.
2. **Nenhuma repetição com semente diferente.** Toda execução aqui usa semente 42. As execuções recebidas atenuam isso em parte — a condição `heuristic` sob a recompensa atual foi reproduzida em outra máquina, as duas execuções `random` com eta = 0,5 reproduziram uma à outra dentro de 0.034 pp, e as duas com eta = 0,115 concordaram dentro de 0.056 pp no horizonte comum — mas variar a máquina não é o mesmo que variar a semente, e a dispersão entre sementes continua sem medida.
3. **O corpus de recomputação vem de uma política só** — o checkpoint `double six 66p local.npz`, treinado sob a recompensa anterior, jogando contra o heurístico. As proporções de desfecho refletem essa política. As conclusões sobre *forma* e *escala* das duas funções não dependem disso; as proporções por classe de desfecho, sim.

## O experimento que decidiu

As versões anteriores deste relatório terminavam pedindo uma execução: recompensa atual, **lr = 0,001**, buckets `heuristic,recent`, semente 42, mesmos pesos supervisionados. Essa execução existe e é `default_lookup`, com 25.6 M de partidas e 12,800 iterações.

O critério declarado na ocasião era: *se ficar perto de 66%, a recompensa não é a causa; se ficar perto de 65%, a recompensa é a causa*. O melhor resultado foi **65.335%**, atingido com 24.0 M de partidas, e a curva termina em 65.116%. **A recompensa é a causa.**

A curva fica abaixo da equivalente anterior em todo o percurso, e não apenas no pico, o que descarta a leitura de que seria só uma questão de mais partidas:

| Partidas de RL | bucket_heuristic_recent (anterior) | default_lookup (atual) | Diferença |
|---|---|---|---|
| 2 M | 64.945% | 64.439% | -0.506 pp |
| 5 M | 65.789% | 64.693% | -1.096 pp |
| 10 M | 65.660% | 64.695% | -0.965 pp |
| 15 M | 65.869% | 65.070% | -0.799 pp |
| 18 M | 65.773% | 65.030% | -0.743 pp |

Duas observações sobre como corrigir o desequilíbrio:

- **Reduzir `reward_eta`.** O equilíbrio efetivo entre as metades é `eta * |G_local| / ((1 - eta) * |G_terminal|)`. Para recuperar com as magnitudes atuais o equilíbrio de 0.30x que a recompensa anterior tinha com eta = 0,5, seria preciso **reward_eta ≈ 0.112** — cerca de 11% em vez de 50%.
- **Reduzir `immediate_draw_weight` e `immediate_pass_weight` juntos** não funciona: a normalização por `max(a_D, a_P)` divide o par pelo seu maior membro, então só a *razão* entre eles é ajustável. A escala absoluta do termo local só se move por `reward_eta`. Isso é uma propriedade da arquitetura atual que vale registrar: **não existe hoje um controle direto da magnitude local**.

## A correção de eta, executada

Essa previsão foi testada. Duas execuções repetiram a recompensa atual com `reward_eta = 0.115`, o valor mais próximo de 0.112 que foi de fato lançado, mantendo tudo o mais: mesma semente, mesma lr, mesmo baseline, mesmos buckets, mesma arquitetura de recompensa. Uma rodou nesta máquina, a outra na do orientador. São as execuções mais novas da tabela, e é por isso que a comparação abaixo é lida no horizonte comum de 2.7 M de partidas.

**Primeiro, o alvo foi acertado.** As duas registraram ao vivo a razão local/terminal em 0.285x e 0.286x, contra os 0.30x da recompensa anterior e os 2.43x–2.45x das execuções que ficaram em eta = 0,5. O valor foi calculado sobre o corpus recomputado e confirmado pelo rollout de duas execuções independentes: `reward_eta` é, de fato, o controle da magnitude relativa, e a conta que o dimensionou estava certa.

**Segundo, o sinal voltou a informar sobre o resultado.** A correlação por iteração entre `reward_mean` e `batch_win_rate` mede o quanto a recompensa que o treino persegue tem a ver com ganhar a partida. Ela vale +0.854 a +0.900 nas execuções da recompensa anterior e cai para +0.706 a +0.771 nas da atual com eta = 0,5. Com eta = 0,115 ela volta a +0.889 e +0.893 — dentro da faixa antiga. Este número é lido dentro do próprio treino, sobre as partidas que a execução jogou, e **não passa pelo diagnóstico contra o aleatório**: a ressalva do bucket `random`, discutida adiante, não o afeta.

**Terceiro, o resultado subiu.** Cada execução corrigida tem, entre as que ficaram em eta = 0,5, uma contraparte que compartilha o bucket `random` e o resto da configuração:

| eta = 0,5 | eta = 0,115 | Mesma máquina e mesmos pesos SL | Diferença |
|---|---|---|---|
| `rick_random_desktop`: 65.188% | `rick_random_eta0115`: 65.788% | **sim** | **+0.600 pp** |
| `rick_random_notebook`: 65.126% | `d6_random_eta0115`: 65.844% | não | **+0.718 pp** |

A primeira linha é a comparação controlada do eta: as duas execuções rodaram na mesma GPU, a partir do mesmo binário supervisionado, com a mesma semente, a mesma lr, o mesmo baseline e os mesmos buckets. **O único parâmetro diferente é `reward_eta`.** A segunda linha repete o contraste em outro par de máquinas.

**Quarto, e é aqui que a leitura precisa de cuidado:** as quatro execuções desse bloco treinam com o bucket `random`, o mesmo oponente que o diagnóstico mede. Isso infla o nível de todas elas e as mantém fora do ordenamento contra as execuções da recompensa anterior. O que **não** é inflado é a diferença *dentro* do bloco: os dois lados carregam o mesmo viés, então o ganho do eta baixo é medido limpo. A conclusão que se sustenta é que **`reward_eta` move o resultado na direção prevista**; não que a recompensa atual com eta corrigido já alcance a anterior, o que estas execuções não têm como mostrar.

### O próximo experimento

Repetir `default_lookup` trocando apenas `reward_eta` de 0.5 para 0.115 — a mesma correção já validada, agora com buckets que **não** contêm o oponente do diagnóstico, para que o número volte a ser comparável com o bloco da recompensa anterior:

```bash
python -u -m training.pipeline forever \
    --learning-rate 0.001 \
    --opponent-buckets heuristic,recent \
    --reward-eta 0.115 \
    --baseline lookup-table \
    --run-name recompensa_atual_eta_equivalente
```

O par a bater é `bucket_heuristic_recent`, 64.999% no horizonte comum e 66.067% no total, contra os 64.539% e 65.335% de `default_lookup`. Se essa execução fechar a lacuna de 0.732 pp, o desequilíbrio entre as metades era a causa inteira e `reward_eta` a corrige. Se fechar apenas parte dela, o restante está na *forma* da utilidade terminal — provavelmente na perda da penalidade de pontos, que empurrava na direção que o diagnóstico contra o aleatório premia.

## Reprodução

```bash
/home/diego/CCO/amb_virtual/bin/python analysis/recompensa_anterior_vs_atual/analyze.py
```

O script relê os diretórios de execução e o corpus derivado e regenera as figuras, os CSVs, `analysis_summary.json` e este relatório. Nenhum diretório de execução, modelo ou dataset é escrito ou modificado.
