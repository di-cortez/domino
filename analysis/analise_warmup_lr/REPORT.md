# Primeira corrida real com `--warmup-lr`: funcionamento e efeito

## Conclusão

**O warmup funcionou exatamente como especificado.** 16 de 16 verificações passaram (tabela abaixo). A escada percorreu os sete degraus de 8.779e-05 a 0.001, com promoções nas iterações 100, 300, 500, 700, 900, 1100 — o mínimo teórico de 1100 iterações — e o replay offline de `WarmupSchedule` reproduz cada uma das 2903 decisões gravadas. O passo real dos pesos acompanha a lr de cada degrau, o que confirma que a taxa foi de fato aplicada ao otimizador, não apenas registrada. A corrida GPI 12000, interrompida por SIGTERM e retomada, manteve métricas e diagnósticos contíguos.

**Na lr nominal 0,001 o portão de KL nunca atuou.** A EMA máxima durante a escada foi 0.00167, cerca de 22% do limiar 0,0075, e a maior KL individual da corrida foi 0.00244. Isso confirma a previsão do roteiro (seção 2.1 e risco 6.1): em 0,001 não há pico inicial de KL para o warmup remover, então a escada rodou na velocidade mínima e o seu efeito é apenas treinar mais devagar durante as primeiras 1100 iterações (8.5 M de partidas com GPI 8000).

**Não há evidência de ganho de desempenho; o que aparece é o custo esperado da escada.** A corrida chegou a 66,391% de vitórias contra random (melhor 66,574%) em 22.3 M de partidas. A corrida GPI 12000 do mesmo código terminou em 66,557% (melhor 66,692%). As duas diferem em GPI *e* em warmup, e o intervalo de confiança de cada ponto é de ±0,29 pp, então a comparação não isola o warmup. Durante a escada a curva do warmup ficou sistematicamente abaixo das corridas de lr 0,001 fixo e recuperou a maior parte da distância depois. Medir esse custo sem confusão precisa de um controle com GPI 8000, lr 0,001 e `turn-turn` sem a flag.

## Verificações

| Verificação | Resultado | Detalhe |
|---|:---:|---|
| Trace contíguo | ✅ | 2903 linhas, iterações 1..2903; 2903 linhas de métricas; training_state 2903 |
| Configuração gravada | ✅ | warmup_lr=True, parâmetros {'exponent': 6, 'factor': 1.5, 'ema_alpha': 0.9, 'kl_threshold': 0.0075, 'hold_iterations': 100, 'cooldown_iterations': 100}, lr nominal 0.001 |
| Degraus exatos | ✅ | degraus observados: 8.77915e-05, 0.000131687, 0.000197531, 0.000296296, 0.000444444, 0.000666667, 0.001 |
| lr aplicada = lr instalada na iteração anterior | ✅ | applied_learning_rate[i+1] == next_learning_rate[i] em todas as iterações |
| Replay offline reproduz cada decisão | ✅ | 2903/2903 linhas idênticas ao replay de WarmupSchedule |
| Escada na velocidade mínima teórica | ✅ | promoções em [100, 300, 500, 700, 900, 1100]; EMA máxima durante a escada 0.00167 < limiar 0.0075 |
| Após a escada, lr nominal constante | ✅ | 1803 iterações após a iteração 1100 com lr 0.001 e agenda inativa |
| Trace coincide com training_metrics.jsonl | ✅ | maior diferença de max_approx_kl (métricas arredondadas a 5 casas): 0.00e+00 |
| Checkpoint de retomada | ✅ | otimizador lr 0.001; estado salvo {'cooldown': 100, 'ema': 0.0017280801901489338, 'exponent': 0, 'streak': 0} |
| PPO sem anomalias | ✅ | paradas por KL 0; épocas mínimas 16; iterações com gradiente recortado 0; buffer ['gpu']; KL ausente 0 |
| Avaliação em lotes de 4096 sem fallback de memória | ✅ | batch_size_memory_fallbacks = 0 |
| Diagnósticos e métricas contíguos (warmup) | ✅ | 224 pontos de 0 a 22,300,000 a cada 100 mil; 2903 iterações contíguas; janelas encadeadas: True |
| Diagnósticos e métricas contíguos (gpi12000) | ✅ | 250 pontos de 0 a 24,900,000 a cada 100 mil; 2249 iterações contíguas; janelas encadeadas: True |
| Log test_gpi_12000_diego_notebook.attempt1.log | ✅ | 0 erros; parada por SIGTERM com checkpoint seguro (sim); última contagem 2,900,000 partidas |
| Log test_gpi_12000_diego_notebook.attempt2.log | ✅ | 0 erros; parada por SIGTERM com checkpoint seguro (sim); última contagem 24,996,000 partidas |
| Log test_warmup_diego_notebook.attempt1.log | ✅ | 0 erros; parada por SIGTERM com checkpoint seguro (sim); última contagem 22,332,000 partidas |

Observação menor, sem efeito no treino: depois da última promoção o estado salvo no checkpoint mantém `cooldown` = 100, enquanto o trace registra `cooldown_remaining` = 0. A agenda inativa retorna antes de decrementar o cooldown, e uma agenda com expoente 0 nunca volta a consultá-lo; é só uma inconsistência de apresentação entre o trace e o `state_dict`.

## A escada, degrau por degrau

| Expoente | lr | Iterações | Início (M partidas) | KL mediana | KL p95 | KL máx. | EMA máx. | Clip mediano | Entropia mediana | lr ÷ nominal | passo ÷ passo lr fixo |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 8.779e-05 | 1–100 | 0.0 | 0.00125 | 0.00150 | 0.00169 | 0.00138 | 0.0132 | 0.2732 | 0.088 | 0.130 |
| 5 | 0.0001317 | 101–300 | 0.8 | 0.00131 | 0.00152 | 0.00180 | 0.00141 | 0.0141 | 0.2668 | 0.132 | 0.168 |
| 4 | 0.0001975 | 301–500 | 2.3 | 0.00137 | 0.00161 | 0.00184 | 0.00147 | 0.0149 | 0.2609 | 0.198 | 0.228 |
| 3 | 0.0002963 | 501–700 | 3.8 | 0.00139 | 0.00166 | 0.00189 | 0.00147 | 0.0148 | 0.2578 | 0.296 | 0.319 |
| 2 | 0.0004444 | 701–900 | 5.4 | 0.00145 | 0.00171 | 0.00201 | 0.00153 | 0.0160 | 0.2544 | 0.444 | 0.446 |
| 1 | 0.0006667 | 901–1100 | 6.9 | 0.00158 | 0.00183 | 0.00207 | 0.00167 | 0.0186 | 0.2518 | 0.667 | 0.628 |
| 0 | 0.001 | 1101–2903 | 8.5 | 0.00181 | 0.00206 | 0.00244 | 0.00196 | 0.0236 | 0.2334 | 1.000 | 0.874 |

A KL mediana cresce com a lr como KL ≈ 0.00438·lr^0.137 (R² = 0.904). Multiplicar a lr por 1,5 multiplica a KL por 1.06. O expoente é bem menor que o 0,726 medido na grade de lr do `double-three`, mas os degraus não são corridas independentes: cada degrau vem depois do anterior no mesmo treino, e a KL também muda com o avanço da política, então o ajuste mistura lr e fase do treino.

A última coluna mede se a taxa foi aplicada de verdade. Ela divide o deslocamento mediano dos pesos por iteração, medido entre checkpoints arquivados a cada 10 iterações, pelo da corrida GPI 12000 com lr 0,001 fixo nas mesmas iterações. Se a escada só fosse registrada, a razão ficaria constante; ela sobe degrau a degrau, de 0.13 a 0.87, acompanhando a razão de lr de 0.088 a 1 com expoente log-log 0.80. O crescimento é sublinear, e o degrau final fica abaixo de 1 porque a corrida de referência faz 1,5x mais passos do otimizador por iteração (GPI 12000); nenhuma das duas coisas muda a leitura, porque a razão sobe junto com cada promoção.

## Taxa de vitória contra random

Todas as corridas usam a semente 52, o mesmo checkpoint supervisionado e o mesmo painel de 100.000 partidas do diagnóstico periódico, por isso começam no mesmo ponto (62,580%). As outras diferenças de configuração estão na tabela.

| Corrida | Configuração | Partidas | Final | Melhor (em M) | Média até 8.5 M | Em 10 M | Média 12–17.6 M | Média até 17.6 M |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| warmup (GPI 8000, lr 0,001 com escada) | código atual, --warmup-lr, GPI 8000, lr 0,001, turn-turn | 22.3 M | 66,391% | 66,574% (21.1) | 65,248% | 65,972% | 66,164% | 65,684% |
| GPI 12000 (lr 0,001 fixo) | código atual, sem warmup, GPI 12000, lr 0,001, turn-turn | 25.0 M | 66,557% | 66,692% (22.3) | 65,805% | 66,372% | 66,357% | 66,079% |
| sweep gpi_8000 (lr 0,01, código antigo) | código antigo, sem warmup, GPI 8000, lr 0,01, decision-decision | 17.7 M | 65,788% | 66,002% (15.7) | 65,221% | 65,692% | 65,788% | 65,474% |
| lr_0p001 (GPI 2000, código antigo) | código antigo, sem warmup, GPI 2000, lr 0,001, decision-decision | 18.8 M | 66,334% | 66,397% (16.2) | 65,550% | 65,918% | 66,123% | 65,821% |

Durante a escada (até 8.5 M de partidas) a média do warmup foi 65,248%, contra 65,805% da GPI 12000 e 65,550% da `lr_0p001` antiga. Um ponto isolado do painel tem IC de ±0,29 pp, mas a curva do warmup fica abaixo das duas ao mesmo tempo em 78 dos 84 pontos do trecho (figura 05). Depois a distância diminui: entre 12 M e 17.6 M as médias são 66,164% (warmup), 66,357% (GPI 12000) e 66,123% (`lr_0p001`), ou seja, 0,19 pp atrás da GPI 12000 e empatado com a `lr_0p001`. É o padrão esperado do custo de treinar com lr reduzida no começo: aprendizado mais lento durante a escada e recuperação depois. Como as corridas diferem em GPI e em código, o tamanho desse custo não pode ser atribuído só ao warmup.

A entropia conta a mesma história (figura 07): a política do warmup permanece menos determinística por mais tempo. Entre 15 M e 17.6 M de partidas a entropia mediana foi 0.232 no warmup contra 0.220 na GPI 12000 e 0.221 na `lr_0p001`, com a curva do warmup deslocada para a direita, como se estivesse alguns milhões de partidas atrasada.

## Custo e desempenho do código novo

| Corrida | Commit | Iterações | Tempo de RL | Diagnóstico | Partidas/s de RL | Update/it | Rollout/it | ms/passo |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| warmup (GPI 8000, lr 0,001 com escada) | `2c3e046` | 2,903 | 2.74 h | 2.24 h | 2267 | 1.22 s | 2.06 s | 1.16 |
| GPI 12000 (lr 0,001 fixo) | `2c3e046` | 2,249 | 3.11 h | 1.87 h | 2236 | 1.92 s | 3.13 s | 1.22 |
| sweep gpi_8000 (lr 0,01, código antigo) | `066fa27` | 2,300 | 3.59 h | 1.39 h | 1370 | 2.75 s | 1.77 s | 2.61 |
| lr_0p001 (GPI 2000, código antigo) | `87b1ce1` | 9,380 | 3.79 h | 1.41 h | 1376 | 0.73 s | 0.64 s | 2.76 |

Os custos por iteração vêm das primeiras 1000 iterações de cada corrida. As últimas horas do ponto `gpi_8000` do sweep dividiram a máquina com os benchmarks de PPO de 12/09, e o custo por passo dele subiu de ~2,6 ms para mais de 6 ms nessa fase; o tempo total de RL e de diagnóstico dessa corrida carrega essa contaminação.

Na mesma GPI 8000, o update PPO custou 1.16 ms por passo do otimizador no código novo contra 2.61 ms na corrida do sweep (`066fa27`), 2.3x menos, já incluída a avaliação do buffer inteiro: a confirmação em produção das otimizações de PPO. O rollout, que as otimizações não tocaram, custou 2.06 s por iteração contra 1.77 s; a diferença acompanha o autotune de workers de rollout, que escolheu 8 workers na corrida nova e 10 na antiga. As duas corridas não usam a mesma lr, então a comparação vale para o custo, não para o aprendizado.

O sweep `gpi_8000` (lr 0,01, sem warmup) mostra na figura 07 exatamente o problema que o warmup foi feito para resolver: KL máxima perto de 0,014 nas primeiras iterações, fração no clip acima de 0,11 e 8 paradas por KL na corrida. Nenhuma das corridas com lr 0,001 tem esse pico.

O efeito colateral é que o diagnóstico periódico síncrono passou a ocupar 45% do orçamento de 5 h do warmup (2.24 h), porque o RL mais rápido chega a mais marcos de 100 mil partidas e cada marco para o treino por um diagnóstico de 100 mil partidas. `--async-periodic-diagnostics` existe exatamente para devolver esse tempo ao treino.

## Limites de interpretação

- Uma corrida por configuração, sem repetição de semente.
- Nenhuma corrida de referência difere do warmup em um único fator. A GPI 12000 difere na GPI; as corridas do sweep diferem em código, lr, GPI e modo de distância.
- O orçamento é de tempo de parede. Com o código novo o warmup jogou mais partidas que as corridas antigas no mesmo orçamento; as comparações por partidas são as mais justas para aprendizado.
- A `lr_0p001` antiga foi rodada fora do script de sequência, em duas sessões que somam 5,2 h, por isso passa da linha de 5 h na figura 08.
- A lr nominal 0,001 é justamente a faixa em que o roteiro previa que o warmup não teria o que corrigir. O teste que mede o benefício pretendido é o estágio 7 do roteiro: `--learning-rate 0.01` com e sem `--warmup-lr`.

## Próximos passos sugeridos

1. Rodar o controle sem warmup nas mesmas condições (GPI 8000, lr 0,001, código atual) para medir o custo real das 1100 iterações lentas.
2. Rodar o par do estágio 7 (`--learning-rate 0.01` com e sem `--warmup-lr`), onde o roteiro mede pico inicial de KL de 2,4x e o portão deve atuar.
3. Usar `--async-periodic-diagnostics` nas próximas corridas com orçamento de tempo, ou descontar o diagnóstico ao comparar.

## Artefatos

- `01_escada_lr.png` — lr aplicada, sequência abaixo do limiar e cooldown.
- `02_kl_e_ema.png` — max KL, EMA, limiar e `stop_kl`, durante a escada e na corrida inteira.
- `03_kl_por_degrau.png` — distribuição da KL por degrau e ajuste de potência.
- `04_passo_dos_pesos.png` — tamanho do passo nos pesos por iteração e normalizado pela lr.
- `05_vitoria_por_partidas.png` e `06_vitoria_por_tempo.png` — curvas de vitória contra random.
- `07_saude_do_ppo.png` — clip, entropia, KL e norma do gradiente ao longo das partidas.
- `08_custo_do_tempo.png` — divisão do orçamento de parede e custo por passo do otimizador.
- `trajetoria_warmup.csv`, `resumo_degraus.csv`, `curvas_vitoria.csv`, `resumo_execucoes.csv`, `verificacoes.csv`, `analysis_summary.json`.
