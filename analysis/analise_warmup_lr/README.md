# Análise da primeira corrida real com `--warmup-lr`

Verifica se o warmup de taxa de aprendizado controlado por KL funcionou como
especificado em `references/roteiros/WARMUP_LR_ROADMAP.md` e mede o que ele fez
com o treino. A corrida analisada é o ponto `warmup` de
`train_script/run_gpi12000_warmup_tests_diego_notebook.sh`: 5 h de tempo de
parede, `--warmup-lr` com os parâmetros padrão e todas as outras flags no padrão
do projeto (GPI 8000, lr 0,001, `turn-turn`, `batch-mean`), no commit `2c3e046`.

Corridas de referência, todas com semente 52, o mesmo checkpoint supervisionado
e o mesmo painel de 100.000 partidas do diagnóstico periódico:

| Chave | Diretório em `models/rl/` | Diferenças em relação ao warmup |
|---|---|---|
| `gpi12000` | `domino_rl_forever_seed52_runtest_gpi_12000_diego_notebook` | sem warmup, GPI 12000; mesmo código e mesmo orçamento |
| `gpi8000_old` | `domino_rl_forever_seed52_runone_factor_gpi_8000_diego_notebook` | sem warmup, lr 0,01, `decision-decision`, código anterior às otimizações de PPO |
| `lr0p001_old` | `domino_rl_forever_seed52_runlr_0p001` | sem warmup, GPI 2000, `decision-decision`, código antigo |

Fontes lidas (nada é escrito fora deste diretório):

- `warmup_schedule.jsonl` — uma linha por iteração com a lr aplicada, a lr
  instalada, expoente, EMA, sequência abaixo do limiar, cooldown e promoção;
- `training_metrics.jsonl` — KL, clip, entropia, épocas, tempos por iteração;
- histórico de diagnóstico periódico (`periodic_diagnostics.jsonl`);
- `checkpoint_archive/` — pesos a cada 10 iterações, para medir o passo real;
- `latest_weights.npz` + `latest.resume.npz` — estado salvo do otimizador e da agenda;
- `diagnostics/runtime_profile.json` e os logs da sequência em
  `train_script/grid_search_results/diego_notebook/gpi12000_warmup_test/`.

Comando, a partir da raiz do repositório:

```bash
/home/diego/CCO/amb_virtual/bin/python analysis/analise_warmup_lr/analyze.py
```

Regenera `REPORT.md`, as figuras e as tabelas:

| Arquivo | Conteúdo |
|---|---|
| `REPORT.md` | Conclusões, verificações e tabelas |
| `01_escada_lr.png` | lr aplicada por iteração, sequência abaixo do limiar e cooldown |
| `02_kl_e_ema.png` | max KL e EMA contra o limiar 0,0075 e `stop_kl` |
| `03_kl_por_degrau.png` | Distribuição da KL por degrau e ajuste KL × lr |
| `04_passo_dos_pesos.png` | Passo real nos pesos, comparado à corrida de lr fixo |
| `05_vitoria_por_partidas.png` | Vitórias contra random por partidas, com zoom na escada |
| `06_vitoria_por_tempo.png` | Vitórias por tempo de RL e por tempo de parede |
| `07_saude_do_ppo.png` | Clip, entropia, KL e norma do gradiente |
| `08_custo_do_tempo.png` | Divisão do orçamento de 5 h e custo por passo do otimizador |
| `verificacoes.csv` | As 16 verificações automáticas e seus detalhes |
| `trajetoria_warmup.csv` | Trace do warmup junto das métricas de cada iteração |
| `resumo_degraus.csv` | Estatísticas por degrau da escada |
| `curvas_vitoria.csv` | Curvas de vitória das quatro corridas, formato longo |
| `resumo_execucoes.csv`, `analysis_summary.json` | Números usados no relatório |
