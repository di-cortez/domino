# Análise da escada de GPI: 1k, 2k, 4k, 8k e 12k

Mede o efeito das partidas por iteração (GPI) na taxa de vitória contra o
oponente aleatório, no notebook do Diego, e estende os dois gráficos de
referência do orientador (`references/atualizacoes/atualizacoes_1209/GPI/`)
até 12.000. Os números desses gráficos são reproduzidos exatamente.

As corridas não formam uma escada única, e é isso que a análise trata:

| Chave | Diretório em `models/rl/` | Configuração |
|---|---|---|
| `gpi1000` | `run_one_factor_tests_diego_notebook/..._gpi_1000_diego_notebook` | sweep: lr 0,01, `decision-decision` |
| `gpi2000` | `run_one_factor_tests_diego_notebook/..._control_diego_notebook` | sweep (controle) |
| `gpi4000` | `run_one_factor_tests_diego_notebook/..._gpi_4000_diego_notebook` | sweep |
| `gpi8000` | `domino_rl_forever_seed52_runone_factor_gpi_8000_diego_notebook` | sweep |
| `gpi8000_nova` | `domino_rl_forever_seed52_runtest_control_diego_notebook` | commit `2c3e046`: lr 0,001, `turn-turn` |
| `gpi12000_nova` | `domino_rl_forever_seed52_runtest_gpi_12000_diego_notebook` | commit `2c3e046`: lr 0,001, `turn-turn` |

O código de treino é idêntico entre os commits das quatro corridas do sweep. A
de 12k só é comparada diretamente com `gpi8000_nova`, que difere dela apenas no
GPI. Para desenhá-la na escada antiga, a análise usa uma ponte
(`gpi8000 + (gpi12000_nova − gpi8000_nova)`) e testa a aditividade que essa
ponte supõe.

Comandos, a partir da raiz do repositório:

```bash
/home/diego/CCO/amb_virtual/bin/python analysis/analise_gpi/analyze.py
/home/diego/CCO/amb_virtual/bin/python analysis/analise_gpi/build_page.py
```

| Arquivo | Conteúdo |
|---|---|
| `analyze.py` | Lê as seis corridas e calcula janelas, ajustes, par direto, ponte e mecanismo do PPO |
| `build_page.py` | Preenche `pagina_modelo.html` com os números e os dados e grava `pagina.html` |
| `pagina.html` | Página autossuficiente enviada ao orientador |
| `dados_pagina.json` | Tudo o que a página desenha |
| `analysis_summary.json` | Os mesmos resultados sem as curvas, legível |
| `curvas_vitoria.csv` | Curvas brutas e média móvel de 5 pontos, formato longo |
| `resumo_gpi.csv` | Uma linha por corrida, com métricas de vitória e do PPO |
