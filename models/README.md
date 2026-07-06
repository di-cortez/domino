# Models

Pesos das redes neurais treinadas. Todo o conteúdo além deste README é gerado
e **não versionado** (veja `.gitignore`) — regenere localmente ou treine do
zero conforme os pipelines em `training/`.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `pesos_domino_sl.npz` | Pesos da `RedeNeuralSupervisionada` (86→256→128→58) treinada por imitação do heurístico — a época salva é a de menor Val Custo da execução, não necessariamente a última. Gerado por `training_loop.py`; consumido por `AgenteNeuralNumPy` (opção `neural` no menu). |
| `pesos_domino_rl.npz` | Pesos da `RedeNeuralPolitica` (mesma arquitetura), refinados a partir do SL por `self_play.py` — por padrão via self-play puro contra um pool de snapshots congelados de si mesma, mantido só em memória durante o treino (`oponente_treino="self_play"`; o `AgenteEstrategico` não participa do treino, só das avaliações). Consumido por `AgenteRL` (opção `rl` no menu, sempre greedy). |

Os dois arquivos compartilham os seis arrays da política (`W1`, `b1`, `W2`,
`b2`, `W3`, `b3`) via `numpy.savez`, o que permite que um checkpoint de SL
seja carregado diretamente como ponto de partida da política de RL
(`RedeNeuralPolitica.carregar_de_sl`). O `pesos_domino_rl.npz` guarda ainda
mais dois arrays (`Wv`, `bv`) — a cabeça de valor (crítico) usada como
baseline do gradiente de política; checkpoints de SL não têm esses dois
arrays, e checkpoints de RL gerados antes dessa cabeça existir continuam
carregando normalmente (`Wv`/`bv` iniciam em zero nesse caso).

## Como (re)gerar

```bash
python -m training.gerador          # dataset/dataset_2.jsonl
python -m training.training_loop    # -> models/pesos_domino_sl.npz
python -m training.self_play        # -> models/pesos_domino_rl.npz (a partir do SL)
```

Para comparar checkpoints entre si ou contra os agentes de regra, use
`python -m diagnostico.avaliar --pesos <caminho.npz>` (veja `diagnostico/README.md`).

Para comparar especificamente os dois regimes de treino de RL
(`oponente_treino="self_play"` vs. `"heuristico"`), veja
`diagnostico/avalia_self-play/README.md` — espera checkpoints em
`pesos_domino_rl_self_play.npz` e `pesos_domino_rl_heuristico.npz`
(nomes configuráveis).
