# Models

Pesos das redes neurais treinadas. Todo o conteúdo além deste README é gerado
e **não versionado** (veja `.gitignore`) — regenere localmente ou treine do
zero conforme os pipelines em `training/`.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `pesos_domino_sl.npz` | Pesos da `RedeNeuralSupervisionada` (79→256→128→58) treinada por imitação do heurístico. Gerado por `training_loop.py`; consumido por `AgenteNeuralNumPy` (opção `neural` no menu). |
| `pesos_domino_rl.npz` | Pesos da `RedeNeuralPolitica` (mesma arquitetura), refinados por self-play a partir do SL. Gerado por `self_play.py`; consumido por `AgenteRL` (opção `rl` no menu, sempre greedy). |

Ambos os arquivos guardam os mesmos seis arrays (`W1`, `b1`, `W2`, `b2`, `W3`,
`b3`) via `numpy.savez`, o que permite que um checkpoint de SL seja carregado
diretamente como ponto de partida da política de RL (`RedeNeuralPolitica.carregar_de_sl`).

## Como (re)gerar

```bash
python -m training.gerador          # dataset/dataset_2.jsonl
python -m training.training_loop    # -> models/pesos_domino_sl.npz
python -m training.self_play        # -> models/pesos_domino_rl.npz (a partir do SL)
```

Para comparar checkpoints entre si ou contra os agentes de regra, use
`python -m diagnostico.avaliar --pesos <caminho.npz>` (veja `diagnostico/README.md`).
