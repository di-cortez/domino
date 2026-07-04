# Treinamento

Os dois pipelines de treinamento do projeto — supervisionado (SL, imitação do
heurístico) e por reforço (RL, self-play) — moram aqui. Ver a seção
**Visão geral da arquitetura** no `README.md` da raiz para os diagramas de
cada pipeline.

## Arquivos

| Arquivo | Responsabilidade |
|---|---|
| `gerador.py` | `gerar_dataset(...)` simula partidas heurístico vs. heurístico via `GerenciadorPartida` e grava cada par (estado, ação) como uma linha JSON em `dataset/dataset_2.jsonl`. Executar com `python -m training.gerador`. |
| `training_loop.py` | Lê o dataset JSONL, codifica com `CodificadorDomino`, separa 85%/15% treino/validação, treina a `RedeNeuralSupervisionada` e salva `models/pesos_domino_sl.npz`. Executar com `python -m training.training_loop`. |
| `self_play.py` | Loop de RL por self-play (`treinar`). Currículo misto self-play/vs. heurístico, REINFORCE + baseline + entropia, checkpoints periódicos e avaliação contra o heurístico. Executar com `python -m training.self_play`. |
| `__init__.py` | Marca `training` como pacote Python; sem conteúdo. |

## Ordem recomendada de execução

```bash
python -m training.gerador          # 1. gera dataset/dataset_2.jsonl
python -m training.training_loop    # 2. treina e salva models/pesos_domino_sl.npz
python -m training.self_play        # 3. refina por self-play -> models/pesos_domino_rl.npz
```

Para medir se o RL está de fato melhorando em relação ao SL/heurístico, use o
módulo de avaliação em `diagnostico/` (veja `diagnostico/README.md`).
