# Dataset

Dados de treinamento gerados por `training/gerador.py` para o pipeline
supervisionado (SL). Todo o conteúdo além deste README é gerado e **não
versionado** (veja `.gitignore`) — regenere localmente quando precisar.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `dataset_2.jsonl` | Dataset JSON Lines gerado por `training/gerador.py`: uma linha por par (estado, ação) `{"estado": {...}, "acao_alvo": ...}` observado em partidas heurístico vs. heurístico. |

`estado` é o dicionário de `MotorDomino._obter_estado()` (sem o campo
`cadeia_visual`, que é metadado de renderização) e `acao_alvo` é a jogada
escolhida pelo heurístico naquele estado, no mesmo formato consumido por
`MotorDomino.step`.

## Como gerar

```bash
python -m training.gerador
```

Por padrão simula 10 000 partidas heurístico-vs-heurístico, o que costuma
gerar entre 150 000 e 200 000 pares (estado, ação). Ajuste `num_partidas` e
`arquivo_saida` em `training/gerador.py` para gerar um dataset maior/menor ou
em outro caminho.

## Como é consumido

`training/training_loop.py` lê este arquivo linha a linha, converte cada
estado para o vetor de 79 dimensões via `CodificadorDomino.encode_estado` e
cada ação para o índice de 58 posições, separa 85%/15% em treino/validação e
treina a `RedeNeuralSupervisionada`. O resultado é salvo em
`models/pesos_domino_sl.npz`.
