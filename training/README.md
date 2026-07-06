# Treinamento

Os dois pipelines de treinamento do projeto — supervisionado (SL, imitação do
heurístico) e por reforço (RL, self-play) — moram aqui. Ver a seção
**Visão geral da arquitetura** no `README.md` da raiz para os diagramas de
cada pipeline.

## Arquivos

| Arquivo | Responsabilidade |
|---|---|
| `gerador.py` | `gerar_dataset(...)` simula partidas heurístico vs. heurístico via `GerenciadorPartida` e grava cada par (estado, ação) como uma linha JSON em `dataset/dataset_2.jsonl`. Executar com `python -m training.gerador`. |
| `training_loop.py` | Lê o dataset JSONL, codifica com `CodificadorDomino`, separa 85%/15% treino/validação, treina a `RedeNeuralSupervisionada` e salva `models/pesos_domino_sl.npz` com os pesos da época de **menor Val Custo** da execução (guardados em memória a cada validação via o callback `ao_validar` de `RedeNeuralSupervisionada.treinar`), não os da última época. Executar com `python -m training.training_loop`. |
| `self_play.py` | Loop de RL por REINFORCE + ator-crítico (`treinar`), com o oponente de treino selecionável por `oponente_treino` (ou a constante `OPONENTE_TREINO` no topo do arquivo). Executar com `python -m training.self_play`. |
| `__init__.py` | Marca `training` como pacote Python; sem conteúdo. |

## Ordem recomendada de execução

```bash
python -m training.gerador          # 1. gera dataset/dataset_2.jsonl
python -m training.training_loop    # 2. treina e salva models/pesos_domino_sl.npz
python -m training.self_play        # 3. refina por self-play -> models/pesos_domino_rl.npz
```

Para medir se o RL está de fato melhorando em relação ao SL/heurístico, use o
módulo de avaliação em `diagnostico/` (veja `diagnostico/README.md`).

## Oponente de treino (`oponente_treino` / `OPONENTE_TREINO`)

`self_play.py` tem uma constante de configuração no topo do arquivo, no
mesmo espírito do `AGENTE`/`OPONENTE` de `diagnostico/avaliar.py`, que
decide contra quem o `AgenteRL` joga em cada partida de **treino**
(`avaliar_contra_heuristico`, chamado a cada checkpoint, sempre usa o
`AgenteEstrategico` como referência externa, independente desta escolha):

| Valor | Comportamento |
|---|---|
| `"self_play"` (padrão) | A política treina contra um pool de snapshots congelados de iterações passadas de si mesma (`RedeNeuralPolitica.clonar`) — o `AgenteEstrategico` não participa do treino. Ver `references/fundamentos_rl.pdf`, seção "Self-play e o paralelo com o AlphaGo". |
| `"heuristico"` | Toda partida de treino é direto contra o `AgenteEstrategico` fixo; não há pool. Útil para gerar um checkpoint de comparação controlada (ver `diagnostico/avalia_self-play/`). |

```bash
python -c "from training.self_play import treinar; treinar(oponente_treino='heuristico')"
```

### Pool de oponentes (modo `"self_play"`)

O pool de snapshots congelados (`tamanho_pool_max`, padrão 50) vive **só em
memória**, durante a chamada a `treinar` — nenhum snapshot é gravado em
disco. Isso foi uma escolha deliberada: salvar um `.npz` por atualização do
pool (a cada `intervalo_pool` iterações) cresce sem limite ao longo de um
treino longo — um teste de 5.000 iterações chegou a acumular 500 arquivos e
~235MB em `models/rl_pool/`. Manter o pool só em memória troca isso por uma
limitação aceita: se o treinamento for interrompido e retomado
(`caminho_pesos_rl` já existente), o pool reinicia vazio (só com a rede
recém-carregada) — a diversidade de oponentes acumulada na execução
anterior não sobrevive à retomada, mas o disco usado pelo treino continua
sendo só o de sempre: um único arquivo em `caminho_pesos_rl`.

## Comparando regimes de treino

`diagnostico/avalia_self-play/` compara um checkpoint treinado com
`oponente_treino="self_play"` contra um treinado com
`oponente_treino="heuristico"` — veja o README dessa pasta para o passo a
passo completo (como gerar os dois checkpoints e rodar a comparação).

**Nota sobre o self-play puro:** comparando dois checkpoints treinados do
zero por 5.000 iterações a partir do mesmo warm-start de SL — um com
`oponente_treino="self_play"`, outro com `"heuristico"` —, em 1.000
partidas de avaliação cada, ambos os regimes ficaram em ~46–47% de vitórias
vs. o heurístico (IC 95% sobrepostos), e o confronto direto entre os dois
(self-play+pool venceu 52.4%, IC95% [49.3%, 55.5%]) inclui 50% no
intervalo — sem diferença estatisticamente significativa nesse orçamento de
treino, e nenhum dos dois supera claramente o heurístico ainda. Números
completos e como reproduzir: `diagnostico/avalia_self-play/README.md`. A
mudança para self-play puro como padrão é uma correção de metodologia (o
treino deixa de depender do heurístico, alinhando com o self-play do
AlphaGo — ver `references/fundamentos_rl.pdf`), não uma melhora de
desempenho comprovada nessa escala de treino.
