# Diagnóstico do treinamento

Avalia qualquer agente do projeto contra qualquer oponente em N partidas e
gera métricas + gráficos, para verificar se o modelo de RL está de fato
aprendendo.

## Uso — o jeito simples (editar variáveis)

Abra `diagnostico/avaliar.py` e edite a seção **CONFIGURAÇÃO** no topo do
arquivo:

```python
AGENTE = "rl"            # jogador avaliado
OPONENTE = "heuristico"  # adversário
NUM_PARTIDAS = 500
```

Depois, a partir da raiz do repositório e com o ambiente virtual ativo:

```bash
source ../../amb_virtual/bin/activate   # ajuste o caminho se necessário
python3 -m diagnostico.avaliar
```

Nada mais é necessário: trocar `"heuristico"` por `"aleatorio"`, por exemplo,
já avalia contra o jogador aleatório na próxima execução.

## Uso — pelo terminal (opcional)

Qualquer valor da CONFIGURAÇÃO pode ser sobrescrito sem editar o arquivo:

```bash
# RL vs aleatório, 1000 partidas, com semente fixa
python3 -m diagnostico.avaliar --agente rl --oponente aleatorio -n 1000 --seed 42

# Baseline: quanto o SL puro ganha do heurístico?
python3 -m diagnostico.avaliar --agente sl --oponente heuristico -n 500

# Sanidade: aleatório vs aleatório deve dar ~50% de vitórias por posição
python3 -m diagnostico.avaliar --agente aleatorio --oponente aleatorio -n 500

# Comparar dois checkpoints de RL entre si
python3 -m diagnostico.avaliar --agente rl --pesos models/pesos_domino_rl.npz \
    --oponente rl --pesos-oponente models/checkpoint_antigo.npz -n 500
```

Agentes disponíveis para `--agente` e `--oponente`:

|----------------------------------------------------------------------------------|
|     Nome     |        Classe       |                  Descrição                  |
|--------------|---------------------|---------------------------------------------|
| `rl`         | `AgenteRL`          | política REINFORCE (modo avaliação, greedy) |
|--------------|---------------------|---------------------------------------------|
| `sl`         | `AgenteNeuralNumPy` | rede supervisionada (imitação do heurístico)|
|--------------|---------------------|---------------------------------------------|
| `heuristico` | `AgenteEstrategico` | agente de regras                            |
|--------------|---------------------|---------------------------------------------|
| `guloso`     | `AgenteGuloso`      | joga a peça de maior soma de pips           |
| `aleatorio`  | `AgenteAleatorio`   | jogada válida uniforme                      |
------------------------------------------------------------------------------------

A posição inicial do agente avaliado é alternada a cada partida, para o
resultado não ser contaminado por vantagem de quem começa.

Organização dos arquivos: `avaliar.py` cuida da configuração e da simulação
das partidas; `gera_graficos.py` contém as métricas (resumo estatístico,
IC de Wilson) e os gráficos.

## Saídas

Ficam em `diagnostico/resultados/<agente>_vs_<oponente>/` (ou em `--saida`):

- `resumo.json` — taxas de vitória/empate/derrota, IC 95% (Wilson),
  desempenho por posição, duração média, pips restantes.
- `partidas.csv` — uma linha por partida (posição, resultado, turnos, pips).
- `taxas_acumuladas.png` — convergência das taxas ao longo das partidas.
- `distribuicao_resultados.png` — contagem final de vitórias/empates/derrotas.
- `vitorias_por_posicao.png` — taxa de vitória como jogador 0 vs jogador 1,
  com barras de erro (IC 95%).
- `duracao_partidas.png` — histograma de turnos por partida.

## Como interpretar

- **RL treinando bem**: taxa de vitória vs `heuristico` acima da do `sl`
  (o ponto de partida do RL) e crescendo entre checkpoints.
- **IC 95%**: se os intervalos de dois modelos se sobrepõem muito, a diferença
  pode ser ruído — aumente `-n`.
- **Por posição**: uma assimetria grande entre jogador 0 e 1 indica vantagem
  estrutural de quem começa; compare sempre com o mesmo protocolo alternado.
- **Empates**: em dominó fechado, empate alto contra oponentes fortes é normal;
  acompanhe a soma vitória + meio empate como score, se preferir.
