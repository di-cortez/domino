# Dominó — Neural vs Heurístico

Simulador interativo de dominó para dois jogadores, com visualização 3D em OpenGL e dois tipos de agentes de IA: um agente **heurístico** baseado em regras e um agente **neural** treinado por aprendizado supervisionado (imitação do agente heurístico).

O projeto foi desenvolvido como trabalho de Iniciação Científica (IC) na UNIFEI.

---

## Visão geral da arquitetura

A organização em diretórios segue a arquitetura descrita em
`arquitetura_rede_neural.pdf`: UI, Middleware/Adaptador (com o protocolo
único `Agente`), Agentes (heurístico e neural), Treinamento (gerador de
partidas + treinador) e os artefatos de dados (`dataset/`, `models/`).

```
 ┌────────────────────────────────────────────────────────────┐
 │   ui/main_visual.py  ← ponto de entrada da simulação visual│
 └──────────────────────────┬─────────────────────────────────┘
                            │ instancia e conecta
       ┌────────────────────┴─────────────────────┐
       ▼                                          ▼
┌───────────────────────────┐            ┌─────────────────────────────┐
│ ControladorPartida        │            │  GerenciadorPartida         │
│ ui/controle_partida.py    │───────────▶│  middleware/middleware.py   │
│ (histórico, pausa, menu)  │            │  (jogar_turno)              │
└────────────┬──────────────┘            └──────────────┬──────────────┘
             │ renderiza                                │ usa
             ▼                                          ▼
┌───────────────────────────────────┐       ┌─────────────────────────────┐
│ RenderizadorEspacial + HudRenderer│       │  MotorDomino                │ 
│ ui/interface.py + ui/hud.py       │       │  middleware/motor_domino.py │
└───────────────────────────────────┘       └──────────────┬──────────────┘
                                                           │ escolher_jogada()
                                                           │ 
                                                           ▼
                                              ┌────────────────────────────┐
                                              │ agents/agent.py (Agente)   │
                                              │ agents/heuristic_agent.py  │
                                              │ agents/agent_neural.py     │
                                              │ agents/codificador.py      │
                                              │ agents/nn.py               │
                                              └────────────────────────────┘
```

### Pipeline de treinamento (SL — imitação do heurístico)

```
training/gerador.py ──▶ dataset/dataset_*.jsonl ──▶ training/training_loop.py ──▶ models/pesos_domino_sl.npz
   (simula partidas               (carrega, codifica,                (pesos da rede
    com agentes heurísticos)        treina a rede neural)              salva em disco)
```

### Pipeline de treinamento (RL — refinamento por self-play)

```
models/pesos_domino_sl.npz ──▶ training/self_play.py ──▶ models/pesos_domino_rl.npz
  (warm-start: ponto de         (self-play + partidas         (pesos da política
   partida da política RL)       vs AgenteEstrategico;          RL salvos em disco,
                                  REINFORCE + baseline)          checkpoints periódicos)
```

`agents/rl_nn.py` (`RedeNeuralPolitica`) usa a mesma arquitetura 79→256→128→58
de `agents/nn.py`, então qualquer checkpoint de `pesos_domino_sl.npz` carrega
direto como ponto de partida. O pipeline de SL não é alterado por isso: os
dois fluxos de treinamento — e os dois arquivos de pesos — são independentes
e podem ser regenerados/rodados sem afetar um ao outro.

---

## Estrutura de diretórios

```
.
├── ui/                    # Interface de usuário (Python + PyOpenGL)
│   ├── main_visual.py
│   ├── interface.py
│   ├── hud.py
│   └── controle_partida.py
├── middleware/            # Middleware / Adaptador
│   ├── middleware.py
│   └── motor_domino.py
├── agents/                # Agente heurístico + Agente neural (e protocolo Agente)
│   ├── agent.py
│   ├── heuristic_agent.py
│   ├── agent_neural.py
│   ├── codificador.py
│   └── nn.py
├── training/              # Gerador de partidas + Treinador de modelos (SL e RL)
│   ├── gerador.py
│   ├── training_loop.py
│   └── self_play.py
├── diagnostico/           # Avaliação de agentes: métricas + gráficos
│   ├── avaliar.py
│   └── gera_graficos.py
├── dataset/               # Dataset de partidas (gerado, não versionado)
└── models/                # Modelo treinado (pesos .npz, gerado, não versionado)
```

## Estrutura dos módulos
-------------------------------------------------------------------------------------------------
|            Arquivo            |                       Responsabilidade                        |
|-------------------------------|---------------------------------------------------------------|
| `middleware/motor_domino.py`  | Regras completas do dominó: embaralhamento, distribuição,     |
|                               | `step()`, `reset()`, estado serializado                       |
|-----------------------------------------------------------------------------------------------|
|      `agents/agent.py`        | Classe-base abstrata `Agente`                                 |
|                               | (protocolo único `escolher_jogada`)                           |
|-----------------------------------------------------------------------------------------------|
| `agents/heuristic_agent.py`   | `AgenteEstrategico` — decisões por função de utilidade        |
|                               |   (diversidade de mão, cobertura de pontas, urgência)         |
|-----------------------------------------------------------------------------------------------|
|  `agents/agent_neural.py`     | `AgenteNeuralNumPy` — forward pass + action masking;          |
|                               |   carrega pesos de `.npz`                                     |
|-----------------------------------------------------------------------------------------------|
|    `agents/codificador.py`    | `CodificadorDomino` — converte estado ↔ vetor de 79 dimensões |
|                               |   e ação ↔ índice em espaço de 58 ações                       |
|-----------------------------------------------------------------------------------------------|
|        `agents/nn.py`         | `RedeNeuralSupervisionada` — rede 79→256→128→58               |
|                               |  (ReLU + Softmax), forward e backprop em NumPy/CuPy           |
|-----------------------------------------------------------------------------------------------|
|      `agents/rl_nn.py`        | `RedeNeuralPolitica` — mesma arquitetura 79→256→128→58;       |
|                               |  atualizada por REINFORCE + baseline em vez de cross-entropy  |
|-----------------------------------------------------------------------------------------------|
|    `agents/agent_rl.py`       | `AgenteRL` — joga amostrando da política (treino) ou          |
|                               |  greedy (avaliação/UI); registra a trajetória do episódio     |
|-----------------------------------------------------------------------------------------------|
| `middleware/middleware.py`    | `GerenciadorPartida` — orquestra motor ↔ agentes por turno;   |
|                               |  registra pares (estado, ação) para SL                        |
|-----------------------------------------------------------------------------------------------|
|   `ui/controle_partida.py`    | `ControladorPartida` — lógica de pausa, avanço/retrocesso no  |
|                               |  histórico, menu de configuração, notificações                |
|-----------------------------------------------------------------------------------------------|
|       `ui/interface.py`       | `RenderizadorEspacial` — layout snake das peças em OpenGL;    |
|                               |  funções de desenho de peças e pips                           |
|-----------------------------------------------------------------------------------------------|
|          `ui/hud.py`          | `HudRenderer` — overlay 2D em OpenGL: barra de turno,         |
|                               |  contagem de peças, notificações, menu                        |
|-----------------------------------------------------------------------------------------------|
|     `training/gerador.py`     | Gera dataset JSONL simulando partidas entre dois agentes      |
|                               | heurísticos                                                   |
|-----------------------------------------------------------------------------------------------|
| `training/training_loop.py`   | Carrega dataset, codifica estados/ações, treina a rede e      |
|                               | salva os pesos                                                |
|-----------------------------------------------------------------------------------------------|
|    `training/self_play.py`    | Treina `AgenteRL` por self-play (REINFORCE + currículo misto  |
|                               | com `AgenteEstrategico`); salva checkpoints e avalia          |
|                               | periodicamente                                                |
|-----------------------------------------------------------------------------------------------|
|      `ui/main_visual.py`      | Ponto de entrada: inicializa Pygame/OpenGL, instancia todos   |
|                               | os componentes, executa o loop principal                      |
|-----------------------------------------------------------------------------------------------|
|   `diagnostico/avaliar.py`    | Configura e simula N partidas de um agente contra um oponente;|
|                               | salva `resumo.json`/`partidas.csv`                            |
|-----------------------------------------------------------------------------------------------|
| `diagnostico/gera_graficos.py`| Métricas (IC de Wilson, desempenho por posição) e os PNGs     |
|                               | de diagnóstico                                                |
-----------------------------------------------------------------------------------------------

## Pré-requisitos

```
Python >= 3.10
pygame
PyOpenGL
PyOpenGL-accelerate   # opcional, melhora performance
numpy
cupy                  # opcional — ativa treinamento/inferência na GPU
```

Instale as dependências:

```bash
pip install pygame PyOpenGL PyOpenGL-accelerate numpy
# GPU (CUDA):
pip install cupy-cuda12x   # ajuste a versão do CUDA conforme necessário
```

---

## Como executar

Todos os comandos abaixo devem ser executados a partir da raiz do projeto
(este diretório), usando `python -m` para que os pacotes `ui`, `middleware`,
`agents` e `training` sejam resolvidos corretamente.

### Simulação visual

```bash
python -m ui.main_visual
```

Abre uma janela 1024×768 com a partida em andamento. Por padrão: **Jogador 0 = Neural**, **Jogador 1 = Heurístico**. Pelo menu (`M`), cada posição também pode ser trocada para **RL (self-play)**, além de Aleatório e Humano.

### Gerar dataset de treinamento

```bash
python -m training.gerador
```

Produz `dataset/dataset_2.jsonl` com ~400 000 pares (estado, ação) a partir de 20 000 partidas entre agentes heurísticos.

### Treinar a rede neural

```bash
python -m training.training_loop
```

Lê o dataset, treina por múltiplos épocas e salva `models/pesos_domino_sl.npz`.

### Refinar a política por self-play (RL)

```bash
python -m training.self_play
```

Carrega `models/pesos_domino_rl.npz` se já existir (retomando o treinamento)
ou faz warm-start a partir de `models/pesos_domino_sl.npz`. Joga lotes de
partidas — self-play e contra o `AgenteEstrategico`, na proporção configurada
— e atualiza a política via REINFORCE com baseline a cada lote. Salva
checkpoints e imprime o win-rate contra o heurístico periodicamente.

---

## Controles da simulação visual

---------------------------------------------------------------------------------------
|     Tecla    |                                  Ação                                |
|-------------------------------------------------------------------------------------|
|   `Espaço`   | Pausa / retoma o avanço automático                                   |
|-------------------------------------------------------------------------------------|
|     `→`      | Avança um turno (e pausa)                                            |
|-------------------------------------------------------------------------------------|
|     `←`      | Retrocede ao turno anterior (e pausa)                                |
|-------------------------------------------------------------------------------------|
|  `+` / `-`   | Muda a velocidade do avanço automático entre 1/4x, 1/2x, 1x, 2x e 4x |
|-------------------------------------------------------------------------------------|
|   `J` / `K`  | Alterna a visibilidade da mão do Jogador 0 / Jogador 1 (quando o     |
|              | modo permite)                                                        |
|-------------------------------------------------------------------------------------|
|     `R`      | Reinício rápido: pede confirmação (aperte `R` de novo em até 2s) se  |
|              |  a partida ainda não acabou; reinicia direto se já tiver terminado   |
|-------------------------------------------------------------------------------------|
|     `M`      | Abre / fecha o menu de configuração                                  |
|-------------------------------------------------------------------------------------|
|    `ESC`     | Fecha o menu (se aberto) ou encerra a aplicação                      |
---------------------------------------------------------------------------------------

Uma notificação centralizada confirma a mudança de velocidade, o pedido de confirmação do `R` e as compras de peça do estoque.

### Controles do jogador humano

Quando a posição de um jogador está configurada como **Humano** (pelo menu `M`) e é a vez dele:

-----------------------------------------------------------------------------
|       Tecla       |                          Ação                         |
|---------------------------------------------------------------------------|
|     `←` / `→`     | Navega entre as peças da mão                          |
|---------------------------------------------------------------------------|
| `↑` / `↓` / `Tab` | Alterna a ponta da mesa escolhida (esquerda/direita), |
|                   | quando a jogada permite                               |
|---------------------------------------------------------------------------|
|      `Enter`      | Joga a peça selecionada na ponta escolhida            |
|---------------------------------------------------------------------------|
|        `C`        | Compra uma peça do estoque                            |
|---------------------------------------------------------------------------|
|        `P`        | Passa a vez                                           |
-----------------------------------------------------------------------------

### Menu de configuração (`M`)

- **Setas ↑↓** — navega entre os itens: Jogador 0, Jogador 1 e Reiniciar
- **Enter / Espaço** — alterna o tipo de agente do jogador selecionado (`Neural` → `Heurístico` → `Aleatório` → `Humano` → `RL`, em ciclo) ou executa o reinício
- **M / ESC** — fecha o menu sem alterar nada

O menu recalcula suas dimensões dinamicamente para acomodar o texto mais largo de cada opção.

---

## HUD (interface na tela)

**Barra superior** — exibe para cada jogador: nome, tipo de agente, contagem atual de peças na mão e badge `[VEZ]` destacando de quem é a vez.

**Barra inferior** — resumo dos atalhos de teclado.

**Notificação de compra** — quando um agente compra uma peça do estoque, aparece uma mensagem centralizada logo abaixo da barra superior com fade-out ao longo de 3 segundos. A notificação é ativada tanto durante o avanço automático quanto ao navegar pelo histórico com as setas.

**Banner de fim de jogo** — exibe o vencedor (ou empate) ao término da partida.

---

## Rede neural

------------------------------------------------------------------------
|     Parâmetro     |                       Valor                      |
|----------------------------------------------------------------------|
|    Arquitetura    | 79 → 256 → 128 → 58                              |
|----------------------------------------------------------------------|
| Ativação oculta   | ReLU                                             |
|----------------------------------------------------------------------|
| Ativação de saída | Softmax                                          |
|----------------------------------------------------------------------|
|   Inicialização   | He (camadas ocultas), Xavier (saída)             |
|----------------------------------------------------------------------|
|    Treinamento    | Supervisionado por imitação do agente heurístico |
|----------------------------------------------------------------------|
|     Backend       | NumPy (CPU) ou CuPy (GPU, automático)            |
------------------------------------------------------------------------

**Vetor de estado (79 dims):**
- Peças na mão do jogador atual (28 bits)
- Ponta esquerda da mesa (7 bits)
- Ponta direita da mesa (7 bits)
- Tamanho das mãos dos jogadores (2 valores)
- Tamanho do estoque (1 valor)
- Turno normalizado (1 valor)
- ... campos adicionais de contexto

**Espaço de ações (58 ações):**
- 28 jogadas pelo lado esquerdo (uma por peça do conjunto)
- 28 jogadas pelo lado direito
- Comprar do estoque
- Passar a vez

### Treinamento por reforço (RL — self-play)

`agents/agent_rl.py` (`AgenteRL`) usa a mesma rede (via `agents/rl_nn.py`), mas
troca o treinamento supervisionado por REINFORCE com baseline:

---------------------------------------------------------------------------------------------------------|
|       Parâmetro       |            Valor padrão          |                    Descrição                |
|--------------------------------------------------------------------------------------------------------|
|       Algoritmo       | REINFORCE + baseline             | Baseline = média dos retornos do lote       |
|                       |                                  | (reduz variância)                           |
|--------------------------------------------------------------------------------------------------------|
|       Recompensa      | +1 vitória / -1 derrota /        | Esparsa: só ao fim da partida, propagada    |
|                       | 0 empate                         | (γ = 1) a todas as jogadas do               |
|                       |                                  | agente no episódio (Monte Carlo)            |
|--------------------------------------------------------------------------------------------------------|
|     Regularização     | Bônus de entropia                | Evita colapso prematuro da política para    |
|                       | (`entropia_coef=0.01`)           | determinística                              |
|                       |                                  |                                             |
|--------------------------------------------------------------------------------------------------------|
|      Currículo        | 80% self-play / 20% vs.          | `proporcao_self_play` em                    |  
|                       |`AgenteEstrategico`               | `training/self_play.py`                     |
|--------------------------------------------------------------------------------------------------------|
| Partidas por iteração | 20                               | `partidas_por_iteracao`                     |
|--------------------------------------------------------------------------------------------------------|
|       Iterações       | 500                              | `iteracoes`                                 |
|--------------------------------------------------------------------------------------------------------|
|   Taxa de aprendizado | 0.001                            | `taxa_aprendizado`                          |
|--------------------------------------------------------------------------------------------------------|
|        Checkpoints    | a cada 50 iterações              | salva `models/pesos_domino_rl.npz` e avalia |
|                       |                                  | 200 partidas(greedy) vs. `AgenteEstrategico`|
|--------------------------------------------------------------------------------------------------------|

**Modos do `AgenteRL`:**
- `treino` — amostra a jogada estocasticamente a partir da distribuição softmax e registra (estado, ação) da trajetória do episódio, para o cálculo do gradiente ao final da partida.
- `avaliacao` — joga greedy (argmax), igual ao `AgenteNeuralNumPy`; usado nos benchmarks e como agente final na UI/menu (opção `RL`).

---

## Avaliação de modelos (`diagnostico/`)

Módulo para comparar qualquer agente do projeto contra qualquer oponente em N
partidas e gerar métricas + gráficos — útil para verificar se o RL está de
fato superando o SL/heurístico ao longo do treinamento.

```bash
python3 -m diagnostico.avaliar --agente rl --oponente heuristico -n 1000
```

Agentes disponíveis (`--agente` / `--oponente`): `rl`, `sl`, `heuristico`,
`guloso` (joga sempre a peça de maior soma de pips) e `aleatorio`. A posição
inicial do agente avaliado é alternada a cada partida para não contaminar o
resultado com vantagem de quem começa.

Saídas em `diagnostico/resultados/<agente>_vs_<oponente>/` (ou `--saida`):

|------------------------------------------------------------------------------------------|
|            Arquivo            |                         Conteúdo                         |
|------------------------------------------------------------------------------------------|
|          `resumo.json`        | Taxas de vitória/empate/derrota, IC 95% (Wilson),        |
|                               |  desempenho por posição, duração média, pips restantes   |
|------------------------------------------------------------------------------------------|
|         `partidas.csv`        | Uma linha por partida (posição, resultado, turnos, pips) |
|------------------------------------------------------------------------------------------|
|     `taxas_acumuladas.png`    | Convergência das taxas ao longo das partidas             |
|------------------------------------------------------------------------------------------|
| `distribuicao_resultados.png` | Contagem final de vitórias/empates/derrotas              |
|------------------------------------------------------------------------------------------|
|  `vitorias_por_posicao.png`   | Taxa de vitória como jogador 0 vs. jogador 1, com IC 95% |
|------------------------------------------------------------------------------------------|
|     `duracao_partidas.png`    | Histograma de turnos por partida                         |
|------------------------------------------------------------------------------------------|

Configuração padrão (agente, oponente, número de partidas, seed) fica editável
no topo de `diagnostico/avaliar.py`; qualquer valor pode ser sobrescrito por
linha de comando (`--help` lista todas as opções). Detalhes completos e mais
exemplos em `diagnostico/README.md`.

---

## Arquivos gerados (não versionados)

|------------------------------------------------------------------------------------------------------|
|           Arquivo            |                                Descrição                              |
|------------------------------------------------------------------------------------------------------|
|   `dataset/dataset_2.jsonl`  | Dataset gerado pelo `training/gerador.py`                             |
| `models/pesos_domino_sl.npz` | Pesos da rede neural treinada por SL (imitação)                       |
| `models/pesos_domino_rl.npz` | Pesos da política RL refinada por self-play (`training/self_play.py`) |
|------------------------------------------------------------------------------------------------------|

Para reproduzir o modelo de SL do zero: execute `python -m training.gerador` e depois `python -m training.training_loop`.
Para refinar a política de RL a partir dele: execute `python -m training.self_play`.
