# Agentes

Todos os jogadores do projeto — heurístico, neural (SL) e por reforço (RL) —
implementam o mesmo protocolo `Agente.escolher_jogada(estado, jogadas_legais)`,
o que permite ao `GerenciadorPartida` (em `middleware/middleware.py`) jogar
qualquer combinação deles sem saber como cada um decide sua jogada.

## Arquivos

| Arquivo | Responsabilidade |
|---|---|
| `agent.py` | Módulo histórico e autoexecutável (`python -m agents.agent`). Define `Agente`/`GerenciadorPartida` (duplicados dos de `middleware/middleware.py`) e os agentes baseline usados pelo `diagnostico/`: `AgenteAleatorio` (jogada ao acaso) e `AgenteGuloso` (maior soma de pips). |
| `heuristic_agent.py` | `AgenteEstrategico` — agente professor por regras. Decide pela maior utilidade (peso da peça, bônus de dupla, diversidade da mão, cobertura de pontas, urgência). Gera os rótulos do dataset de SL e serve de oponente/baseline no RL e nas avaliações. |
| `codificador.py` | `CodificadorDomino` — fonte única da verdade para estado ↔ vetor (79 dims) e jogada ↔ índice (58 ações). Usado por todos os agentes neurais. Oferece `decode_saida` (argmax mascarado) e `amostrar_acao` (amostragem estocástica do self-play). |
| `nn.py` | `RedeNeuralSupervisionada` — MLP 79→256→128→58 (ReLU + Softmax, inicialização He/Xavier). Implementa `forward`, `backward` (cross-entropy) e o loop `treinar` com mini-batches. Roda em NumPy (CPU) ou CuPy (GPU), escolhido na importação. |
| `rl_nn.py` | `RedeNeuralPolitica` — mesma arquitetura, herda o `forward`, mas treina por `backward_policy_gradient` (REINFORCE + baseline + entropia). Adiciona `carregar_de_sl` (warm-start) e `carregar`/`salvar` para checkpoints RL. |
| `agent_neural.py` | `AgenteNeuralNumPy` — agente de inferência da rede de SL. Carrega pesos de `.npz`, aplica o `CodificadorDomino` e decide por política epsilon-greedy com action masking (`epsilon=0.0` por padrão). |
| `agent_rl.py` | `AgenteRL` — agente de inferência/treino da política de RL. Em `modo="treino"` amostra e registra a trajetória; em `modo="avaliacao"` joga greedy. `finalizar_episodio` propaga a recompensa final como retorno de Monte Carlo. |
| `__init__.py` | Marca `agents` como pacote Python; sem conteúdo. |

## Vetor de estado e espaço de ações

Ver a seção **Rede neural** no `README.md` da raiz do projeto para o detalhamento
completo do vetor de 79 dimensões e das 58 ações possíveis — a codificação vive
inteiramente em `codificador.py` e é compartilhada por todos os agentes neurais.
