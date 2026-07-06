# Agentes

Todos os jogadores do projeto — heurístico, neural (SL) e por reforço (RL) —
implementam o mesmo protocolo `Agente.escolher_jogada(estado, jogadas_legais)`,
o que permite ao `GerenciadorPartida` (em `middleware/middleware.py`) jogar
qualquer combinação deles sem saber como cada um decide sua jogada.

## Arquivos

| Arquivo | Responsabilidade |
|---|---|
| `agent.py` | Módulo histórico e autoexecutável (`python -m agents.agent`). Reexporta `Agente`/`GerenciadorPartida` de `middleware/middleware.py` (fonte única da verdade) e define os agentes baseline usados pelo `diagnostico/`: `AgenteAleatorio` (jogada ao acaso) e `AgenteGuloso` (maior soma de pips). |
| `heuristic_agent.py` | `AgenteEstrategico` — agente professor por regras. Decide pela maior utilidade (peso da peça, bônus de dupla, diversidade da mão, cobertura de pontas, urgência, e bloqueio dos naipes que os oponentes já demonstraram não ter). A inferência dos naipes mortos (`_inferir_ausencias_oponentes`) delega para `middleware.motor_domino.inferir_naipes_mortos` — a mesma função que o motor usa para preencher a chave `"naipes_mortos_oponente"` do estado, evitando duas implementações divergentes. Parâmetros de peso configuráveis pelo construtor; `ruido_desempate` opcional para diversificar a geração do dataset. Gera os rótulos do dataset de SL; no RL, serve só de referência externa e fixa nas avaliações periódicas (`avaliar_contra_heuristico`) — não participa mais do treino de self-play. |
| `codificador.py` | `CodificadorDomino` — fonte única da verdade para estado ↔ vetor (86 dims, incluindo os 7 bits de naipes mortos do oponente) e jogada ↔ índice (58 ações). Usado por todos os agentes neurais. Oferece `decode_saida` (argmax mascarado) e `amostrar_acao` (amostragem estocástica do self-play). |
| `nn.py` | `RedeNeuralSupervisionada` — MLP 86→256→128→58 (ReLU + Softmax, inicialização He/Xavier). Implementa `forward`, `backward` (cross-entropy) e o loop `treinar` com mini-batches, com callback `ao_validar` opcional (usado por `training/training_loop.py` para guardar em memória os pesos da época de menor Val Custo). Roda em NumPy (CPU) ou CuPy (GPU), escolhido na importação. |
| `rl_nn.py` | `RedeNeuralPolitica` — mesma arquitetura, herda o `forward`, mas treina por `backward_policy_gradient` (REINFORCE + baseline + entropia). Tem uma cabeça de valor linear (`Wv`/`bv`, ver `prever_valores`) treinada por regressão contra os retornos — um baseline dependente do estado (ator-crítico simples) usado por `training/self_play.py` para reduzir a variância do gradiente. Gradientes têm norma global limitada (`clip_grad_norm`) para estabilidade. `clonar` (snapshot congelado) alimenta o pool de oponentes do self-play — ver `training/self_play.py`. Adiciona `carregar_de_sl` (warm-start) e `carregar`/`salvar` para checkpoints RL — checkpoints antigos sem cabeça de valor continuam carregando (`Wv`/`bv` iniciam em zero). |
| `agent_neural.py` | `AgenteNeuralNumPy` — agente de inferência da rede de SL. Carrega pesos de `.npz`, aplica o `CodificadorDomino` e decide por política epsilon-greedy com action masking (`epsilon=0.0` por padrão). |
| `agent_rl.py` | `AgenteRL` — agente de inferência/treino da política de RL. Em `modo="treino"` amostra e registra a trajetória; em `modo="avaliacao"` joga greedy. `finalizar_episodio` propaga a recompensa final como retorno de Monte Carlo. Usado tanto para a rede em treino quanto para os oponentes congelados do pool em `training/self_play.py`. |
| `__init__.py` | Marca `agents` como pacote Python; sem conteúdo. |

## Vetor de estado e espaço de ações

Ver a seção **Rede neural** no `README.md` da raiz do projeto para o detalhamento
completo do vetor de 86 dimensões e das 58 ações possíveis — a codificação vive
inteiramente em `codificador.py` e é compartilhada por todos os agentes neurais.
