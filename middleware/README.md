# Middleware

Camada intermediária entre o motor de regras do dominó e os agentes de IA/UI.
Nenhum arquivo aqui decide estratégia de jogo — isso é responsabilidade dos
`agents/`; o middleware só garante as regras do jogo e o fluxo de turnos.

## Arquivos

| Arquivo | Responsabilidade |
|---|---|
| `motor_domino.py` | `MotorDomino` — regras completas do dominó: embaralhamento e distribuição (`reset`), jogadas legais (`acoes_validas`), transição de turno (`step`), detecção de fim de jogo/empate e estado serializado (`_obter_estado`, `to_dict`). |
| `middleware.py` | `Agente` (interface) e `GerenciadorPartida` — orquestrador usado pelo projeto. `jogar_turno` liga motor e agentes e registra pares (estado, ação) em `historico_treinamento`, a fonte do dataset de SL. |
| `__init__.py` | Marca `middleware` como pacote Python; sem conteúdo. |

## Formato de uma ação

Uma ação é sempre uma das quatro formas abaixo, consumidas por `acoes_validas`/`step`:

- `(peca, lado)` — joga a peça `(a, b)` na ponta `0` (esquerda) ou `1` (direita);
- `("COMPRAR", None)` — compra uma peça do monte;
- `None` — passa a vez (só permitido quando não há jogada nem peça para comprar).

Essa é também a convenção usada por `agents/codificador.py` para mapear ações a índices.
