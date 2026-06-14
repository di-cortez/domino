"""
agentes_ui.py

Pequena camada de adaptação entre a interface visual e os agentes do projeto.

O restante da aplicação conhece os jogadores por strings simples:

    "neural", "heuristico", "aleatorio", "humano"

Este módulo é o único lugar que traduz essas strings para objetos de agente
compatíveis com o `GerenciadorPartida`. Isso deixa o controlador visual livre
de imports diretos para cada implementação de agente e evita espalhar regras de
nome/apresentação pela HUD.
"""

import random


TIPOS_AGENTE = ('neural', 'heuristico', 'aleatorio', 'humano')


class AgenteAleatorioUI:
    """Agente simples usado só pela UI para sorteio entre jogadas legais."""

    def escolher_jogada(self, estado, jogadas_legais):
        return random.choice(jogadas_legais)


class AgenteHumanoBloqueado:
    """
    Sentinela para jogador humano.

    O humano nunca deve ser chamado pelo `GerenciadorPartida`: a jogada dele é
    executada diretamente pelo `ControladorPartida`, após teclado/validação.
    Se este agente for chamado, há um erro no fluxo de controle da UI.
    """

    def escolher_jogada(self, estado, jogadas_legais):
        raise RuntimeError("Turno humano deve ser tratado pelo controlador da UI.")


def nome_tipo_agente(tipo):
    """Nome amigável para exibir na HUD e em notificações."""
    nomes = {
        'neural': 'Neural',
        'heuristico': 'Heurístico',
        'aleatorio': 'Aleatório',
        'humano': 'Humano',
    }
    return nomes.get(tipo, tipo.capitalize())


def criar_agente_por_tipo(tipo):
    """
    Fábrica central de agentes da UI.

    Mantemos os imports dentro da função para evitar carregar rede neural e
    dependências de treinamento quando o usuário só está usando outro modo.
    """
    if tipo == 'neural':
        from agents.agent_neural import AgenteNeuralNumPy
        return AgenteNeuralNumPy.carregar("models/pesos_domino_sl.npz")

    if tipo == 'heuristico':
        from agents.heuristic_agent import AgenteEstrategico
        return AgenteEstrategico()

    if tipo == 'aleatorio':
        return AgenteAleatorioUI()

    if tipo == 'humano':
        return AgenteHumanoBloqueado()

    raise ValueError(f"Tipo de agente inválido: {tipo}")
