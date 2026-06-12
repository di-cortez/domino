"""
controle_partida.py

Camada de controle/playback da simulação de dominó.
Separa a entrada de teclado e o avanço/retrocesso de turnos do laço
principal de renderização (main_visual.py).

Teclas:
    M        -> abre/fecha menu de configuração
    Espaço   -> pausa / retoma o avanço automático
    Direita  -> avança um turno (e pausa o automático)
    Esquerda -> retrocede um turno (e pausa o automático)
    ESC      -> encerra (fecha menu se estiver aberto)
"""

import copy
import pygame

_TIPOS_AGENTE = ('neural', 'heuristico')


class ControladorPartida:
    def __init__(self, gerenciador, motor, intervalo_ms=1000, tipos_agentes=None):
        self.gerenciador = gerenciador
        self.motor = motor
        self.intervalo_ms = intervalo_ms
        self.tipos_agentes = list(tipos_agentes) if tipos_agentes else ['neural', 'heuristico']

        # O motor só sabe avançar; guardamos snapshots para poder revisitar
        # estados anteriores (necessário para a seta Esquerda).
        # historico_info[i] armazena o info da jogada que gerou historico[i]
        # (None para o estado inicial).
        self.historico      = []
        self.historico_info = []
        self.indice = 0
        self.pausado = False
        self.fim_de_jogo = False
        self.info_final = None
        self._acumulador_ms = 0.0

        # Notificação temporária (compra do estoque)
        self.notificacao = None   # {"texto": str, "tempo_ms": int}

        # Menu de configuração
        self.menu_aberto = False
        self.menu_cursor = 0
        self._NUM_ITENS_MENU = 3   # J0, J1, Reiniciar
        self._pausado_antes_menu = False

        self._capturar_estado()   # snapshot do estado inicial

    # ------------------------------------------------------------------ #
    # Estado / histórico
    # ------------------------------------------------------------------ #
    def _capturar_estado(self, info=None):
        # Cópia profunda: o motor reaproveita/altera suas listas internas,
        # então congelamos o estado para conseguir renderizar o passado.
        self.historico.append(copy.deepcopy(self.motor._obter_estado()))
        self.historico_info.append(info)

    @property
    def no_vivo(self):
        """True quando estamos vendo o estado mais recente (a 'ponta viva')."""
        return self.indice >= len(self.historico) - 1

    def estado_atual(self):
        return self.historico[self.indice]

    def _atualizar_notificacao(self):
        """Deriva a notificação do info armazenado no índice atual do histórico."""
        info = self.historico_info[self.indice]
        if info and info.get("acao") == ("COMPRAR", None):
            j    = info["jogador_acao"]
            nome = "Neural" if self.tipos_agentes[j] == "neural" else "Heurístico"
            self.notificacao = {
                "texto":    f"Jogador {j} ({nome}) comprou uma peça do estoque",
                "tempo_ms": 3000,
            }
        else:
            self.notificacao = None

    def _jogar_proximo_turno(self):
        if self.fim_de_jogo:
            return
        try:
            fim, info = self.gerenciador.jogar_turno()
            self._capturar_estado(info)
            self.indice = len(self.historico) - 1
            self._atualizar_notificacao()
            if fim:
                self.fim_de_jogo = True
                self.info_final = info
                if info is not None and "anunciado" not in info:
                    print(f"Partida finalizada! Vencedor: Jogador {info.get('vencedor')}")
                    info["anunciado"] = True
        except Exception as e:
            print(f"Erro durante o turno: {e}")
            self.fim_de_jogo = True

    # ------------------------------------------------------------------ #
    # Navegação
    # ------------------------------------------------------------------ #
    def avancar(self):
        if self.indice < len(self.historico) - 1:
            # Ainda há histórico à frente: apenas reposiciona o cursor.
            self.indice += 1
            self._atualizar_notificacao()
        else:
            # Estamos na ponta viva: joga um turno real novo.
            self._jogar_proximo_turno()

    def retroceder(self):
        if self.indice > 0:
            self.indice -= 1
            self._atualizar_notificacao()

    def alternar_pausa(self):
        self.pausado = not self.pausado
        self._acumulador_ms = 0.0

    # ------------------------------------------------------------------ #
    # Menu de configuração
    # ------------------------------------------------------------------ #
    def _abrir_menu(self):
        self.menu_aberto = True
        self.menu_cursor = 0
        self._pausado_antes_menu = self.pausado
        self.pausado = True

    def _fechar_menu(self):
        self.menu_aberto = False
        self.pausado = self._pausado_antes_menu

    def _ativar_item_menu(self):
        if self.menu_cursor == 0:
            idx = _TIPOS_AGENTE.index(self.tipos_agentes[0])
            self.tipos_agentes[0] = _TIPOS_AGENTE[(idx + 1) % len(_TIPOS_AGENTE)]
        elif self.menu_cursor == 1:
            idx = _TIPOS_AGENTE.index(self.tipos_agentes[1])
            self.tipos_agentes[1] = _TIPOS_AGENTE[(idx + 1) % len(_TIPOS_AGENTE)]
        elif self.menu_cursor == 2:
            self._reiniciar_partida()
            self._fechar_menu()

    def _reiniciar_partida(self):
        from agents.heuristic_agent import AgenteEstrategico
        from agents.agent_neural import AgenteNeuralNumPy
        from middleware.middleware import GerenciadorPartida

        self.motor.reset()
        novos_agentes = []
        for tipo in self.tipos_agentes:
            if tipo == 'neural':
                novos_agentes.append(AgenteNeuralNumPy.carregar("models/pesos_domino_sl.npz"))
            else:
                novos_agentes.append(AgenteEstrategico())
        self.gerenciador = GerenciadorPartida(self.motor, novos_agentes)

        self.historico      = []
        self.historico_info = []
        self.indice = 0
        self.pausado = False
        self.fim_de_jogo = False
        self.info_final = None
        self.notificacao = None
        self._acumulador_ms = 0.0
        self._capturar_estado()

    # ------------------------------------------------------------------ #
    # Laço
    # ------------------------------------------------------------------ #
    def processar_entrada(self):
        """Consome os eventos do pygame. Retorna False quando for para encerrar."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if self.menu_aberto:
                    if event.key in (pygame.K_ESCAPE, pygame.K_m):
                        self._fechar_menu()
                    elif event.key == pygame.K_UP:
                        self.menu_cursor = (self.menu_cursor - 1) % self._NUM_ITENS_MENU
                    elif event.key == pygame.K_DOWN:
                        self.menu_cursor = (self.menu_cursor + 1) % self._NUM_ITENS_MENU
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                        self._ativar_item_menu()
                else:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    elif event.key == pygame.K_m:
                        self._abrir_menu()
                    elif event.key == pygame.K_SPACE:
                        self.alternar_pausa()
                    elif event.key == pygame.K_RIGHT:
                        self.pausado = True   # passo manual implica pausa
                        self.avancar()
                    elif event.key == pygame.K_LEFT:
                        self.pausado = True   # passo manual implica pausa
                        self.retroceder()
        return True

    def atualizar(self, dt_ms):
        """Avanço automático baseado em tempo — não bloqueia o laço."""
        if self.notificacao:
            self.notificacao["tempo_ms"] -= dt_ms
            if self.notificacao["tempo_ms"] <= 0:
                self.notificacao = None

        if self.pausado:
            return
        if self.no_vivo and self.fim_de_jogo:
            return
        self._acumulador_ms += dt_ms
        if self._acumulador_ms >= self.intervalo_ms:
            self._acumulador_ms -= self.intervalo_ms
            self.avancar()