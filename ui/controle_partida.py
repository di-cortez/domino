"""
controle_partida.py

Orquestrador da partida visual.

Este arquivo fica no meio entre três mundos:

- o `MotorDomino`, que conhece as regras do jogo;
- o `GerenciadorPartida`, que chama os agentes automáticos;
- a UI em Pygame/OpenGL, que precisa responder a teclado, pausa, histórico,
  notificações e renderização.

Depois da refatoração, as regras específicas de interação foram separadas em
arquivos menores:

- `agentes_ui.py`: fábrica e nomes dos agentes usados no menu;
- `controle_humano.py`: seleção da mão, ponta escolhida, compra/passe/jogada;
- `visibilidade_maos.py`: regras para mostrar ou ocultar as mãos na HUD.

O controlador continua sendo a "fachada" pública da UI. A HUD e os testes
continuam conversando com `ControladorPartida`, enquanto os detalhes ficam
organizados nos mixins.

Teclas principais:
    M              -> abre/fecha menu de configuração
    R              -> reinício rápido, com confirmação se a partida ainda vive
    Espaço         -> pausa / retoma o avanço automático
    Direita        -> passo para frente no modo automático
    Esquerda       -> passo para trás no histórico
    + / -          -> muda velocidade entre 1/4x, 1/2x, 1x, 2x e 4x
    J / K          -> alterna visibilidade quando o modo permite
    ESC            -> encerra ou fecha menu

Quando há humano jogando, as setas esquerda/direita escolhem uma peça da mão,
setas cima/baixo alternam a ponta quando isso faz sentido, Enter joga, C compra
e P passa.
"""

import copy
import contextlib
import io

import pygame

from ui.agentes_ui import (
    TIPOS_AGENTE as _TIPOS_AGENTE,
    criar_agente_por_tipo,
    nome_tipo_agente,
)
from ui.controle_humano import ControleHumanoMixin
from ui.visibilidade_maos import VisibilidadeMaosMixin


_VELOCIDADES = (0.25, 0.5, 1.0, 2.0, 4.0)
_REINICIO_CONFIRMAR_MS = 2000


class ControladorPartida(VisibilidadeMaosMixin, ControleHumanoMixin):
    """
    Controla uma partida do ponto de vista da UI.

    O papel desta classe não é decidir regra de dominó. O motor faz isso. Aqui
    ficam as decisões visuais e operacionais:

    - manter um histórico de snapshots para voltar/avançar no tempo;
    - pausar e destravar a simulação conforme o modo de jogo;
    - encaminhar turnos de IA ao `GerenciadorPartida`;
    - parar no turno humano até o usuário escolher uma ação;
    - reiniciar a partida mantendo a configuração atual de agentes;
    - guardar notificações temporárias para a HUD desenhar.
    """

    def __init__(self, gerenciador, motor, intervalo_ms=1000, tipos_agentes=None):
        self.gerenciador = gerenciador
        self.motor = motor
        self.intervalo_base_ms = intervalo_ms
        self.velocidade_indice = _VELOCIDADES.index(1.0)
        self.tipos_agentes = list(tipos_agentes) if tipos_agentes else ['neural', 'heuristico']

        # O motor só sabe avançar. Para permitir a seta Esquerda, guardamos
        # snapshots completos do estado visual. `historico_info[i]` é o info da
        # jogada que gerou `historico[i]`; o estado inicial usa None.
        self.historico = []
        self.historico_info = []
        self.indice = 0

        # Estado do avanço automático.
        self.pausado = False
        self.fim_de_jogo = False
        self.info_final = None
        self._acumulador_ms = 0.0

        # O atalho R tem uma confirmação curta quando a partida não acabou.
        # Nesse período a partida pausa, mas volta ao estado anterior se o
        # usuário não confirma dentro de `_REINICIO_CONFIRMAR_MS`.
        self._confirmacao_reinicio_ms = 0
        self._pausado_antes_confirmacao_reinicio = False

        # Notificação temporária desenhada pela HUD:
        # {"texto": str, "tempo_ms": int}
        self.notificacao = None

        # Estado mínimo do turno humano. Os métodos que manipulam estes campos
        # vivem em `ControleHumanoMixin`.
        self.indice_peca_selecionada = 0
        self.ponta_selecionada = 'esquerda'
        self._chave_selecao_humana = None

        # Estado-base de visibilidade. O cálculo final fica em
        # `VisibilidadeMaosMixin.mao_visivel`.
        self.maos_visiveis_usuario = [True, True]

        # Menu de configuração.
        self.menu_aberto = False
        self.menu_cursor = 0
        self._NUM_ITENS_MENU = 3   # J0, J1, Reiniciar
        self._pausado_antes_menu = False

        self._configurar_visibilidade_maos_por_modo()
        self._capturar_estado()
        self._sincronizar_selecao_humana()

    # ------------------------------------------------------------------ #
    # Estado / histórico
    # ------------------------------------------------------------------ #

    def _capturar_estado(self, info=None):
        """
        Congela o estado atual para renderização e navegação no histórico.

        `motor._obter_estado()` entrega o estado básico usado pelos agentes.
        Para a HUD, precisamos também das mãos completas e do monte. Esses
        campos entram apenas no snapshot visual; não alteram o contrato do
        motor nem influenciam as decisões dos agentes.
        """
        estado = copy.deepcopy(self.motor._obter_estado())

        estado_completo = self.motor.to_dict()
        estado["maos"] = copy.deepcopy(estado_completo.get("maos", []))
        estado["monte"] = copy.deepcopy(estado_completo.get("monte", []))

        self.historico.append(estado)
        self.historico_info.append(info)

    @property
    def no_vivo(self):
        """True quando a UI está posicionada no snapshot mais recente."""
        return self.indice >= len(self.historico) - 1

    def estado_atual(self):
        """Snapshot que deve ser desenhado neste frame."""
        return self.historico[self.indice]

    def jogador_humano_ativo(self):
        """
        Informa se a vez atual pertence a um humano controlado pelo teclado.

        A checagem exige estar na ponta viva do histórico. Quando o usuário
        volta para estados passados, a UI só observa, não executa ações humanas.
        """
        if not self.no_vivo or self.fim_de_jogo:
            return False

        jogador = self.estado_atual().get("jogador_atual", 0)
        return self.tipos_agentes[jogador] == 'humano'

    # ------------------------------------------------------------------ #
    # Notificações
    # ------------------------------------------------------------------ #

    def _definir_notificacao(self, texto, tempo_ms=3000):
        self.notificacao = {
            "texto": texto,
            "tempo_ms": tempo_ms,
        }

    def _atualizar_notificacao(self):
        """
        Recria a notificação associada ao snapshot atual.

        Ao navegar pelo histórico, uma notificação de compra deve reaparecer no
        frame correspondente e sumir nos demais. Notificações manuais, como
        velocidade e confirmação de reinício, usam `_definir_notificacao`.
        """
        info = self.historico_info[self.indice]

        if info and info.get("acao") == ("COMPRAR", None):
            jogador = info["jogador_acao"]
            nome = nome_tipo_agente(self.tipos_agentes[jogador])
            self.notificacao = {
                "texto": f"Jogador {jogador} ({nome}) comprou uma peça do estoque",
                "tempo_ms": 3000,
            }
            return

        self.notificacao = None

    # ------------------------------------------------------------------ #
    # Velocidade
    # ------------------------------------------------------------------ #

    @property
    def velocidade(self):
        return _VELOCIDADES[self.velocidade_indice]

    def _texto_velocidade(self):
        textos = {
            0.25: "1/4x",
            0.5: "1/2x",
            1.0: "1x",
            2.0: "2x",
            4.0: "4x",
        }
        return textos[self.velocidade]

    def _intervalo_atual_ms(self):
        """Intervalo real entre turnos automáticos, já considerando velocidade."""
        return self.intervalo_base_ms / self.velocidade

    def _alterar_velocidade(self, delta):
        novo_indice = self.velocidade_indice + delta
        novo_indice = max(0, min(len(_VELOCIDADES) - 1, novo_indice))

        if novo_indice == self.velocidade_indice:
            return

        self.velocidade_indice = novo_indice
        self._acumulador_ms = 0.0
        self._definir_notificacao(f"Velocidade: {self._texto_velocidade()}", tempo_ms=1400)

    @staticmethod
    def _eh_tecla_mais(key):
        return key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS)

    @staticmethod
    def _eh_tecla_menos(key):
        return key in (pygame.K_MINUS, pygame.K_KP_MINUS)

    # ------------------------------------------------------------------ #
    # Reinício rápido com confirmação
    # ------------------------------------------------------------------ #

    def _confirmacao_reinicio_ativa(self):
        return self._confirmacao_reinicio_ms > 0

    def _cancelar_confirmacao_reinicio(self, restaurar_pausa=True):
        """
        Cancela a janela de confirmação do R.

        Hoje o fluxo normal deixa a confirmação expirar por tempo, mas manter
        este método deixa explícita a regra e facilita ajustes futuros.
        """
        if not self._confirmacao_reinicio_ativa():
            return

        self._confirmacao_reinicio_ms = 0

        if restaurar_pausa:
            self.pausado = self._pausado_antes_confirmacao_reinicio

    def _atalho_reiniciar(self):
        """
        Implementa a tecla R.

        - Se a partida acabou, reinicia imediatamente.
        - Se a confirmação já está aberta, o segundo R confirma.
        - Caso contrário, pausa por dois segundos esperando confirmação.
        """
        if self.fim_de_jogo or self._confirmacao_reinicio_ativa():
            self._reiniciar_partida()
            return

        self._pausado_antes_confirmacao_reinicio = self.pausado
        self._confirmacao_reinicio_ms = _REINICIO_CONFIRMAR_MS
        self.pausado = True
        self._acumulador_ms = 0.0
        self._definir_notificacao(
            "Partida ainda não acabou. aperte R novamente para reiniciar",
            tempo_ms=_REINICIO_CONFIRMAR_MS,
        )

    def _atualizar_confirmacao_reinicio(self, dt_ms):
        if not self._confirmacao_reinicio_ativa():
            return

        self._confirmacao_reinicio_ms -= dt_ms

        if self._confirmacao_reinicio_ms <= 0:
            self._confirmacao_reinicio_ms = 0
            self.pausado = self._pausado_antes_confirmacao_reinicio

    # ------------------------------------------------------------------ #
    # Execução de turnos
    # ------------------------------------------------------------------ #

    def _registrar_fim_de_jogo(self, fim, info):
        """Marca fim de jogo e imprime o vencedor uma única vez."""
        if not fim:
            return

        self.fim_de_jogo = True
        self.info_final = info

        if info is not None and "anunciado" not in info:
            print(f"Partida finalizada! Vencedor: Jogador {info.get('vencedor')}")
            info["anunciado"] = True

    def _jogar_turno_com_console_filtrado(self):
        """
        Executa um turno de IA, filtrando o excesso de saída no console.

        O middleware imprime bastante informação. Para a aula, ficou combinado
        manter só as linhas de jogadas possíveis, que ajudam a acompanhar a
        decisão sem despejar as mãos completas dos jogadores.
        """
        buffer = io.StringIO()

        try:
            with contextlib.redirect_stdout(buffer):
                return self.gerenciador.jogar_turno()
        finally:
            for linha in buffer.getvalue().splitlines():
                if "Jogadas Poss" in linha:
                    print(linha)

    def _jogar_proximo_turno(self):
        """
        Avança a ponta viva da partida.

        Se for turno humano, não chama agente nenhum: apenas sincroniza a mão
        selecionada e espera teclado. Se for IA, delega ao gerenciador.
        """
        if self.fim_de_jogo:
            return

        if self.jogador_humano_ativo():
            self._sincronizar_selecao_humana()
            return

        try:
            fim, info = self._jogar_turno_com_console_filtrado()
            self._capturar_estado(info)
            self.indice = len(self.historico) - 1
            self._atualizar_notificacao()
            self._sincronizar_selecao_humana()
            self._registrar_fim_de_jogo(fim, info)
        except Exception as exc:
            print(f"Erro durante o turno: {exc}")
            self.fim_de_jogo = True

    # ------------------------------------------------------------------ #
    # Navegação no histórico
    # ------------------------------------------------------------------ #

    def avancar(self):
        if self.indice < len(self.historico) - 1:
            # Ainda há histórico à frente: apenas reposiciona o cursor.
            self.indice += 1
            self._atualizar_notificacao()
            self._sincronizar_selecao_humana()
            return

        # Estamos na ponta viva: joga um turno real novo.
        self._jogar_proximo_turno()

    def retroceder(self):
        if self.indice > 0:
            self.indice -= 1
            self._atualizar_notificacao()
            self._sincronizar_selecao_humana()

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

    def _atualizar_agente_do_jogador(self, jogador):
        if not hasattr(self.gerenciador, "agentes"):
            return

        self.gerenciador.agentes[jogador] = criar_agente_por_tipo(self.tipos_agentes[jogador])

    def _alternar_tipo_jogador(self, jogador):
        idx = _TIPOS_AGENTE.index(self.tipos_agentes[jogador])
        self.tipos_agentes[jogador] = _TIPOS_AGENTE[(idx + 1) % len(_TIPOS_AGENTE)]
        self._atualizar_agente_do_jogador(jogador)
        self._configurar_visibilidade_maos_por_modo()
        self._sincronizar_selecao_humana()

    def _ativar_item_menu(self):
        if self.menu_cursor == 0:
            self._alternar_tipo_jogador(0)
        elif self.menu_cursor == 1:
            self._alternar_tipo_jogador(1)
        elif self.menu_cursor == 2:
            self._reiniciar_partida()
            self._fechar_menu()

    def _reiniciar_partida(self):
        """
        Reinicia a partida preservando o modo de jogadores selecionado.

        O motor é resetado, os agentes são recriados pela fábrica e o histórico
        volta a conter apenas o snapshot inicial da nova partida.
        """
        from middleware.middleware import GerenciadorPartida

        self.motor.reset()
        novos_agentes = [
            criar_agente_por_tipo(tipo)
            for tipo in self.tipos_agentes
        ]
        self.gerenciador = GerenciadorPartida(self.motor, novos_agentes)

        self.historico = []
        self.historico_info = []
        self.indice = 0
        self.pausado = False
        self.fim_de_jogo = False
        self.info_final = None
        self.notificacao = None
        self._acumulador_ms = 0.0
        self._confirmacao_reinicio_ms = 0
        self._pausado_antes_confirmacao_reinicio = False
        self._chave_selecao_humana = None

        self._configurar_visibilidade_maos_por_modo()
        self._capturar_estado()
        self._sincronizar_selecao_humana()

    # ------------------------------------------------------------------ #
    # Entrada e laço de atualização
    # ------------------------------------------------------------------ #

    def processar_entrada(self):
        """Consome eventos do pygame. Retorna False quando a janela deve fechar."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type != pygame.KEYDOWN:
                continue

            if self.menu_aberto:
                if event.key in (pygame.K_ESCAPE, pygame.K_m):
                    self._fechar_menu()
                elif event.key == pygame.K_UP:
                    self.menu_cursor = (self.menu_cursor - 1) % self._NUM_ITENS_MENU
                elif event.key == pygame.K_DOWN:
                    self.menu_cursor = (self.menu_cursor + 1) % self._NUM_ITENS_MENU
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                    self._ativar_item_menu()
                continue

            if event.key == pygame.K_ESCAPE:
                return False
            if event.key == pygame.K_r:
                self._atalho_reiniciar()
                continue

            # Enquanto o R aguarda confirmação, ignoramos atalhos que poderiam
            # avançar ou alterar a partida. O segundo R foi tratado acima.
            if self._confirmacao_reinicio_ativa():
                continue

            if event.key == pygame.K_m:
                self._abrir_menu()
            elif event.key == pygame.K_j:
                self._alternar_visibilidade_mao(0)
            elif event.key == pygame.K_k:
                self._alternar_visibilidade_mao(1)
            elif self._eh_tecla_mais(event.key):
                self._alterar_velocidade(1)
            elif self._eh_tecla_menos(event.key):
                self._alterar_velocidade(-1)
            elif self.jogador_humano_ativo():
                self._processar_tecla_humana(event.key)
            else:
                self._processar_tecla_automatica(event.key)

        return True

    def _processar_tecla_humana(self, key):
        """Atalhos válidos quando o jogador da vez é humano."""
        self._sincronizar_selecao_humana()

        if key == pygame.K_RIGHT:
            self._navegar_peca_humana(1)
        elif key == pygame.K_LEFT:
            self._navegar_peca_humana(-1)
        elif key in (pygame.K_UP, pygame.K_DOWN, pygame.K_TAB):
            self._alternar_ponta_humana()
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._jogar_peca_humana()
        elif key == pygame.K_c:
            self._comprar_humana()
        elif key == pygame.K_p:
            self._passar_humana()

    def _processar_tecla_automatica(self, key):
        """Atalhos válidos quando a partida está em modo automático/observação."""
        if key == pygame.K_SPACE:
            self.alternar_pausa()
        elif key == pygame.K_RIGHT:
            self.pausado = True
            self.avancar()
        elif key == pygame.K_LEFT:
            self.pausado = True
            self.retroceder()

    def atualizar(self, dt_ms):
        """
        Atualiza temporizadores e avança a partida quando permitido.

        O método é chamado uma vez por frame pelo `main_visual.py`. Ele não
        bloqueia o laço: apenas acumula tempo e executa um turno quando o
        intervalo configurado é atingido.
        """
        if self.notificacao:
            self.notificacao["tempo_ms"] -= dt_ms
            if self.notificacao["tempo_ms"] <= 0:
                self.notificacao = None

        self._atualizar_confirmacao_reinicio(dt_ms)

        if self.jogador_humano_ativo():
            self._sincronizar_selecao_humana()
            return

        if self.pausado:
            return

        if self.no_vivo and self.fim_de_jogo:
            return

        self._acumulador_ms += dt_ms
        intervalo_ms = self._intervalo_atual_ms()

        if self._acumulador_ms >= intervalo_ms:
            self._acumulador_ms -= intervalo_ms
            self.avancar()
