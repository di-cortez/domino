"""
visibilidade_maos.py

Regras de visibilidade das mãos na HUD.

Esta parte é separada do controlador principal porque a regra é de interface,
não de dominó: ela define o que cada jogador pode ver, conforme o modo de jogo.

Regras atuais:

- IA vs IA: as duas mãos ficam sempre visíveis; J/K não alteram nada.
- Humano vs IA: a mão humana é sempre visível; a mão da IA começa oculta e
  pode ser alternada com J/K.
- Humano vs humano: apenas a mão do jogador da vez fica visível; J/K não
  alteram nada, para permitir uma partida "honesta" no mesmo computador.
"""


class VisibilidadeMaosMixin:
    """
    Mixin usado por `ControladorPartida`.

    O mixin espera que a classe final tenha:
    - `tipos_agentes`
    - `estado_atual()`
    - `_definir_notificacao(texto, tempo_ms=...)`
    """

    def _quantidade_humanos(self):
        return sum(1 for tipo in self.tipos_agentes if tipo == 'humano')

    def _configurar_visibilidade_maos_por_modo(self):
        """
        Recalcula o estado-base de visibilidade quando o modo de jogo muda.

        No modo Humano vs IA, o usuário pode alternar a mão da IA. Nos demais
        modos, a visibilidade é derivada diretamente das regras acima.
        """
        humanos = self._quantidade_humanos()

        if humanos == 1:
            self.maos_visiveis_usuario = [
                tipo == 'humano'
                for tipo in self.tipos_agentes
            ]
            return

        self.maos_visiveis_usuario = [True for _tipo in self.tipos_agentes]

    def pode_alternar_visibilidade_mao(self, jogador):
        humanos = self._quantidade_humanos()

        if humanos != 1:
            return False

        return self.tipos_agentes[jogador] != 'humano'

    def mao_visivel(self, jogador):
        """
        Consulta final usada pela HUD.

        Esta função combina o modo de jogo com o estado escolhido pelo usuário
        para responder se a mão deve ser desenhada aberta ou virada.
        """
        humanos = self._quantidade_humanos()

        if humanos == 0:
            return True

        if humanos >= 2:
            jogador_atual = self.estado_atual().get("jogador_atual", 0)
            return jogador == jogador_atual

        if self.tipos_agentes[jogador] == 'humano':
            return True

        return self.maos_visiveis_usuario[jogador]

    def mao_oculta(self, jogador):
        return not self.mao_visivel(jogador)

    def _alternar_visibilidade_mao(self, jogador):
        """
        Acionado por J/K.

        Quando a regra do modo não permite alternar, apenas mostra uma
        notificação curta. Isso evita que o usuário pense que o comando falhou.
        """
        if not self.pode_alternar_visibilidade_mao(jogador):
            humanos = self._quantidade_humanos()

            if humanos == 0:
                self._definir_notificacao("IA vs IA: mãos sempre visíveis", tempo_ms=1600)
            elif humanos >= 2:
                self._definir_notificacao("Humano vs humano: só a mão da vez aparece", tempo_ms=1800)
            else:
                self._definir_notificacao("A mão do humano fica sempre visível", tempo_ms=1600)

            return

        self.maos_visiveis_usuario[jogador] = not self.maos_visiveis_usuario[jogador]

        estado = "visível" if self.maos_visiveis_usuario[jogador] else "oculta"
        self._definir_notificacao(f"Mão J{jogador}: {estado}", tempo_ms=1400)
