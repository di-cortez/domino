"""
controle_humano.py

Lógica específica do jogador humano.

O motor de dominó já sabe validar ações (`motor.acoes_validas`) e executar
jogadas (`motor.step`). Este módulo não reimplementa a regra do jogo; ele só
mantém o estado de interação da UI:

- qual peça da mão está selecionada;
- em qual ponta o humano pretende jogar;
- como comprar, passar e jogar a peça selecionada;
- onde a seta amarela deve ser desenhada na peça da mão.

O arquivo é um mixin para manter `ControladorPartida` pequeno sem criar uma
camada nova de objetos. Os métodos continuam disponíveis no controlador final,
o que preserva os testes unitários e a HUD.
"""


PONTA_PARA_LADO = {'esquerda': 0, 'direita': 1}
LADO_PARA_PONTA = {0: 'esquerda', 1: 'direita'}


class ControleHumanoMixin:
    """
    Mixin usado por `ControladorPartida`.

    O mixin espera que a classe final tenha:
    - `motor`
    - `fim_de_jogo`
    - `jogador_humano_ativo()`
    - `_capturar_estado(info)`
    - `_atualizar_notificacao()`
    - `_definir_notificacao(texto, tempo_ms=...)`
    - `_registrar_fim_de_jogo(fim, info)`
    """

    def _normalizar_acao(self, acao):
        """
        Normaliza ações vindas do motor.

        O motor pode expor peças como listas em snapshots, enquanto as ações
        válidas usam tuplas. Para comparar de modo estável, convertemos sempre
        a peça para tupla e preservamos `None`/`COMPRAR`.
        """
        if acao is None:
            return None
        if acao == ("COMPRAR", None):
            return ("COMPRAR", None)

        peca, lado = acao
        return tuple(peca), lado

    def _acoes_validas_set(self, jogador):
        return {
            self._normalizar_acao(acao)
            for acao in self.motor.acoes_validas(jogador)
        }

    def _acao_peca_lado(self, peca, lado):
        return tuple(peca), lado

    def _pontas_iguais(self):
        return len(self.motor.pontas) == 2 and self.motor.pontas[0] == self.motor.pontas[1]

    def _lados_validos_para_peca(self, peca):
        """
        Devolve nomes de pontas (`esquerda`/`direita`) onde a peça encaixa.

        Quando as pontas da mesa têm o mesmo número, jogar na esquerda ou na
        direita é logicamente equivalente. Para manter a UI previsível, a
        interface mostra sempre `direita` nesse caso, mas a ação enviada ao
        motor continua sendo uma ação válida para o motor.
        """
        jogador = self.motor.jogador_atual
        validas = self._acoes_validas_set(jogador)

        if self._pontas_iguais() and self._acao_peca_lado(peca, 0) in validas:
            return ['direita']

        lados = []

        for ponta, lado in PONTA_PARA_LADO.items():
            if self._acao_peca_lado(peca, lado) in validas:
                lados.append(ponta)

        return lados

    def _ajustar_ponta_para_peca_selecionada(self, preferir_atual=True):
        jogador = self.motor.jogador_atual
        mao = self.motor.maos[jogador]

        if not mao:
            self.ponta_selecionada = 'esquerda'
            return

        self.indice_peca_selecionada %= len(mao)
        peca = mao[self.indice_peca_selecionada]
        lados = self._lados_validos_para_peca(peca)

        if not lados:
            return

        if preferir_atual and self.ponta_selecionada in lados:
            return

        # Preferir direita ajuda nos casos equivalentes e deixa a seta menos
        # instável ao navegar por peças que encaixam nos dois lados.
        if 'direita' in lados:
            self.ponta_selecionada = 'direita'
        else:
            self.ponta_selecionada = lados[0]

    def _selecionar_primeira_jogada_valida(self):
        """
        Escolhe a peça inicial quando começa um turno humano.

        A prioridade é: primeira peça jogável na mão. Se nenhuma peça encaixa,
        mantemos uma peça selecionada só para navegação visual.
        """
        jogador = self.motor.jogador_atual
        mao = self.motor.maos[jogador]

        if not mao:
            self.indice_peca_selecionada = 0
            self.ponta_selecionada = 'esquerda'
            return

        validas = self._acoes_validas_set(jogador)

        for indice, peca in enumerate(mao):
            if any(self._acao_peca_lado(peca, lado) in validas for lado in LADO_PARA_PONTA):
                self.indice_peca_selecionada = indice
                self._ajustar_ponta_para_peca_selecionada(preferir_atual=False)
                return

        self.indice_peca_selecionada = min(self.indice_peca_selecionada, len(mao) - 1)
        self._ajustar_ponta_para_peca_selecionada(preferir_atual=False)

    def _sincronizar_selecao_humana(self):
        """
        Mantém a seleção coerente com o estado vivo do motor.

        Sempre que muda jogador, turno ou mão do jogador, recalculamos a peça
        inicial. Se nada mudou, apenas garantimos que o índice ainda cabe na mão.
        """
        if not self.jogador_humano_ativo():
            self._chave_selecao_humana = None
            return

        jogador = self.motor.jogador_atual
        mao = tuple(tuple(peca) for peca in self.motor.maos[jogador])
        chave = (jogador, self.motor.turno, mao)

        if chave != self._chave_selecao_humana:
            self._selecionar_primeira_jogada_valida()
            self._chave_selecao_humana = chave
            return

        if mao:
            self.indice_peca_selecionada %= len(mao)
        else:
            self.indice_peca_selecionada = 0

    def _navegar_peca_humana(self, delta):
        """Move a seleção da mão em modo circular."""
        self._sincronizar_selecao_humana()

        jogador = self.motor.jogador_atual
        mao = self.motor.maos[jogador]

        if not mao:
            self.indice_peca_selecionada = 0
            return

        self.indice_peca_selecionada = (
            self.indice_peca_selecionada + delta
        ) % len(mao)
        self._ajustar_ponta_para_peca_selecionada(preferir_atual=False)

    def _alternar_ponta_humana(self):
        """
        Alterna a ponta da jogada para a peça selecionada.

        Só alterna quando a peça realmente pode jogar nos dois lados. Se ela
        só encaixa em uma ponta, a UI mantém a ponta correta e avisa o usuário.
        """
        jogador = self.motor.jogador_atual
        mao = self.motor.maos[jogador]

        if not mao:
            return

        self.indice_peca_selecionada %= len(mao)
        peca = mao[self.indice_peca_selecionada]
        lados = self._lados_validos_para_peca(peca)

        if len(lados) < 2:
            if lados:
                self._definir_notificacao(f"Essa peça só joga na ponta {lados[0]}", tempo_ms=1400)
            else:
                self._definir_notificacao("Essa peça não encaixa", tempo_ms=1400)
            return

        if self.ponta_selecionada == 'esquerda':
            self.ponta_selecionada = 'direita'
        else:
            self.ponta_selecionada = 'esquerda'

    def _acao_humana_selecionada(self):
        """Monta a ação correspondente à peça/ponta selecionadas."""
        self._sincronizar_selecao_humana()

        jogador = self.motor.jogador_atual
        mao = self.motor.maos[jogador]

        if not mao:
            return None

        self.indice_peca_selecionada %= len(mao)
        peca = mao[self.indice_peca_selecionada]
        lado = PONTA_PARA_LADO[self.ponta_selecionada]

        acao = self._acao_peca_lado(peca, lado)

        if acao in self._acoes_validas_set(jogador):
            return acao

        if self._pontas_iguais():
            acao_equivalente = self._acao_peca_lado(peca, 0)

            if acao_equivalente in self._acoes_validas_set(jogador):
                return acao_equivalente

        return acao

    def posicao_seta_peca_selecionada(self):
        """
        Calcula se a seta da HUD deve ficar em cima ou embaixo da peça.

        A seta não representa apenas "ponta esquerda/direita"; ela aponta para
        a metade da peça que conecta na mesa. Ex.: com pontas 2 e 3, a peça
        (1, 2) deve mostrar seta embaixo, porque o valor conectado é o 2.
        """
        if not self.jogador_humano_ativo():
            return None

        jogador = self.motor.jogador_atual
        mao = self.motor.maos[jogador]

        if not mao:
            return None

        self.indice_peca_selecionada %= len(mao)
        peca = mao[self.indice_peca_selecionada]

        if not self.motor.pontas:
            return "baixo"

        if self._pontas_iguais():
            valor_conectado = self.motor.pontas[0]
        else:
            lado = PONTA_PARA_LADO[self.ponta_selecionada]
            valor_conectado = self.motor.pontas[lado]

        if peca[1] == valor_conectado:
            return "baixo"

        if peca[0] == valor_conectado:
            return "cima"

        return "baixo"

    def _executar_acao_humana(self, acao):
        """
        Executa compra, passe ou jogada de peça do humano.

        A execução real fica no motor. Depois do `step`, o controlador captura
        um snapshot visual e atualiza fim de jogo/notificações.
        """
        if self.fim_de_jogo:
            return

        jogador = self.motor.jogador_atual

        try:
            _estado, fim, info = self.motor.step(acao)
        except ValueError as exc:
            print(exc)
            self._definir_notificacao("Jogada inválida")
            return

        info["acao"] = acao
        info["jogador_acao"] = jogador

        self._capturar_estado(info)
        self.indice = len(self.historico) - 1
        self._atualizar_notificacao()
        self._sincronizar_selecao_humana()
        self._registrar_fim_de_jogo(fim, info)

        # Se a jogada humana passou a vez para IA, destrava o avanço automático.
        if not self.fim_de_jogo and not self.jogador_humano_ativo():
            self.pausado = False
            self._acumulador_ms = 0.0

    def _jogar_peca_humana(self):
        acao = self._acao_humana_selecionada()

        if acao is None:
            self._definir_notificacao("Sem peça selecionada")
            return

        jogador = self.motor.jogador_atual
        if acao not in self._acoes_validas_set(jogador):
            self._definir_notificacao("Essa peça não encaixa nessa ponta")
            return

        self._executar_acao_humana(acao)

    def _comprar_humana(self):
        jogador = self.motor.jogador_atual

        if ("COMPRAR", None) not in self._acoes_validas_set(jogador):
            self._definir_notificacao("Compra não permitida agora")
            return

        self._executar_acao_humana(("COMPRAR", None))

    def _passar_humana(self):
        jogador = self.motor.jogador_atual

        if None not in self._acoes_validas_set(jogador):
            self._definir_notificacao("Passe não permitido agora")
            return

        self._executar_acao_humana(None)
