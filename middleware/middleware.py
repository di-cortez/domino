class Agente:
    """
    Classe base (Interface) para todos os jogadores (Humano, Heurístico, Neural).
    """
    def escolher_jogada(self, estado, jogadas_legais):
        """
        Recebe o estado atual e a lista de jogadas válidas.
        Deve retornar uma jogada presente em 'jogadas_legais'.
        """
        raise NotImplementedError("Todos os agentes devem implementar este método.")


class GerenciadorPartida:
    """Controla o fluxo do jogo entre o Motor e os Agentes, registrando dados para SL."""
    def __init__(self, motor, agentes):
        if len(agentes) != motor.num_jogadores:
            raise ValueError("O número de agentes deve ser igual ao número de jogadores.")
        self.motor = motor
        self.agentes = agentes
        self.historico_treinamento = [] # Armazena os pares (Estado, Ação)

    def jogar_turno(self):
        estado = self.motor._obter_estado()
        jogador_atual = estado["jogador_atual"]
        jogadas_legais = self.motor.acoes_validas(jogador_atual)
        
        agente_da_vez = self.agentes[jogador_atual]
        acao_escolhida = agente_da_vez.escolher_jogada(estado, jogadas_legais)
        
        # LOGGING PARA SUPERVISED LEARNING
        # Salvamos o estado exato e a ação que o agente "Professor" decidiu tomar.
        # cadeia_visual é excluída pois é metadata de renderização, nunca usada pelo encoder.
        estado_treino = {k: v for k, v in estado.items() if k != "cadeia_visual"}
        self.historico_treinamento.append({
            "estado": estado_treino,
            "acao_alvo": acao_escolhida
        })
        
        estado_atualizado, fim_de_jogo, info = self.motor.step(acao_escolhida)
        info["acao"]         = acao_escolhida
        info["jogador_acao"] = jogador_atual
        return fim_de_jogo, info

    def jogar_partida_completa(self):
        self.historico_treinamento = [] # Reseta o log para a nova partida
        fim_de_jogo = False
        info = {}
        
        while not fim_de_jogo:
            fim_de_jogo, info = self.jogar_turno()
            
        return info, self.historico_treinamento