import numpy as np

class CodificadorDomino:
    TAMANHO_VETOR = 86

    def __init__(self):
        self.todas_pecas = [(i, j) for i in range(7) for j in range(i, 7)]
        
        self.todas_acoes = []
        for peca in self.todas_pecas:
            self.todas_acoes.append((peca, 0)) # Índices 0-27: Esquerda
        for peca in self.todas_pecas:
            self.todas_acoes.append((peca, 1)) # Índices 28-55: Direita
            
        self.todas_acoes.append(("COMPRAR", None)) # Índice 56
        self.todas_acoes.append(None)              # Índice 57
        
        self.acao_para_indice = {acao: idx for idx, acao in enumerate(self.todas_acoes)}

    def encode_estado(self, estado):
        """Converte o estado do jogo em um vetor rígido (86, 1)"""
        vetor = np.zeros((self.TAMANHO_VETOR, 1))
        
        # 1. Mão do Jogador (Índices 0-27)
        for peca in estado["mao_jogador"]:
            idx = self.todas_pecas.index(tuple(peca))
            vetor[idx, 0] = 1.0
            
        # 2. Ponta Esquerda (Índices 28-34)
        if estado["pontas"]:
            vetor[28 + estado["pontas"][0], 0] = 1.0
            
        # 3. Ponta Direita (Índices 35-41)
        if estado["pontas"]:
            vetor[35 + estado["pontas"][1], 0] = 1.0
            
        # 4. Tamanhos das mãos (Índices 42-48)
        # Correção da Issue 3: Codifica todos os jogadores, incluindo o próprio, para evitar ambiguidade de 0.0
        for i, tamanho in enumerate(estado["tamanhos_maos"]):
            vetor[42 + i, 0] = tamanho / 7.0
                
        # 5. Tamanho do Monte (Índice 49)
        vetor[49, 0] = estado.get("monte_tamanho", 0) / 14.0
        
        # 6. Histórico da Mesa (Índices 50-77) e Contagem de Compras (Índice 78)
        draws = 0
        for acao in estado.get("historico_mesa", []):
            if acao is None:
                continue
            
            # Correção da Issue 2: Conta as ações de compra para extrair urgência estratégica
            if acao == ["COMPRAR", None] or acao == ("COMPRAR", None):
                draws += 1
                continue
            
            peca = tuple(acao[0])
            idx_historico = self.todas_pecas.index(peca)
            vetor[50 + idx_historico, 0] = 1.0
            
        # Adiciona a contagem de compras no último índice (Normalizado por 14, limite do jogo de 2 jogadores)
        vetor[78, 0] = draws / 14.0

        # 7. Naipes Mortos do Oponente (Índices 79-85)
        # Um bit por valor (0-6) que o oponente já provou não ter (ver
        # middleware.motor_domino.inferir_naipes_mortos).
        for valor in estado.get("naipes_mortos_oponente", []):
            vetor[79 + valor, 0] = 1.0

        return vetor

    def _indice_da_jogada(self, jogada):
        """Resolve o índice (0-57) de uma jogada, aceitando peça como tupla ou lista."""
        if jogada is None:
            return 57
        if jogada[0] != "COMPRAR" and isinstance(jogada[0], list):
            jogada = (tuple(jogada[0]), jogada[1])
        return self.acao_para_indice[jogada]

    def decode_saida(self, logits, jogadas_legais):
        """
        Recebe a saída (58, 1) da rede e aplica a máscara
        para retornar a jogada legal de maior probabilidade.
        """
        q_valores_mascarados = np.full(58, -np.inf)

        for jogada in jogadas_legais:
            idx = self._indice_da_jogada(jogada)
            q_valores_mascarados[idx] = logits[idx, 0]

        melhor_indice = np.argmax(q_valores_mascarados)
        return self.todas_acoes[melhor_indice]

    def amostrar_acao(self, probabilidades, jogadas_legais):
        """
        Versão estocástica de decode_saida: amostra uma jogada entre as legais
        proporcionalmente à probabilidade que a rede atribuiu a cada uma
        (renormalizada só sobre a máscara de jogadas legais).

        Usada pelo AgenteRL durante o self-play (exploração); decode_saida
        (argmax) continua sendo a política de inferência determinística usada
        pelo AgenteNeuralNumPy e pelo AgenteRL em modo de avaliação.

        :return: (jogada_escolhida, indice_da_jogada_escolhida)
        """
        indices_legais = [self._indice_da_jogada(jogada) for jogada in jogadas_legais]

        probs_legais = probabilidades[indices_legais, 0]
        soma = probs_legais.sum()
        if soma <= 0:
            probs_legais = np.ones(len(indices_legais)) / len(indices_legais)
        else:
            probs_legais = probs_legais / soma

        posicao_escolhida = np.random.choice(len(indices_legais), p=probs_legais)
        return jogadas_legais[posicao_escolhida], indices_legais[posicao_escolhida]