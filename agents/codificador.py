import numpy as np

class CodificadorDomino:
    TAMANHO_VETOR = 79 

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
        """Converte o estado do jogo em um vetor rígido (79, 1)"""
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
        
        return vetor

    def decode_saida(self, logits, jogadas_legais):
        """
        Recebe a saída (58, 1) da rede e aplica a máscara 
        para retornar a jogada legal de maior probabilidade.
        """
        q_valores_mascarados = np.full(58, -np.inf)
        
        for jogada in jogadas_legais:
            if jogada is None:
                idx = 57
            else:
                if jogada[0] != "COMPRAR" and isinstance(jogada[0], list):
                    jogada_formatada = (tuple(jogada[0]), jogada[1])
                else:
                    jogada_formatada = jogada
                idx = self.acao_para_indice[jogada_formatada]
                
            q_valores_mascarados[idx] = logits[idx, 0]
            
        melhor_indice = np.argmax(q_valores_mascarados)
        return self.todas_acoes[melhor_indice]