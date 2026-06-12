import random
import copy

class MotorDomino:
    def __init__(self, num_jogadores=2):
        self.num_jogadores = num_jogadores
        self.todas_pecas = [(i, j) for i in range(7) for j in range(i, 7)]
        self.reset()

    def print_pecas(self):
        # FIX 4: Idiomatic loop
        for peca in self.todas_pecas:
            print(peca)

    def reset(self):
        """
        Inicia uma nova partida. Embaralha, distribui as peças e define quem começa.
        """
        pecas_embaralhadas = self.todas_pecas.copy()
        random.shuffle(pecas_embaralhadas)

        self.maos = []
        for i in range(self.num_jogadores):
            self.maos.append(pecas_embaralhadas[i*7 : (i+1)*7])
        
        self.monte = pecas_embaralhadas[self.num_jogadores*7 :]
        self.mesa = [] 
        self.cadeia_mesa = [] 
        self.pontas = []      
        self.vencedor = None
        self.fim_de_jogo = False
        self.turno = 0 
        
        self.passes_consecutivos = 0 
        self.comprou_neste_turno = {i: False for i in range(self.num_jogadores)} 
        # self.comprou_neste turno = {0: False, 1: False}

        # ===============================================================
        # RETIRAR CADEIA_MESA E INFORMACAO VISUAL
        # ===============================================================
        #         
        # Track horizontal direction [left_side_dir, right_side_dir]
        # -1 = moving left, 1 = moving right
        self.direcao_horizontal = [-1, 1] 

        self.jogador_atual = 0
        maior_duplo = -1
        for i, mao in enumerate(self.maos):
            for peca in mao:
                if peca[0] == peca[1] and peca[0] > maior_duplo:
                    maior_duplo = peca[0]
                    self.jogador_atual = i
                    
        self.primeira_peca_obrigatoria = None
        if maior_duplo != -1:
            self.primeira_peca_obrigatoria = (maior_duplo, maior_duplo)

        return self._obter_estado()

    def acoes_validas(self, jogador=None):
        """
        Retorna uma lista de ações possíveis sem duplicatas.
        Se jogador is None, avalia o jogador_atual.
        """
        if jogador is None:
            jogador = self.jogador_atual
            
        mao = self.maos[jogador]
        
        if not self.pontas:
            if self.primeira_peca_obrigatoria and self.primeira_peca_obrigatoria in mao:
                return [(self.primeira_peca_obrigatoria, 0)]
            return list(set([(peca, 0) for peca in mao])) 
        
        acoes = set() 
        ponta_esq, ponta_dir = self.pontas

        for peca in mao:
            if peca[0] == ponta_esq or peca[1] == ponta_esq:
                acoes.add((peca, 0))
            if peca[0] == ponta_dir or peca[1] == ponta_dir:
                acoes.add((peca, 1))

        if ponta_esq == ponta_dir:
            acoes = {(peca, 0) for (peca, lado) in acoes}

        if not acoes:
            if self.monte and not self.comprou_neste_turno[jogador]:
                return [("COMPRAR", None)]
            return [None]
            
        return list(acoes)

    def step(self, acao):
        """
        Executa uma jogada e retorna (estado, fim, info).
        """
        if self.fim_de_jogo:
            raise Exception("O jogo já acabou. Chame reset().")

        mao = self.maos[self.jogador_atual]
        jogadas_possiveis = self.acoes_validas(self.jogador_atual)

        if acao is None and jogadas_possiveis != [None]:
            raise ValueError(f"Ação inválida: O jogador {self.jogador_atual} não pode passar a vez. Peças/Monte disponíveis.")
        if acao is not None and acao not in jogadas_possiveis:
            raise ValueError(f"Ação inválida: {acao} não é permitida neste momento.")

        avancar_jogador = True 

        if acao == ("COMPRAR", None):
            peca_comprada = self.monte.pop(0)
            mao.append(peca_comprada)
            self.mesa.append(acao)
            
            self.comprou_neste_turno[self.jogador_atual] = True
            avancar_jogador = False 
            
        elif acao is not None:
            peca, lado = acao
            mao.remove(peca)
            self.mesa.append(acao)
            
            self.passes_consecutivos = 0
            self.comprou_neste_turno[self.jogador_atual] = False
            
            # lógica para jgoada da primeira peça
            if self.primeira_peca_obrigatoria and peca == self.primeira_peca_obrigatoria:
                self.primeira_peca_obrigatoria = None

            eh_duplo = (peca[0] == peca[1])
            orientacao = "vertical" if eh_duplo else "horizontal"

            info_visual = {
                "peca": list(peca),
                "lado_jogado": lado,
                "indice_turno": self.turno,
                "orientacao": orientacao,
                "valor_conectado": None,
                "valor_exposto": None
            }

            if not self.pontas:
                self.pontas = [peca[0], peca[1]]
                info_visual["lado_jogado"] = None
                self.cadeia_mesa.append(info_visual)
            else:
                ponta_esq, ponta_dir = self.pontas
                
                if lado == 0: 
                    valor_conectado = ponta_esq
                    if peca[0] == ponta_esq:
                        nova_ponta = peca[1]
                    elif peca[1] == ponta_esq:
                        nova_ponta = peca[0]
                    else:
                        raise ValueError(f"Peça {peca} não conecta à ponta esquerda {ponta_esq}")
                        
                    self.pontas[0] = nova_ponta
                    info_visual["valor_conectado"] = valor_conectado
                    info_visual["valor_exposto"] = nova_ponta
                    
                    self.cadeia_mesa.insert(0, info_visual)

                elif lado == 1: 
                    valor_conectado = ponta_dir
                    if peca[0] == ponta_dir:
                        nova_ponta = peca[1]
                    elif peca[1] == ponta_dir:
                        nova_ponta = peca[0]
                    else:
                        raise ValueError(f"Peça {peca} não conecta à ponta direita {ponta_dir}")
                        
                    self.pontas[1] = nova_ponta
                    info_visual["valor_conectado"] = valor_conectado
                    info_visual["valor_exposto"] = nova_ponta
                    
                    self.cadeia_mesa.append(info_visual)
                    
        else:
            self.mesa.append(None) # passou a vez, mesa recebe None
            self.passes_consecutivos += 1
            self.comprou_neste_turno[self.jogador_atual] = False

        self.turno += 1

        if len(mao) == 0:
            self.fim_de_jogo = True
            self.vencedor = self.jogador_atual 

        elif self.passes_consecutivos >= self.num_jogadores and not self.monte:
            self.fim_de_jogo = True
            somas = [sum(p[0] + p[1] for p in m) for m in self.maos]
            menor_soma = min(somas)
            vencedores_possiveis = [i for i, soma in enumerate(somas) if soma == menor_soma]
            
            if len(vencedores_possiveis) == 1:
                self.vencedor = vencedores_possiveis[0]
            else:
                self.vencedor = -1 

        if not self.fim_de_jogo and avancar_jogador:
            self.jogador_atual = (self.jogador_atual + 1) % self.num_jogadores

        return self._obter_estado(), self.fim_de_jogo, {"vencedor": self.vencedor}

    def _obter_estado(self):
        """
        Retorna o dicionário de estado enriquecido.
        """
        tamanhos_maos = [len(m) for m in self.maos]
        
        return {
            "pontas": list(self.pontas),
            "mao_jogador": [list(p) for p in self.maos[self.jogador_atual]], # FIX 3: Tuple purging
            "jogador_atual": self.jogador_atual,
            "turno": self.turno,
            "tamanhos_maos": tamanhos_maos,
            "historico_mesa": [self._serializar_acao(a) for a in self.mesa], # FIX 2: Tuple purging
            "cadeia_visual": copy.deepcopy(self.cadeia_mesa),
            "monte_tamanho": len(self.monte)
        }

    def _serializar_acao(self, acao):
        """
        Helper method to serialize tuples to lists for JSON compatibility.
        """
        if acao is None:
            return None
        if acao == ("COMPRAR", None):
            return ["COMPRAR", None]
        return [list(acao[0]), acao[1]]

    def to_dict(self):
        """
        Gera um dicionário serializável do ambiente completo. 
        """
        return {
            "num_jogadores": self.num_jogadores,
            "jogador_atual": self.jogador_atual,
            "pontas": list(self.pontas),
            "mesa_logica": [self._serializar_acao(a) for a in self.mesa],
            "mesa_visual": copy.deepcopy(self.cadeia_mesa), 
            "maos": [[list(p) for p in mao] for mao in self.maos], 
            "monte": [list(p) for p in self.monte], 
            "turno": self.turno,
            "fim_de_jogo": self.fim_de_jogo,
            "vencedor": self.vencedor
        }