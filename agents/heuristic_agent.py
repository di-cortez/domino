import random

from agents.agent import Agente

class AgenteEstrategico(Agente):
    """
    Agente 'Professor' baseado em heurísticas de utilidade.
    Otimizado estritamente para Supervised Learning (SL) sem ruído exploratório.
    """
    def escolher_jogada(self, estado, jogadas_legais):
        if jogadas_legais == [None] or jogadas_legais == [("COMPRAR", None)]:
            return jogadas_legais[0]

        melhor_jogada = None
        maior_utilidade = -float('inf')
        
        mao = estado["mao_jogador"]
        pontas_atuais = estado["pontas"]
        jogador_atual = estado["jogador_atual"]
        tamanhos = estado["tamanhos_maos"]

        oponentes = [t for i, t in enumerate(tamanhos) if i != jogador_atual]
        fator_urgencia = 2.0 if min(oponentes) <= 2 else 1.0

        for jogada in jogadas_legais:
            # print(jogada)
            peca, lado = jogada
            
            # print(f"MAO: {mao}")
            
            mao_restante = list(mao) 
            # print(f"MAO_RESTANTE:{type(mao_restante)}")
            # print(f"PECA:{type(peca)}")
            mao_restante.remove(list(peca))
            
            # Issue 1 Fix: Evitar dupla contagem de peso para duplos
            eh_duplo = (peca[0] == peca[1])
            if eh_duplo:
                peso = 0
                bonus_duplo = peca[0] * 3.5
            else:
                peso = peca[0] + peca[1]
                bonus_duplo = 0
            
            diversidade = 0
            cobertura_ponta = 0
            
            if pontas_atuais:
                valor_conectado = pontas_atuais[lado]
                
                if peca[0] == valor_conectado:
                    nova_ponta = peca[1]
                elif peca[1] == valor_conectado:
                    nova_ponta = peca[0]
                else:
                    nova_ponta = None
                    
                if nova_ponta is not None:
                    # Issue 2 Fix: Medir diversidade contra o estado da mesa PÓS-JOGADA
                    pontas_apos = list(pontas_atuais)
                    pontas_apos[lado] = nova_ponta
                    
                    diversidade = sum(1 for p in mao_restante if (p[0] in pontas_apos or p[1] in pontas_apos)) * 3 
                    cobertura_ponta = sum(1 for p in mao_restante if nova_ponta in p) * 2 
            else:
                numeros_restantes = set()
                for p in mao_restante:
                    numeros_restantes.add(p[0])
                    numeros_restantes.add(p[1])
                diversidade = len(numeros_restantes) * 2
                cobertura_ponta = 0

            # Issue 3 Fix: Aplicar urgência sobre o peso E sobre o bônus duplo
            utilidade_total = ((peso + bonus_duplo) * fator_urgencia) + diversidade + cobertura_ponta

            if utilidade_total > maior_utilidade:
                maior_utilidade = utilidade_total
                melhor_jogada = jogada

        return melhor_jogada