import random

from agents.agent import Agente
from middleware.motor_domino import inferir_naipes_mortos


class AgenteEstrategico(Agente):
    """
    Agente 'Professor' baseado em heurísticas de utilidade.

    Avalia cada jogada legal por uma função de utilidade que combina:
      - descarte de peso (pips), com bônus para duplos (peças inflexíveis);
      - diversidade da mão restante e cobertura da ponta recém-criada;
      - BLOQUEIO: bônus para jogadas que deixam pontas em "naipes mortos"
        do oponente — valores que ele demonstrou não ter ao passar ou
        comprar (inferidos do histórico da mesa);
      - urgência quando algum oponente está prestes a bater (<= 2 peças),
        que amplifica tanto o descarte de peso quanto o bloqueio.

    Determinístico por padrão (rótulos estáveis para o dataset de SL);
    `ruido_desempate` > 0 sorteia entre jogadas de utilidade quase igual,
    útil para diversificar os estados visitados na geração de dataset.

    Os pesos são parâmetros do construtor para permitir calibração empírica
    (ex.: busca em grade jogando partidas contra a configuração atual).
    """

    def __init__(
        self,
        peso_duplo=3.5,
        peso_diversidade=3.0,
        peso_cobertura=2.0,
        peso_bloqueio=8.0,
        ruido_desempate=0.0,
    ):
        self.peso_duplo = peso_duplo
        self.peso_diversidade = peso_diversidade
        self.peso_cobertura = peso_cobertura
        self.peso_bloqueio = peso_bloqueio
        self.ruido_desempate = ruido_desempate

    def _inferir_ausencias_oponentes(self, estado):
        """
        Reconstrói do histórico os "naipes mortos" de cada jogador via
        middleware.motor_domino.inferir_naipes_mortos (mesmo motor usa essa
        lógica para preencher a chave "naipes_mortos_oponente" do estado).

        Aproximação consciente: a ausência é tratada como permanente, mas o
        oponente pode vir a comprar uma peça do naipe morto depois. Com 2
        jogadores o erro é pequeno e rastrear a incerteza custaria muito.
        """
        return inferir_naipes_mortos(
            estado.get("historico_mesa", []),
            estado["tamanhos_maos"],
            estado["jogador_atual"],
        )

    def escolher_jogada(self, estado, jogadas_legais):
        if not jogadas_legais:
            return None

        # Ações forçadas (passar/comprar) não têm o formato (peça, lado);
        # o guarda é por tipo de ação, não por igualdade exata com a lista,
        # para não quebrar se o motor um dia misturar as opções.
        jogadas_de_peca = [
            j for j in jogadas_legais if j is not None and j[0] != "COMPRAR"
        ]
        if not jogadas_de_peca:
            return jogadas_legais[0]

        # Normaliza para tuplas: desacopla do formato de serialização do
        # estado (o motor envia a mão como listas via _obter_estado).
        mao = [tuple(p) for p in estado["mao_jogador"]]
        pontas_atuais = estado["pontas"]
        jogador_atual = estado["jogador_atual"]
        tamanhos = estado["tamanhos_maos"]

        oponentes = [t for i, t in enumerate(tamanhos) if i != jogador_atual]
        fator_urgencia = 2.0 if oponentes and min(oponentes) <= 2 else 1.0

        ausencias = self._inferir_ausencias_oponentes(estado)
        valores_mortos = set()
        for jogador, mortos in ausencias.items():
            if jogador != jogador_atual:
                valores_mortos |= mortos

        avaliacoes = []

        for jogada in jogadas_de_peca:
            peca, lado = jogada
            peca = tuple(peca)

            mao_restante = list(mao)
            mao_restante.remove(peca)

            # Duplos são inflexíveis (só conectam a um valor): descartá-los
            # cedo tem bônus próprio, sem contar o peso em dobro.
            eh_duplo = (peca[0] == peca[1])
            if eh_duplo:
                peso = 0
                bonus_duplo = peca[0] * self.peso_duplo
            else:
                peso = peca[0] + peca[1]
                bonus_duplo = 0

            diversidade = 0
            cobertura_ponta = 0
            bloqueio = 0

            if pontas_atuais:
                valor_conectado = pontas_atuais[lado]

                if peca[0] == valor_conectado:
                    nova_ponta = peca[1]
                elif peca[1] == valor_conectado:
                    nova_ponta = peca[0]
                else:
                    nova_ponta = None

                if nova_ponta is not None:
                    pontas_apos = list(pontas_atuais)
                    pontas_apos[lado] = nova_ponta

                    # Intencional: uma peça que casa com a nova_ponta conta
                    # na diversidade E na cobertura — a ponta recém-criada
                    # vale mais porque é a que a jogada acabou de definir.
                    diversidade = sum(
                        1 for p in mao_restante
                        if (p[0] in pontas_apos or p[1] in pontas_apos)
                    ) * self.peso_diversidade
                    cobertura_ponta = sum(
                        1 for p in mao_restante if nova_ponta in p
                    ) * self.peso_cobertura

                    # Bloqueio: cada ponta resultante num naipe morto do
                    # oponente é uma ponta que ele (provavelmente) não cobre.
                    bloqueio = sum(
                        1 for v in pontas_apos if v in valores_mortos
                    ) * self.peso_bloqueio
            else:
                numeros_restantes = set()
                for p in mao_restante:
                    numeros_restantes.add(p[0])
                    numeros_restantes.add(p[1])
                diversidade = len(numeros_restantes) * 2

            # Urgência amplifica o descarte (minimiza pips num eventual jogo
            # travado) E o bloqueio (reduz a mobilidade de quem vai bater).
            utilidade_total = (
                (peso + bonus_duplo + bloqueio) * fator_urgencia
                + diversidade
                + cobertura_ponta
            )
            avaliacoes.append((utilidade_total, jogada))

        maior_utilidade = max(u for u, _ in avaliacoes)

        if self.ruido_desempate > 0:
            quase_empatadas = [
                j for u, j in avaliacoes
                if u >= maior_utilidade - self.ruido_desempate
            ]
            return random.choice(quase_empatadas)

        for utilidade, jogada in avaliacoes:
            if utilidade == maior_utilidade:
                return jogada
