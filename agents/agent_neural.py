import random
import numpy as np

from agents.agent import Agente
from agents.codificador import CodificadorDomino
from agents.nn import RedeNeuralSupervisionada, _GPU_ATIVA

# Device do array: cupy (GPU) ou numpy (CPU). Resolvido uma única vez.
if _GPU_ATIVA:
    import cupy as xp
else:
    xp = np

class AgenteNeuralNumPy(Agente):
    """
    Agente neural: carrega um modelo treinado e joga via política
    epsilon-greedy com action masking.

    A tradução estado<->vetor e a máscara de ações vivem no CodificadorDomino
    (fonte única da verdade); este agente só orquestra forward + exploração.
    Compatível com GerenciadorPartida — pode enfrentar o AgenteEstrategico.
    """

    def __init__(self, rede, epsilon=0.0):
        self.rede = rede
        self.epsilon = epsilon
        self.codificador = CodificadorDomino()

    # ------------------------------------------------------------------
    # Fábrica: constrói o agente a partir dos pesos salvos em disco.
    # ------------------------------------------------------------------
    @classmethod
    def carregar(cls, caminho_pesos="pesos_domino_sl.npz", epsilon=0.0):
        """
        Carrega pesos de um .npz e devolve um agente pronto para jogar.

        A arquitetura é derivada das PRÓPRIAS matrizes salvas, então um modelo
        treinado com largura diferente carrega sem editar este arquivo. As
        dimensões são validadas contra o codificador para falhar cedo e alto.

        Uso:
            agente = AgenteNeuralNumPy.carregar("pesos_domino_sl.npz")
        """
        dados = np.load(caminho_pesos)

        # W1:(oculto1, entrada)  W2:(oculto2, oculto1)  W3:(saida, oculto2)
        oculto1, entrada = dados["W1"].shape
        oculto2, _ = dados["W2"].shape
        saida, _ = dados["W3"].shape

        cod = CodificadorDomino()
        if entrada != cod.TAMANHO_VETOR:
            raise ValueError(
                f"Pesos esperam entrada={entrada}, mas o codificador produz "
                f"{cod.TAMANHO_VETOR}: modelo e codificação dessincronizados."
            )
        if saida != len(cod.todas_acoes):
            raise ValueError(
                f"Pesos têm saída={saida}, mas o espaço de ações tem "
                f"{len(cod.todas_acoes)}."
            )

        rede = RedeNeuralSupervisionada(
            tamanho_entrada=entrada,
            tamanho_oculto1=oculto1,
            tamanho_oculto2=oculto2,
            tamanho_saida=saida,
        )

        # Move cada peso para o device das operações internas (np.dot no forward).
        for nome in ("W1", "b1", "W2", "b2", "W3", "b3"):
            setattr(rede, nome, xp.array(dados[nome]))

        return cls(rede, epsilon=epsilon)

    # ------------------------------------------------------------------
    # Interface Agente — chamada pelo GerenciadorPartida a cada turno.
    # ------------------------------------------------------------------
    def escolher_jogada(self, estado, jogadas_legais):
        """
        Política epsilon-greedy com action masking.

        :param estado:         Dicionário de MotorDomino._obter_estado().
        :param jogadas_legais: Ações válidas de acoes_validas().
        :return:               Ação no formato de todas_acoes do codificador.
        """
        print(f"Mão Neural: {estado['mao_jogador']}")
        print(f"Jogadas Possíveis (Neural): {jogadas_legais}")

        if not jogadas_legais:
            return None  # sem ação possível; ajustar se o motor garantir [None]

        # Exploração. Em deploy puro use epsilon=0.0 (sempre exploração da rede).
        if self.epsilon > 0.0 and np.random.rand() < self.epsilon:
            return random.choice(jogadas_legais)
    

        # estado -> vetor (TAMANHO_VETOR,1); leva ao device dos pesos quando em GPU.
        X = self.codificador.encode_estado(estado)
        if _GPU_ATIVA:
            X = xp.array(X)

        probabilidades = self.rede.forward(X)        # (58,1), softmax
        if hasattr(probabilidades, "get"):
            probabilidades = probabilidades.get()    # cupy -> numpy

        # Máscara + argmax sobre as ações legais: delegado ao codificador.
        return self.codificador.decode_saida(probabilidades, jogadas_legais)