from agents.agent import Agente
from agents.codificador import CodificadorDomino
from agents.rl_nn import RedeNeuralPolitica
from agents.nn import _GPU_ATIVA

if _GPU_ATIVA:
    import cupy as xp
else:
    import numpy as xp


class AgenteRL(Agente):
    """
    Agente cuja política é refinada por self-play (REINFORCE + baseline).

    Mantém o mesmo protocolo `Agente` usado pelo heurístico e pelo neural
    (SL) — escolher_jogada(estado, jogadas_legais) -> jogada — então o
    GerenciadorPartida não precisa de nenhuma alteração para jogá-lo, seja
    contra ele mesmo (self-play) ou contra o AgenteEstrategico.

    modo="treino":     amostra estocasticamente (via CodificadorDomino) e
                        registra (estado, ação) na trajetória do episódio
                        atual, para training/self_play.py computar o
                        gradiente de política ao fim da partida.
    modo="avaliacao":  joga greedy (argmax), igual ao AgenteNeuralNumPy —
                        usado nos benchmarks e como agente final na UI.
    """

    def __init__(self, rede, modo="treino"):
        self.rede = rede
        self.modo = modo
        self.codificador = CodificadorDomino()
        self.trajetoria = []  # [(X, idx_acao_amostrada), ...] do episódio atual

    @classmethod
    def carregar(cls, caminho_pesos="models/pesos_domino_rl.npz", modo="avaliacao"):
        rede = RedeNeuralPolitica.carregar(caminho_pesos)
        return cls(rede, modo=modo)

    def escolher_jogada(self, estado, jogadas_legais):
        if not jogadas_legais:
            return None

        X = self.codificador.encode_estado(estado)
        if _GPU_ATIVA:
            X = xp.array(X)

        probabilidades = self.rede.forward(X)
        if hasattr(probabilidades, "get"):
            probabilidades = probabilidades.get()  # cupy -> numpy, para a amostragem/argmax

        if self.modo == "treino":
            jogada, idx_amostrado = self.codificador.amostrar_acao(probabilidades, jogadas_legais)
            self.trajetoria.append((X, idx_amostrado))
            return jogada

        return self.codificador.decode_saida(probabilidades, jogadas_legais)

    def finalizar_episodio(self, recompensa_final):
        """
        Encerra o episódio: dominó só dá recompensa no final (esparsa), então
        propagamos `recompensa_final` (+1 vitória / -1 derrota / 0 empate)
        para todas as jogadas que ESTE agente fez na partida — retorno de
        Monte Carlo com gamma=1, episódios curtos o suficiente para não
        precisar de desconto.

        Retorna [(X, idx_acao, retorno), ...] para o lote de treinamento e
        zera a trajetória interna para a próxima partida.
        """
        passos = [(X, idx, recompensa_final) for X, idx in self.trajetoria]
        self.trajetoria = []
        return passos
