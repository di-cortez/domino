import os

from agents.nn import RedeNeuralSupervisionada, _GPU_ATIVA

if _GPU_ATIVA:
    import cupy as xp
else:
    import numpy as xp

import numpy as np


class RedeNeuralPolitica(RedeNeuralSupervisionada):
    """
    Mesma arquitetura 79 -> 256 -> 128 -> 58 da RedeNeuralSupervisionada
    (forward herdado sem alteração), mas atualizada por REINFORCE com
    baseline em vez de cross-entropy supervisionada.

    Por ter exatamente a mesma forma de pesos, qualquer checkpoint salvo pelo
    pipeline de SL (models/pesos_domino_sl.npz) pode ser usado como ponto de
    partida ("warm start") do treinamento por self-play via carregar_de_sl.
    """

    @classmethod
    def _carregar_pesos_npz(cls, caminho, taxa_aprendizado):
        dados = np.load(caminho)

        oculto1, entrada = dados["W1"].shape
        oculto2, _ = dados["W2"].shape
        saida, _ = dados["W3"].shape

        rede = cls(
            tamanho_entrada=entrada,
            tamanho_oculto1=oculto1,
            tamanho_oculto2=oculto2,
            tamanho_saida=saida,
            taxa_aprendizado=taxa_aprendizado,
        )
        for nome in ("W1", "b1", "W2", "b2", "W3", "b3"):
            setattr(rede, nome, xp.array(dados[nome]))
        return rede

    @classmethod
    def carregar_de_sl(cls, caminho_pesos_sl="models/pesos_domino_sl.npz", taxa_aprendizado=0.001):
        """Warm-start: copia os pesos de imitação (SL) como ponto de partida da política RL."""
        return cls._carregar_pesos_npz(caminho_pesos_sl, taxa_aprendizado)

    @classmethod
    def carregar(cls, caminho_pesos_rl, taxa_aprendizado=0.001):
        """Retoma um checkpoint RL salvo por `salvar` (mesmo formato .npz do SL)."""
        return cls._carregar_pesos_npz(caminho_pesos_rl, taxa_aprendizado)

    def salvar(self, caminho_pesos):
        def para_numpy(matriz):
            return matriz.get() if hasattr(matriz, "get") else matriz

        # Garante que a pasta de destino (ex.: models/) exista antes de salvar.
        pasta_pesos = os.path.dirname(caminho_pesos)
        if pasta_pesos:
            os.makedirs(pasta_pesos, exist_ok=True)

        np.savez(
            caminho_pesos,
            W1=para_numpy(self.W1), b1=para_numpy(self.b1),
            W2=para_numpy(self.W2), b2=para_numpy(self.b2),
            W3=para_numpy(self.W3), b3=para_numpy(self.b3),
        )

    def backward_policy_gradient(self, acoes_idx, vantagens, entropia_coef=0.01):
        """
        Atualiza os pesos por gradiente de política (REINFORCE + baseline),
        reaproveitando o cache deixado pela última chamada a `forward` (igual
        ao que `backward` supervisionada já faz).

        :param acoes_idx:     índices (m,) das ações amostradas em cada passo.
        :param vantagens:     vantagens (1, m) = retorno - baseline do lote.
        :param entropia_coef: peso do bônus de entropia; evita que a política
                               colapse para determinística antes da hora.
        :return: (vantagem média do lote, entropia média do lote) — só para log.
        """
        A3, A2, A1, X = self.cache["A3"], self.cache["A2"], self.cache["A1"], self.cache["X"]
        m = A3.shape[1]

        acoes_idx = xp.asarray(acoes_idx)
        vantagens = xp.asarray(vantagens).reshape(1, m)

        Y_amostrado = xp.zeros_like(A3)
        Y_amostrado[acoes_idx, xp.arange(m)] = 1.0

        # d(-vantagem * log pi(a)) / dZ3 = (pi - onehot(a)) * vantagem
        dZ3_pg = (A3 - Y_amostrado) * vantagens

        # d(-H(pi)) / dZ3 = pi * (log(pi) + H), H = -sum(pi * log pi) por coluna
        log_A3 = xp.log(A3 + 1e-8)
        entropia = -xp.sum(A3 * log_A3, axis=0, keepdims=True)
        dZ3_entropia = A3 * (log_A3 + entropia)

        dZ3 = dZ3_pg + entropia_coef * dZ3_entropia

        dW3 = (1. / m) * xp.dot(dZ3, A2.T)
        db3 = (1. / m) * xp.sum(dZ3, axis=1, keepdims=True)

        dA2 = xp.dot(self.W3.T, dZ3)
        dZ2 = dA2 * self.derivada_relu(self.cache["Z2"])
        dW2 = (1. / m) * xp.dot(dZ2, A1.T)
        db2 = (1. / m) * xp.sum(dZ2, axis=1, keepdims=True)

        dA1 = xp.dot(self.W2.T, dZ2)
        dZ1 = dA1 * self.derivada_relu(self.cache["Z1"])
        dW1 = (1. / m) * xp.dot(dZ1, X.T)
        db1 = (1. / m) * xp.sum(dZ1, axis=1, keepdims=True)

        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        return float(xp.mean(vantagens)), float(xp.mean(entropia))
