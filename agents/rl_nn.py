import os

from agents.nn import RedeNeuralSupervisionada, _GPU_ATIVA

if _GPU_ATIVA:
    import cupy as xp
else:
    import numpy as xp

import numpy as np

_PESOS_POLITICA = ("W1", "b1", "W2", "b2", "W3", "b3")
_PESOS_VALOR = ("Wv", "bv")


class RedeNeuralPolitica(RedeNeuralSupervisionada):
    """
    Mesma arquitetura de entrada->256->128->58 da RedeNeuralSupervisionada
    (forward herdado sem alteração), mas atualizada por REINFORCE com
    baseline em vez de cross-entropy supervisionada.

    Além da cabeça de política (softmax de 58 ações), possui uma cabeça de
    VALOR linear (Wv, bv) sobre a mesma segunda camada oculta: V(s) = Wv·A2 + bv.
    Ela é treinada por regressão contra os retornos e serve como baseline
    dependente do estado (ator-crítico simples), reduzindo a variância do
    gradiente em relação ao baseline de média do lote.

    Por ter exatamente a mesma forma dos pesos de política, qualquer
    checkpoint salvo pelo pipeline de SL (models/pesos_domino_sl.npz) pode
    ser usado como ponto de partida ("warm start") via carregar_de_sl;
    checkpoints antigos sem a cabeça de valor também carregam (Wv/bv são
    inicializados do zero nesse caso).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        oculto2 = self.W3.shape[1]
        # Cabeça de valor inicializada em zero: V(s)=0 no início, então o
        # crítico recém-criado não injeta gradiente ruidoso no tronco vindo
        # de ativações ReLU grandes de um warm-start já treinado.
        self.Wv = xp.zeros((1, oculto2))
        self.bv = xp.zeros((1, 1))

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
        for nome in _PESOS_POLITICA:
            setattr(rede, nome, xp.array(dados[nome]))

        # Checkpoints de SL (e RL antigos) não têm a cabeça de valor; nesse
        # caso mantemos a inicialização aleatória feita no __init__.
        if all(nome in dados for nome in _PESOS_VALOR):
            for nome in _PESOS_VALOR:
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
            **{nome: para_numpy(getattr(self, nome)) for nome in _PESOS_POLITICA + _PESOS_VALOR},
        )

    def clonar(self):
        """Cópia congelada (snapshot) da rede, para o pool de oponentes do self-play."""
        clone = RedeNeuralPolitica(
            tamanho_entrada=self.W1.shape[1],
            tamanho_oculto1=self.W1.shape[0],
            tamanho_oculto2=self.W2.shape[0],
            tamanho_saida=self.W3.shape[0],
            taxa_aprendizado=self.lr,
        )
        for nome in _PESOS_POLITICA + _PESOS_VALOR:
            setattr(clone, nome, getattr(self, nome).copy())
        return clone

    def prever_valores(self, X):
        """
        V(s) para cada coluna de X: executa o forward (preenchendo o cache)
        e aplica a cabeça de valor linear sobre A2. Retorna array (1, m).
        """
        self.forward(X)
        return xp.dot(self.Wv, self.cache["A2"]) + self.bv

    def backward_policy_gradient(
        self,
        acoes_idx,
        vantagens,
        retornos=None,
        entropia_coef=0.01,
        valor_coef=0.5,
        clip_grad_norm=5.0,
    ):
        """
        Atualiza os pesos por gradiente de política (REINFORCE + baseline),
        reaproveitando o cache deixado pela última chamada a `forward` (igual
        ao que `backward` supervisionada já faz).

        :param acoes_idx:      índices (m,) das ações amostradas em cada passo.
        :param vantagens:      vantagens (1, m), já normalizadas pelo chamador.
        :param retornos:       retornos (1, m) para treinar a cabeça de valor;
                               None pula a atualização do crítico.
        :param entropia_coef:  peso do bônus de entropia; evita que a política
                               colapse para determinística antes da hora.
        :param valor_coef:     peso da perda do crítico (MSE) no gradiente.
        :param clip_grad_norm: norma global máxima dos gradientes; None desativa.
        :return: dict com métricas do lote (entropia, perda_valor, norma_grad).
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

        # Cabeça de valor: perda 0.5*(V - R)^2, gradiente linear em Zv = Wv·A2 + bv.
        perda_valor = 0.0
        dWv = xp.zeros_like(self.Wv)
        dbv = xp.zeros_like(self.bv)
        if retornos is not None:
            retornos = xp.asarray(retornos).reshape(1, m)
            V = xp.dot(self.Wv, A2) + self.bv
            erro_v = V - retornos
            perda_valor = float(xp.mean(0.5 * erro_v ** 2))

            dZv = valor_coef * erro_v
            dWv = (1. / m) * xp.dot(dZv, A2.T)
            dbv = (1. / m) * xp.sum(dZv, axis=1, keepdims=True)
            dA2 = dA2 + xp.dot(self.Wv.T, dZv)  # crítico compartilha o tronco

        dZ2 = dA2 * self.derivada_relu(self.cache["Z2"])
        dW2 = (1. / m) * xp.dot(dZ2, A1.T)
        db2 = (1. / m) * xp.sum(dZ2, axis=1, keepdims=True)

        dA1 = xp.dot(self.W2.T, dZ2)
        dZ1 = dA1 * self.derivada_relu(self.cache["Z1"])
        dW1 = (1. / m) * xp.dot(dZ1, X.T)
        db1 = (1. / m) * xp.sum(dZ1, axis=1, keepdims=True)

        grads = {
            "W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
            "W3": dW3, "b3": db3, "Wv": dWv, "bv": dbv,
        }

        # Clipping pela norma global: um lote atipicamente desbalanceado não
        # consegue mais desestabilizar a política com um único passo gigante.
        norma_grad = float(xp.sqrt(sum(xp.sum(g ** 2) for g in grads.values())))
        if clip_grad_norm is not None and norma_grad > clip_grad_norm:
            fator = clip_grad_norm / (norma_grad + 1e-8)
            grads = {nome: g * fator for nome, g in grads.items()}

        for nome, grad in grads.items():
            setattr(self, nome, getattr(self, nome) - self.lr * grad)

        return {
            "entropia": float(xp.mean(entropia)),
            "perda_valor": perda_valor,
            "norma_grad": norma_grad,
        }
