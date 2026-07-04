import random
import time

import numpy as np

from middleware.motor_domino import MotorDomino
from middleware.middleware import GerenciadorPartida
from agents.heuristic_agent import AgenteEstrategico
from agents.agent_rl import AgenteRL
from agents.rl_nn import RedeNeuralPolitica
from agents.nn import _GPU_ATIVA

if _GPU_ATIVA:
    import cupy as xp
else:
    xp = np

PESOS_SL = "models/pesos_domino_sl.npz"
PESOS_RL = "models/pesos_domino_rl.npz"


def _jogar_partida(agentes):
    motor = MotorDomino(num_jogadores=len(agentes))
    gerenciador = GerenciadorPartida(motor, agentes)
    info, _ = gerenciador.jogar_partida_completa()
    return info["vencedor"]


def _recompensas_do_vencedor(vencedor, num_jogadores=2):
    if vencedor == -1:
        return [0.0] * num_jogadores
    return [1.0 if i == vencedor else -1.0 for i in range(num_jogadores)]


def _coletar_passos_self_play(rede):
    """Uma partida com as duas posições controladas pela mesma política em treino."""
    agentes = [AgenteRL(rede, modo="treino"), AgenteRL(rede, modo="treino")]
    vencedor = _jogar_partida(agentes)
    recompensas = _recompensas_do_vencedor(vencedor)

    passos = []
    for agente, recompensa in zip(agentes, recompensas):
        passos.extend(agente.finalizar_episodio(recompensa))
    return passos, vencedor


def _coletar_passos_vs_heuristico(rede):
    """Uma partida com o AgenteRL (posição sorteada) contra o AgenteEstrategico fixo."""
    posicao_rl = random.randint(0, 1)
    agente_rl = AgenteRL(rede, modo="treino")
    agentes = [None, None]
    agentes[posicao_rl] = agente_rl
    agentes[1 - posicao_rl] = AgenteEstrategico()

    vencedor = _jogar_partida(agentes)
    recompensa_rl = _recompensas_do_vencedor(vencedor)[posicao_rl]
    return agente_rl.finalizar_episodio(recompensa_rl), vencedor, posicao_rl


def avaliar_contra_heuristico(rede, num_partidas=200):
    """Win-rate do AgenteRL (modo avaliação, greedy) contra o AgenteEstrategico."""
    vitorias = 0
    empates = 0
    for i in range(num_partidas):
        posicao_rl = i % 2
        agente_rl = AgenteRL(rede, modo="avaliacao")
        agentes = [None, None]
        agentes[posicao_rl] = agente_rl
        agentes[1 - posicao_rl] = AgenteEstrategico()

        vencedor = _jogar_partida(agentes)
        if vencedor == posicao_rl:
            vitorias += 1
        elif vencedor == -1:
            empates += 1

    return vitorias / num_partidas, empates / num_partidas


def treinar(
    iteracoes=500,
    partidas_por_iteracao=20,
    proporcao_self_play=0.8,
    taxa_aprendizado=0.001,
    entropia_coef=0.01,
    intervalo_log=10,
    intervalo_checkpoint=50,
    partidas_avaliacao=200,
    caminho_pesos_sl=PESOS_SL,
    caminho_pesos_rl=PESOS_RL,
):
    """
    Loop principal de treinamento por self-play (currículo misto).

    A cada iteração: joga um lote de partidas — self-play e contra o
    AgenteEstrategico na proporção configurada —, agrega todos os passos
    (estado, ação, retorno) produzidos pelas posições do AgenteRL, calcula a
    vantagem com baseline de média do lote e aplica um único passo de
    gradiente de política (REINFORCE + entropia).

    Se já existir um checkpoint em `caminho_pesos_rl`, o treinamento retoma a
    partir dele; caso contrário, faz o warm-start a partir do SL.
    """
    try:
        rede = RedeNeuralPolitica.carregar(caminho_pesos_rl, taxa_aprendizado=taxa_aprendizado)
        print(f"Retomando treinamento RL a partir de {caminho_pesos_rl}")
    except FileNotFoundError:
        rede = RedeNeuralPolitica.carregar_de_sl(caminho_pesos_sl, taxa_aprendizado=taxa_aprendizado)
        print(f"Inicializando a política RL a partir do SL ({caminho_pesos_sl})")

    n_self = round(partidas_por_iteracao * proporcao_self_play)
    n_heur = partidas_por_iteracao - n_self

    inicio = time.time()
    for iteracao in range(1, iteracoes + 1):
        lote = []
        vitorias_vs_heur = 0

        for _ in range(n_self):
            passos, _ = _coletar_passos_self_play(rede)
            lote.extend(passos)

        for _ in range(n_heur):
            passos, vencedor, posicao_rl = _coletar_passos_vs_heuristico(rede)
            lote.extend(passos)
            if vencedor == posicao_rl:
                vitorias_vs_heur += 1

        if not lote:
            continue

        X_lote = xp.hstack([x for x, _, _ in lote])
        acoes_idx = [idx for _, idx, _ in lote]
        retornos = xp.array([r for _, _, r in lote], dtype=float)
        baseline = retornos.mean()
        vantagens = (retornos - baseline).reshape(1, -1)

        rede.forward(X_lote)
        media_vantagem, entropia_media = rede.backward_policy_gradient(
            acoes_idx, vantagens, entropia_coef=entropia_coef
        )

        if iteracao % intervalo_log == 0:
            sufixo_heur = f" | vitórias vs heurístico (treino): {vitorias_vs_heur}/{n_heur}" if n_heur else ""
            print(
                f"Iteração {iteracao} | passos: {len(lote)} | "
                f"vantagem média: {media_vantagem:.3f} | entropia: {entropia_media:.3f}{sufixo_heur}"
            )

        if iteracao % intervalo_checkpoint == 0:
            rede.salvar(caminho_pesos_rl)
            taxa_vitoria, taxa_empate = avaliar_contra_heuristico(rede, num_partidas=partidas_avaliacao)
            print(
                f"  [checkpoint] {caminho_pesos_rl} salvo | "
                f"avaliação greedy vs heurístico: {taxa_vitoria:.1%} vitórias, "
                f"{taxa_empate:.1%} empates ({partidas_avaliacao} partidas)"
            )

    rede.salvar(caminho_pesos_rl)
    tempo_total = time.time() - inicio
    print(f"\nTreinamento concluído em {tempo_total:.1f}s. Pesos finais em {caminho_pesos_rl}")


if __name__ == "__main__":
    treinar()
