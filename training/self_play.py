import random
import time
from collections import deque

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

# ============================================================================
# OPONENTE_TREINO define contra quem o AgenteRL joga em cada partida de
# treino (nunca afeta `avaliar_contra_heuristico`, que é sempre uma medida
# externa e fixa de progresso, chamada só nos checkpoints):
#
#   "self_play"   self-play puro contra um pool de snapshots congelados de
#                 iterações passadas de si mesma (padrão) — ver a seção
#                 "Self-play e o paralelo com o AlphaGo" em
#                 references/fundamentos_rl.pdf.
#   "heuristico"  toda partida de treino é contra o AgenteEstrategico fixo
#                 (nenhum self-play) — útil para gerar um checkpoint de
#                 comparação controlada; ver diagnostico/avalia_self-play/.
# ============================================================================
OPONENTE_TREINO = "self_play"


def _jogar_partida(agentes):
    motor = MotorDomino(num_jogadores=len(agentes))
    gerenciador = GerenciadorPartida(motor, agentes)
    info, _ = gerenciador.jogar_partida_completa()
    return info["vencedor"]


def _recompensas_do_vencedor(vencedor, num_jogadores=2):
    if vencedor == -1:
        return [0.0] * num_jogadores
    return [1.0 if i == vencedor else -1.0 for i in range(num_jogadores)]


def _coletar_passos_self_play(rede, pool):
    """
    Uma partida de self-play puro: a política em treino (`rede`) contra um
    oponente sorteado uniformemente do pool de snapshots congelados de
    iterações passadas (`RedeNeuralPolitica.clonar`) — nunca o
    AgenteEstrategico, que fica reservado à avaliação externa (ver
    `avaliar_contra_heuristico`).

    Só a posição controlada por `rede` gera passos de treino: o oponente é
    congelado (não recebe gradiente desta partida), exatamente como no
    self-play do AlphaGo — "jogamos partidas entre a rede de política atual
    e uma iteração anterior selecionada aleatoriamente... randomizar a
    partir de um pool de oponentes estabiliza o treinamento evitando
    overfitting à política atual" (Silver et al., 2016). Ambos os lados
    amostram estocasticamente (`modo="treino"`), igual ao artigo.

    O pool vive só em memória durante esta chamada a `treinar` (nenhum
    snapshot é gravado em disco): salvar um `.npz` por atualização do pool
    cresce sem limite ao longo de um treino longo, então a diversidade de
    oponentes fica restrita à execução atual em troca de um custo de disco
    previsível (só `caminho_pesos_rl`, sempre um único arquivo).
    """
    posicao_atual = random.randint(0, 1)
    rede_oponente = random.choice(pool) if pool else rede

    agente_atual = AgenteRL(rede, modo="treino")
    agente_oponente = AgenteRL(rede_oponente, modo="treino")
    agentes = [None, None]
    agentes[posicao_atual] = agente_atual
    agentes[1 - posicao_atual] = agente_oponente

    vencedor = _jogar_partida(agentes)
    recompensa = _recompensas_do_vencedor(vencedor)[posicao_atual]
    return agente_atual.finalizar_episodio(recompensa), vencedor, posicao_atual


def _coletar_passos_vs_heuristico(rede):
    """
    Uma partida com o AgenteRL (posição sorteada) contra o AgenteEstrategico
    fixo — usada quando OPONENTE_TREINO="heuristico". Ao contrário do modo
    self-play, aqui NÃO há pool: toda partida de treino é contra o mesmo
    heurístico.
    """
    posicao_rl = random.randint(0, 1)
    agente_rl = AgenteRL(rede, modo="treino")
    agentes = [None, None]
    agentes[posicao_rl] = agente_rl
    agentes[1 - posicao_rl] = AgenteEstrategico()

    vencedor = _jogar_partida(agentes)
    recompensa_rl = _recompensas_do_vencedor(vencedor)[posicao_rl]
    return agente_rl.finalizar_episodio(recompensa_rl), vencedor, posicao_rl


def avaliar_contra_heuristico(rede, num_partidas=200):
    """
    Win-rate do AgenteRL (modo avaliação, greedy) contra o AgenteEstrategico.

    Independente de OPONENTE_TREINO, esta função é sempre uma referência
    externa e fixa de progresso — o mesmo papel que o Pachi cumpriu nas
    avaliações do AlphaGo (um oponente usado só para medir, nunca para
    treinar).
    """
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
    iteracoes=1000,
    partidas_por_iteracao=40,
    oponente_treino=OPONENTE_TREINO,
    taxa_aprendizado=0.001,
    entropia_coef=0.01,
    intervalo_log=10,
    intervalo_checkpoint=50,
    intervalo_pool=10,
    tamanho_pool_max=50,
    partidas_avaliacao=200,
    caminho_pesos_sl=PESOS_SL,
    caminho_pesos_rl=PESOS_RL,
):
    """
    Loop principal de treinamento por REINFORCE + ator-crítico.

    `oponente_treino` escolhe o adversário de cada partida de treino:

    - "self_play": a política em treino enfrenta um oponente sorteado de um
      pool de snapshots congelados de iterações passadas de si mesma,
      mantido só em memória (`_coletar_passos_self_play`). A cada
      `intervalo_pool` iterações, um snapshot congelado é adicionado ao pool
      (mantendo só as `tamanho_pool_max` entradas mais recentes) — nenhum
      snapshot é gravado em disco, então o pool não sobrevive ao fim do
      processo (uma retomada recomeça o pool só com a rede carregada).
    - "heuristico": toda partida de treino é contra o `AgenteEstrategico`
      fixo (`_coletar_passos_vs_heuristico`); não há pool.

    Em qualquer um dos dois modos, agrega os passos (estado, ação, retorno)
    das posições jogadas pela rede em treino, usa a cabeça de valor como
    baseline (ator-crítico) e aplica um passo de gradiente de política
    (REINFORCE + entropia) por lote.

    Nota sobre `partidas_por_iteracao`: cada partida só contribui passos de
    UMA posição (a outra é o oponente — congelado no modo self-play, ou o
    heurístico no modo heuristico —, nenhum dos dois recebe gradiente), por
    isso o padrão é 40 (o dobro do antigo currículo misto, que também
    contava as duas posições em parte das partidas).

    Os pesos da política só são gravados em disco em `caminho_pesos_rl`
    (sempre o mesmo arquivo, sobrescrito): a cada `intervalo_checkpoint`
    iterações e ao final do treinamento. Se já existir um checkpoint em
    `caminho_pesos_rl`, o treinamento retoma a partir dele; caso contrário,
    faz o warm-start a partir do SL.
    """
    if oponente_treino not in ("self_play", "heuristico"):
        raise ValueError(
            f"oponente_treino deve ser 'self_play' ou 'heuristico', recebido {oponente_treino!r}"
        )

    try:
        rede = RedeNeuralPolitica.carregar(caminho_pesos_rl, taxa_aprendizado=taxa_aprendizado)
        print(f"Retomando treinamento RL a partir de {caminho_pesos_rl}")
    except FileNotFoundError:
        rede = RedeNeuralPolitica.carregar_de_sl(caminho_pesos_sl, taxa_aprendizado=taxa_aprendizado)
        print(f"Inicializando a política RL a partir do SL ({caminho_pesos_sl})")

    pool = None
    if oponente_treino == "self_play":
        pool = deque(maxlen=tamanho_pool_max)
        pool.append(rede.clonar())

    inicio = time.time()
    for iteracao in range(1, iteracoes + 1):
        lote = []
        vitorias = 0

        for _ in range(partidas_por_iteracao):
            if oponente_treino == "self_play":
                passos, vencedor, posicao_atual = _coletar_passos_self_play(rede, pool)
            else:
                passos, vencedor, posicao_atual = _coletar_passos_vs_heuristico(rede)
            lote.extend(passos)
            if vencedor == posicao_atual:
                vitorias += 1

        if not lote:
            continue

        X_lote = xp.hstack([x for x, _, _ in lote])
        acoes_idx = [idx for _, idx, _ in lote]
        retornos = xp.array([r for _, _, r in lote], dtype=float).reshape(1, -1)

        # Baseline dependente do estado (crítico) em vez da média do lote:
        # reduz a variância do gradiente de política (ator-crítico simples).
        valores = rede.prever_valores(X_lote)
        vantagens = retornos - valores

        metricas = rede.backward_policy_gradient(
            acoes_idx, vantagens, retornos=retornos, entropia_coef=entropia_coef
        )
        media_vantagem = float(xp.mean(vantagens))
        entropia_media = metricas["entropia"]

        if oponente_treino == "self_play" and iteracao % intervalo_pool == 0:
            pool.append(rede.clonar())

        if iteracao % intervalo_log == 0:
            rotulo_vit = "vs pool" if oponente_treino == "self_play" else "vs heurístico (treino)"
            sufixo_pool = f" | pool: {len(pool)}" if oponente_treino == "self_play" else ""
            print(
                f"Iteração {iteracao} | passos: {len(lote)} | "
                f"vantagem média: {media_vantagem:.3f} | entropia: {entropia_media:.3f} | "
                f"perda crítico: {metricas['perda_valor']:.3f} | "
                f"vitórias {rotulo_vit}: {vitorias}/{partidas_por_iteracao}{sufixo_pool}"
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
