"""
Diagnóstico do treinamento: avalia um agente contra um oponente em N partidas
e gera métricas (taxa de vitória/empate/derrota, IC 95%, desempenho por
posição, duração) + gráficos.

Para rodar (a partir da raiz do repositório, com o ambiente virtual ativo):

    python3 -m diagnostico.avaliar

A escolha dos jogadores e o número de partidas ficam na seção CONFIGURAÇÃO
logo abaixo — basta editar as variáveis e rodar de novo. Quem preferir o
terminal pode sobrescrever qualquer valor com opções de linha de comando
(python3 -m diagnostico.avaliar --help).

As métricas e os gráficos ficam em diagnostico/gera_graficos.py.
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

# ============================================================================
# AGENTE é o jogador avaliado; OPONENTE é o adversário. As opções são:
#
#   "rl"          rede treinada por reforço (self-play)
#   "sl"          rede treinada por imitação (supervisionado)
#   "heuristico"  agente de regras
#   "guloso"      joga sempre a peça de maior soma de pontos
#   "aleatorio"   joga uma peça válida ao acaso
#
# Exemplo: para medir o modelo de reforço contra o jogador aleatório,
# troque OPONENTE = "heuristico" por OPONENTE = "aleatorio".
# ============================================================================
AGENTE = "rl"
OPONENTE = "heuristico"
NUM_PARTIDAS = 10000

SEED = None            # um inteiro (ex.: 42) repete exatamente o mesmo sorteio
PESOS_AGENTE = None    # caminho de um .npz específico (None = pesos padrão)
PESOS_OPONENTE = None  # idem, para o oponente
PASTA_SAIDA = None     # None = diagnostico/resultados/<agente>_vs_<oponente>
GERAR_GRAFICOS = True  # False = só o resumo no console + CSV/JSON
# ============================================================================


# Permite rodar tanto como módulo (-m diagnostico.avaliar) quanto script direto.
RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import numpy as np

from middleware.motor_domino import MotorDomino
from middleware.middleware import GerenciadorPartida
from diagnostico.gera_graficos import ROTULO, resumir, gerar_graficos

AGENTES_DISPONIVEIS = ("rl", "sl", "heuristico", "guloso", "aleatorio")

PESOS_PADRAO = {
    "rl": RAIZ / "models" / "pesos_domino_rl.npz",
    "sl": RAIZ / "models" / "pesos_domino_sl.npz",
}


def criar_agente(nome, caminho_pesos=None):
    """Instancia um agente pelo nome. Imports locais para que agentes que não
    dependem de pesos/rede funcionem mesmo sem os artefatos de modelo."""
    if nome == "rl":
        from agents.agent_rl import AgenteRL
        return AgenteRL.carregar(str(caminho_pesos or PESOS_PADRAO["rl"]), modo="avaliacao")
    if nome == "sl":
        from agents.agent_neural import AgenteNeuralNumPy
        return AgenteNeuralNumPy.carregar(str(caminho_pesos or PESOS_PADRAO["sl"]))
    if nome == "heuristico":
        from agents.heuristic_agent import AgenteEstrategico
        return AgenteEstrategico()
    if nome == "guloso":
        from agents.agent import AgenteGuloso
        return AgenteGuloso()
    if nome == "aleatorio":
        from agents.agent import AgenteAleatorio
        return AgenteAleatorio()
    raise ValueError(f"Agente desconhecido: {nome!r}. Opções: {AGENTES_DISPONIVEIS}")


def jogar_partida(agente, oponente, posicao_agente):
    """Joga uma partida com o agente avaliado na posição indicada.

    Retorna um dicionário com o resultado do ponto de vista do agente avaliado.
    """
    agentes = [None, None]
    agentes[posicao_agente] = agente
    agentes[1 - posicao_agente] = oponente

    motor = MotorDomino(num_jogadores=2)
    gerenciador = GerenciadorPartida(motor, agentes)
    info, _ = gerenciador.jogar_partida_completa()

    vencedor = info["vencedor"]
    if vencedor == -1:
        resultado = "empate"
    elif vencedor == posicao_agente:
        resultado = "vitoria"
    else:
        resultado = "derrota"

    final = motor.to_dict()
    pips = [sum(p[0] + p[1] for p in mao) for mao in final["maos"]]
    return {
        "posicao_agente": posicao_agente,
        "resultado": resultado,
        "turnos": final["turno"],
        "pips_agente": pips[posicao_agente],
        "pips_oponente": pips[1 - posicao_agente],
    }


def avaliar(nome_agente, nome_oponente, num_partidas, pesos=None, pesos_oponente=None,
            seed=None, callback_progresso=None):
    """Roda `num_partidas` alternando a posição inicial do agente avaliado."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    agente = criar_agente(nome_agente, pesos)
    oponente = criar_agente(nome_oponente, pesos_oponente)

    partidas = []
    for i in range(num_partidas):
        registro = jogar_partida(agente, oponente, posicao_agente=i % 2)
        registro["partida"] = i + 1
        partidas.append(registro)
        if callback_progresso:
            callback_progresso(i + 1, num_partidas)
    return partidas


# ---------------------------------------------------------------------------
# Persistência e relatório de console
# ---------------------------------------------------------------------------
def salvar_csv(partidas, caminho):
    campos = ["partida", "posicao_agente", "resultado", "turnos", "pips_agente", "pips_oponente"]
    with open(caminho, "w", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows({c: p[c] for c in campos} for p in partidas)


def imprimir_resumo(resumo, duracao_s):
    n = resumo["num_partidas"]
    print(f"\n===== Diagnóstico: {resumo['agente']} vs {resumo['oponente']} =====")
    print(f"Partidas: {n} | tempo: {duracao_s:.1f}s ({n / duracao_s:.1f} partidas/s)")
    for chave in ("vitoria", "empate", "derrota"):
        taxa = resumo["taxas"][chave]
        print(f"  {ROTULO[chave]:<8} {resumo['contagem'][chave]:>5}  ({taxa:6.1%})")
    lo, hi = resumo["ic95_vitoria"]
    print(f"  IC 95% (vitória): [{lo:.1%}, {hi:.1%}]")
    for pos in ("0", "1"):
        info = resumo["por_posicao"][pos]
        print(f"  Como jogador {pos}: {info['taxa_vitoria']:.1%} de vitórias em {info['partidas']} partidas")
    print(f"  Turnos por partida: {resumo['turnos_media']:.1f} ± {resumo['turnos_desvio']:.1f}")
    print(f"  Pips restantes (média): agente {resumo['pips_restantes_media_agente']:.1f} | "
          f"oponente {resumo['pips_restantes_media_oponente']:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Avalia um agente de dominó contra um oponente em N partidas. "
                    "Os padrões vêm da seção CONFIGURAÇÃO no topo do arquivo; "
                    "as opções abaixo sobrescrevem esses valores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--agente", choices=AGENTES_DISPONIVEIS, default=AGENTE,
                        help="agente avaliado")
    parser.add_argument("--oponente", choices=AGENTES_DISPONIVEIS, default=OPONENTE,
                        help="oponente")
    parser.add_argument("-n", "--partidas", type=int, default=NUM_PARTIDAS,
                        help="número de partidas")
    parser.add_argument("--pesos", type=Path, default=PESOS_AGENTE,
                        help="pesos .npz do agente (para rl/sl)")
    parser.add_argument("--pesos-oponente", type=Path, default=PESOS_OPONENTE,
                        help="pesos .npz do oponente (para rl/sl)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="semente para reprodutibilidade")
    parser.add_argument("--saida", type=Path, default=PASTA_SAIDA,
                        help="pasta de saída (padrão: diagnostico/resultados/<agente>_vs_<oponente>)")
    parser.add_argument("--sem-graficos", action="store_true", default=not GERAR_GRAFICOS,
                        help="não gera os PNGs, apenas CSV/JSON e resumo no console")
    args = parser.parse_args()

    pasta = args.saida or RAIZ / "diagnostico" / "resultados" / f"{args.agente}_vs_{args.oponente}"
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)

    passo = max(1, args.partidas // 10)

    def progresso(i, total):
        if i % passo == 0 or i == total:
            print(f"  {i}/{total} partidas...", flush=True)

    print(f"Avaliando {args.agente} vs {args.oponente} em {args.partidas} partidas "
          f"(posição inicial alternada a cada partida)")
    inicio = time.time()
    partidas = avaliar(args.agente, args.oponente, args.partidas,
                       pesos=args.pesos, pesos_oponente=args.pesos_oponente,
                       seed=args.seed, callback_progresso=progresso)
    duracao = time.time() - inicio

    resumo = resumir(partidas, args.agente, args.oponente, args.seed)
    imprimir_resumo(resumo, duracao)

    salvar_csv(partidas, pasta / "partidas.csv")
    with open(pasta / "resumo.json", "w") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)

    if not args.sem_graficos:
        gerar_graficos(partidas, resumo, pasta)

    print(f"\nResultados salvos em {pasta}/")
    if not args.sem_graficos:
        print("  taxas_acumuladas.png, distribuicao_resultados.png, "
              "vitorias_por_posicao.png, duracao_partidas.png")
    print("  partidas.csv, resumo.json")


if __name__ == "__main__":
    main()
