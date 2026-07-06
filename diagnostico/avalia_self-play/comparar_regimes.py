"""
Compara dois checkpoints de RL treinados por regimes diferentes de
`training/self_play.py` (parâmetro `oponente_treino`):

  - "self_play"  self-play puro contra o pool de snapshots congelados
  - "heuristico" toda partida de treino direto contra o AgenteEstrategico

Roda três confrontos — cada um contra o AgenteEstrategico (referência
externa fixa) e um confronto direto entre os dois checkpoints — usando a
mesma infraestrutura de `diagnostico/avaliar.py` e `diagnostico/gera_graficos.py`
(resumo.json, partidas.csv, PNGs), e imprime uma conclusão objetiva sobre
significância estatística (sobreposição de IC 95%), pronta para apresentar.

Como gerar os dois checkpoints (a partir da raiz do repositório):

    python -c "from training.self_play import treinar; \\
        treinar(oponente_treino='self_play', caminho_pesos_rl='models/pesos_domino_rl_self_play.npz')"
    python -c "from training.self_play import treinar; \\
        treinar(oponente_treino='heuristico', caminho_pesos_rl='models/pesos_domino_rl_heuristico.npz')"

Como rodar esta comparação (a partir da raiz do repositório):

    python "diagnostico/avalia_self-play/comparar_regimes.py"

(Não dá para usar `python -m ...` aqui porque o nome da pasta tem um hífen,
que não é um identificador Python válido — por isso é um script direto, não
um módulo de pacote.)
"""

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from diagnostico.avaliar import avaliar, salvar_csv  # noqa: E402
from diagnostico.gera_graficos import resumir, gerar_graficos  # noqa: E402

# ============================================================================
# CONFIGURAÇÃO — edite os caminhos/valores e rode de novo, ou sobrescreva
# pela linha de comando (--help lista todas as opções).
# ============================================================================
PESOS_SELF_PLAY = RAIZ / "models" / "pesos_domino_rl_self_play.npz"
PESOS_HEURISTICO = RAIZ / "models" / "pesos_domino_rl_heuristico.npz"
NUM_PARTIDAS = 1000
SEED = 7
PASTA_SAIDA = RAIZ / "diagnostico" / "resultados" / "self_play_vs_heuristico_regimes"
# ============================================================================


def _avaliar_e_salvar(nome_a, nome_b, pesos_a, pesos_b, num_partidas, seed, pasta):
    """Roda `avaliar()`, salva resumo.json/partidas.csv/PNGs e devolve o resumo."""
    pasta.mkdir(parents=True, exist_ok=True)
    partidas = avaliar(nome_a, nome_b, num_partidas, pesos=pesos_a, pesos_oponente=pesos_b, seed=seed)
    resumo = resumir(partidas, nome_a, nome_b, seed)
    salvar_csv(partidas, pasta / "partidas.csv")
    with open(pasta / "resumo.json", "w") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    gerar_graficos(partidas, resumo, pasta)
    return resumo


def _ic_se_sobrepoe(resumo_a, resumo_b):
    """IC 95% de vitória de dois resumos (mesmo oponente) se sobrepõem?"""
    lo_a, hi_a = resumo_a["ic95_vitoria"]
    lo_b, hi_b = resumo_b["ic95_vitoria"]
    return not (hi_a < lo_b or hi_b < lo_a)


def main():
    parser = argparse.ArgumentParser(
        description="Compara um checkpoint de RL treinado por self-play puro (pool) contra "
                    "um treinado direto contra o AgenteEstrategico, entre si e contra o heurístico.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pesos-self-play", type=Path, default=PESOS_SELF_PLAY,
                        help="checkpoint treinado com oponente_treino='self_play'")
    parser.add_argument("--pesos-heuristico", type=Path, default=PESOS_HEURISTICO,
                        help="checkpoint treinado com oponente_treino='heuristico'")
    parser.add_argument("-n", "--partidas", type=int, default=NUM_PARTIDAS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--saida", type=Path, default=PASTA_SAIDA)
    args = parser.parse_args()

    for caminho, rotulo in ((args.pesos_self_play, "self-play+pool"), (args.pesos_heuristico, "heurístico")):
        if not caminho.exists():
            print(f"ERRO: checkpoint '{rotulo}' não encontrado em {caminho}.")
            print("Gere-o primeiro com training/self_play.py (veja o docstring deste arquivo).")
            sys.exit(1)

    n, seed, pasta = args.partidas, args.seed, args.saida

    print(f"=== 1/3: self-play+pool vs. AgenteEstrategico (n={n}) ===")
    r_sp = _avaliar_e_salvar("rl", "heuristico", args.pesos_self_play, None, n, seed,
                             pasta / "self_play_vs_heuristico")
    lo, hi = r_sp["ic95_vitoria"]
    print(f"  {r_sp['taxas']['vitoria']:.1%} vitórias | IC95%: [{lo:.1%}, {hi:.1%}]")

    print(f"\n=== 2/3: treinado-vs-heurístico vs. AgenteEstrategico (n={n}) ===")
    r_heur = _avaliar_e_salvar("rl", "heuristico", args.pesos_heuristico, None, n, seed,
                               pasta / "heuristico_vs_heuristico")
    lo, hi = r_heur["ic95_vitoria"]
    print(f"  {r_heur['taxas']['vitoria']:.1%} vitórias | IC95%: [{lo:.1%}, {hi:.1%}]")

    print(f"\n=== 3/3: self-play+pool vs. treinado-vs-heurístico (confronto direto, n={n}) ===")
    r_direto = _avaliar_e_salvar("rl", "rl", args.pesos_self_play, args.pesos_heuristico, n, seed,
                                  pasta / "self_play_vs_heuristico_direto")
    lo, hi = r_direto["ic95_vitoria"]
    print(f"  self-play+pool venceu {r_direto['taxas']['vitoria']:.1%} | IC95%: [{lo:.1%}, {hi:.1%}]")

    print("\n=== Conclusão ===")
    if _ic_se_sobrepoe(r_sp, r_heur):
        print(
            "Os IC 95% de vitória vs. o heurístico SE SOBREPÕEM entre os dois regimes de "
            "treino: não há evidência estatística, neste orçamento de partidas, de que um "
            "regime produz uma política melhor que o outro."
        )
    else:
        melhor = "self-play+pool" if r_sp["taxas"]["vitoria"] > r_heur["taxas"]["vitoria"] else "treinado-vs-heurístico"
        print(f"Os IC 95% NÃO se sobrepõem: '{melhor}' tem taxa de vitória vs. o heurístico "
              f"significativamente maior.")

    lo, hi = r_direto["ic95_vitoria"]
    if lo <= 0.5 <= hi:
        print(f"O confronto direto (IC95% [{lo:.1%}, {hi:.1%}]) inclui 50%: consistente com um "
              f"empate técnico entre os dois regimes.")
    else:
        favorito = "self-play+pool" if r_direto["taxas"]["vitoria"] > 0.5 else "treinado-vs-heurístico"
        print(f"O confronto direto (IC95% [{lo:.1%}, {hi:.1%}]) exclui 50%: '{favorito}' venceu "
              f"mais do que seria esperado por acaso.")

    print(f"\nArtefatos (resumo.json, partidas.csv, PNGs) salvos em {pasta}/")


if __name__ == "__main__":
    main()
