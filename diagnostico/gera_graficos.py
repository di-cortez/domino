"""
Métricas e gráficos do diagnóstico.

Este módulo não conhece o motor de dominó: recebe a lista de partidas
produzida por diagnostico/avaliar.py (uma linha por partida, com posição,
resultado, turnos e pips) e produz o resumo estatístico e os PNGs.
"""

import math

import numpy as np


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def intervalo_wilson(sucessos, total, z=1.96):
    """Intervalo de confiança de Wilson (95%) para uma proporção."""
    if total == 0:
        return 0.0, 0.0
    p = sucessos / total
    denom = 1 + z * z / total
    centro = (p + z * z / (2 * total)) / denom
    raio = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centro - raio), min(1.0, centro + raio)


def resumir(partidas, nome_agente, nome_oponente, seed):
    """Agrega as partidas em um dicionário de métricas (taxas, IC 95%,
    desempenho por posição, duração e pips restantes)."""
    n = len(partidas)
    contagem = {r: sum(1 for p in partidas if p["resultado"] == r)
                for r in ("vitoria", "empate", "derrota")}

    resumo = {
        "agente": nome_agente,
        "oponente": nome_oponente,
        "num_partidas": n,
        "seed": seed,
        "contagem": contagem,
        "taxas": {r: c / n for r, c in contagem.items()},
        "ic95_vitoria": intervalo_wilson(contagem["vitoria"], n),
        "ic95_empate": intervalo_wilson(contagem["empate"], n),
        "turnos_media": float(np.mean([p["turnos"] for p in partidas])),
        "turnos_desvio": float(np.std([p["turnos"] for p in partidas])),
        "pips_restantes_media_agente": float(np.mean([p["pips_agente"] for p in partidas])),
        "pips_restantes_media_oponente": float(np.mean([p["pips_oponente"] for p in partidas])),
        "por_posicao": {},
    }

    for pos in (0, 1):
        grupo = [p for p in partidas if p["posicao_agente"] == pos]
        vit = sum(1 for p in grupo if p["resultado"] == "vitoria")
        resumo["por_posicao"][str(pos)] = {
            "partidas": len(grupo),
            "vitorias": vit,
            "taxa_vitoria": vit / len(grupo) if grupo else 0.0,
            "ic95": intervalo_wilson(vit, len(grupo)),
        }
    return resumo


# ---------------------------------------------------------------------------
# Gráficos — paleta e anatomia seguindo um estilo único e acessível.
# ---------------------------------------------------------------------------
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
MUTED = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"

# Ordem categórica fixa (validada p/ daltonismo): vitória, empate, derrota.
COR = {"vitoria": "#2a78d6", "empate": "#1baf7a", "derrota": "#eda100"}
ROTULO = {"vitoria": "Vitória", "empate": "Empate", "derrota": "Derrota"}


def _preparar_eixo(ax, titulo):
    ax.set_facecolor(SUPERFICIE)
    ax.set_title(titulo, color=TINTA, fontsize=12, loc="left", pad=12)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("bottom", "left"):
        ax.spines[lado].set_color(EIXO)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.grid(axis="y", color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)


def _nova_figura(largura=8.0, altura=4.5):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(largura, altura), facecolor=SUPERFICIE, dpi=150)
    return fig, ax


def _salvar_figura(fig, caminho):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    fig.savefig(caminho, facecolor=SUPERFICIE)
    plt.close(fig)


def grafico_taxas_acumuladas(partidas, caminho, subtitulo):
    """Taxas acumuladas de vitória/empate/derrota ao longo das partidas."""
    fig, ax = _nova_figura()
    _preparar_eixo(ax, f"Taxas acumuladas por partida — {subtitulo}")

    n = len(partidas)
    x = np.arange(1, n + 1)
    for chave in ("vitoria", "empate", "derrota"):
        acumulado = np.cumsum([p["resultado"] == chave for p in partidas]) / x
        ax.plot(x, 100 * acumulado, color=COR[chave], linewidth=2, label=ROTULO[chave])
        # Rótulo direto no fim da linha (as cores claras exigem reforço textual).
        ax.annotate(f"{ROTULO[chave]} {100 * acumulado[-1]:.1f}%",
                    xy=(n, 100 * acumulado[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center",
                    color=TINTA_SECUNDARIA, fontsize=9)

    ax.set_xlim(1, n * 1.18)  # folga à direita para os rótulos diretos
    ax.set_ylim(0, 100)
    ax.set_xlabel("Número de partidas")
    ax.set_ylabel("Taxa acumulada (%)")
    ax.legend(frameon=False, labelcolor=TINTA_SECUNDARIA, fontsize=9, loc="upper right")
    _salvar_figura(fig, caminho)


def grafico_distribuicao(resumo, caminho, subtitulo):
    """Distribuição final dos resultados (barras horizontais)."""
    fig, ax = _nova_figura(altura=3.2)
    _preparar_eixo(ax, f"Distribuição dos resultados — {subtitulo}")
    ax.grid(axis="x", color=GRADE, linewidth=0.8)
    ax.grid(axis="y", visible=False)

    chaves = ["derrota", "empate", "vitoria"]  # vitória no topo
    valores = [resumo["contagem"][c] for c in chaves]
    cores = [COR[c] for c in chaves]
    ax.barh([ROTULO[c] for c in chaves], valores, color=cores, height=0.55)

    n = resumo["num_partidas"]
    for i, (c, v) in enumerate(zip(chaves, valores)):
        ax.annotate(f"{v} ({100 * v / n:.1f}%)", xy=(v, i), xytext=(6, 0),
                    textcoords="offset points", va="center",
                    color=TINTA_SECUNDARIA, fontsize=9)

    ax.set_xlim(0, max(valores) * 1.22 if max(valores) else 1)
    ax.set_xlabel("Partidas")
    ax.tick_params(axis="y", labelcolor=TINTA)
    _salvar_figura(fig, caminho)


def grafico_por_posicao(resumo, caminho, subtitulo):
    """Taxa de vitória do agente conforme a posição em que jogou (com IC 95%)."""
    fig, ax = _nova_figura(largura=7.5, altura=4.2)
    _preparar_eixo(ax, f"Taxa de vitória por posição inicial — {subtitulo}")

    posicoes = ["0", "1"]
    taxas = [100 * resumo["por_posicao"][p]["taxa_vitoria"] for p in posicoes]
    erros_inf = [100 * (resumo["por_posicao"][p]["taxa_vitoria"] - resumo["por_posicao"][p]["ic95"][0])
                 for p in posicoes]
    erros_sup = [100 * (resumo["por_posicao"][p]["ic95"][1] - resumo["por_posicao"][p]["taxa_vitoria"])
                 for p in posicoes]

    # Mesma medida nas duas barras -> um único matiz (magnitude, não identidade).
    ax.bar(["Jogador 0", "Jogador 1"], taxas, color=COR["vitoria"], width=0.5,
           yerr=[erros_inf, erros_sup], ecolor=TINTA_SECUNDARIA, capsize=4)

    for i, p in enumerate(posicoes):
        info = resumo["por_posicao"][p]
        # Rótulo acima do limite superior do IC, para não colidir com a barra de erro.
        ax.annotate(f"{100 * info['taxa_vitoria']:.1f}%  (n={info['partidas']})",
                    xy=(i, taxas[i] + erros_sup[i]), xytext=(0, 8),
                    textcoords="offset points",
                    ha="center", color=TINTA_SECUNDARIA, fontsize=9)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Taxa de vitória (%)")
    ax.tick_params(axis="x", labelcolor=TINTA)
    _salvar_figura(fig, caminho)


def grafico_duracao(partidas, caminho, subtitulo):
    """Histograma da duração das partidas (turnos)."""
    fig, ax = _nova_figura(altura=3.8)
    _preparar_eixo(ax, f"Duração das partidas — {subtitulo}")

    turnos = [p["turnos"] for p in partidas]
    ax.hist(turnos, bins=min(30, max(5, len(set(turnos)))),
            color=COR["vitoria"], edgecolor=SUPERFICIE, linewidth=1)
    media = float(np.mean(turnos))
    ax.axvline(media, color=TINTA_SECUNDARIA, linewidth=1, linestyle="--")
    ax.annotate(f"média {media:.1f}", xy=(media, ax.get_ylim()[1]), xytext=(6, -12),
                textcoords="offset points", color=TINTA_SECUNDARIA, fontsize=9)

    ax.set_xlabel("Turnos por partida")
    ax.set_ylabel("Partidas")
    _salvar_figura(fig, caminho)


def gerar_graficos(partidas, resumo, pasta):
    """Gera os quatro PNGs do diagnóstico na pasta indicada."""
    import matplotlib
    matplotlib.use("Agg")

    subtitulo = f"{resumo['agente']} vs {resumo['oponente']} ({resumo['num_partidas']} partidas)"
    grafico_taxas_acumuladas(partidas, pasta / "taxas_acumuladas.png", subtitulo)
    grafico_distribuicao(resumo, pasta / "distribuicao_resultados.png", subtitulo)
    grafico_por_posicao(resumo, pasta / "vitorias_por_posicao.png", subtitulo)
    grafico_duracao(partidas, pasta / "duracao_partidas.png", subtitulo)
