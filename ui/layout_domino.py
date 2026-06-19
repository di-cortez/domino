"""
Geometria da cadeia de dominós na mesa.

Este módulo não desenha. Ele recebe a cadeia visual do motor e calcula posições,
ângulos e ordem dos valores para cada peça, respeitando limites da mesa e
virando os ramos quando a linha chega perto das bordas.
"""

from ui.config_visual import (
    GAP_ENTRE_PECAS,
    LIMITE_X,
    LIMITE_Y_INFERIOR,
    LIMITE_Y_SUPERIOR,
    LARGURA_PECA_DEITADA,
    ALTURA_PECA_DEITADA,
    LARGURA_PECA_EM_PE,
    ALTURA_PECA_EM_PE,
)


# ============================================================
# Percursos dos ramos
# ============================================================

PERCURSO_DIREITA = [
    {"nome": "topo_direita", "dx": 1, "dy": 0},
    {"nome": "desce_direita", "dx": 0, "dy": -1},
    {"nome": "fundo_esquerda", "dx": -1, "dy": 0},
    {"nome": "sobe_esquerda", "dx": 0, "dy": 1},
    {"nome": "topo_direita_2", "dx": 1, "dy": 0},
]

PERCURSO_ESQUERDA = [
    {"nome": "topo_esquerda", "dx": -1, "dy": 0},
    {"nome": "desce_esquerda", "dx": 0, "dy": -1},
    {"nome": "fundo_direita", "dx": 1, "dy": 0},
    {"nome": "sobe_direita", "dx": 0, "dy": 1},
    {"nome": "topo_esquerda_2", "dx": -1, "dy": 0},
]


def obter_percurso(direcao):
    if direcao == 1:
        return PERCURSO_DIREITA

    if direcao == -1:
        return PERCURSO_ESQUERDA

    raise ValueError(f"Direção de ramo inválida: {direcao}")


# ============================================================
# Peças e valores
# ============================================================

def eh_bucha(info):
    a, b = info["peca"]
    return a == b


def valores_peca(info):
    return tuple(info["peca"])


def valor_comum(info_anterior, info_atual):
    """
    Valor pelo qual info_atual encosta em info_anterior.

    Primeiro tenta usar valor_conectado, se o motor tiver enviado.
    Caso contrário, calcula pela interseção dos valores das duas peças.
    """
    if "valor_conectado" in info_atual:
        return info_atual["valor_conectado"]

    a1, b1 = valores_peca(info_anterior)
    a2, b2 = valores_peca(info_atual)

    valores_anteriores = {a1, b1}

    if a2 in valores_anteriores:
        return a2

    if b2 in valores_anteriores:
        return b2

    raise ValueError(
        f"Peças consecutivas sem valor comum: "
        f"{info_anterior['peca']} e {info_atual['peca']}"
    )


def outro_valor(info, conectado):
    """
    Dada uma peça e o valor conectado, devolve o outro valor da peça.
    """
    a, b = valores_peca(info)

    if a == conectado:
        return b

    if b == conectado:
        return a

    raise ValueError(
        f"Valor conectado {conectado} não aparece na peça {info['peca']}"
    )


# ============================================================
# Direções, ângulos e dimensões
# ============================================================

def movimento_horizontal(dx, dy):
    return dx != 0 and dy == 0


def movimento_vertical(dx, dy):
    return dx == 0 and dy != 0


def angulo_para_direcao(info, dx, dy):
    """
    Regra geral:

    Movimento horizontal:
        - não-bucha: deitada;
        - bucha: em pé.

    Movimento vertical:
        - não-bucha: em pé;
        - bucha: deitada.
    """
    if movimento_horizontal(dx, dy):
        if eh_bucha(info):
            return 90.0

        return 0.0

    if movimento_vertical(dx, dy):
        if eh_bucha(info):
            return 0.0

        return 90.0

    raise ValueError(f"Direção inválida: dx={dx}, dy={dy}")


def angulo_em_linha(info):
    """
    Compatibilidade com o desenho do pivô:
    linha horizontal padrão.
    """
    return angulo_para_direcao(info, dx=1, dy=0)


def angulo_descendo(info):
    """
    Compatibilidade com versões anteriores:
    trecho vertical descendo.
    """
    return angulo_para_direcao(info, dx=0, dy=-1)


def angulo_para_corner(info, segmento_antigo, segmento_novo):
    """
    Ângulo da peça que entra no corner.

    Regra prática:
    - peça comum usa a direção nova;
    - bucha usa a direção antiga.

    Isso preserva o comportamento que já funcionou:
    quando a primeira peça depois do corner é bucha, ela fica atravessada
    em relação ao trecho anterior.
    """
    if eh_bucha(info):
        return angulo_para_direcao(
            info,
            segmento_antigo["dx"],
            segmento_antigo["dy"],
        )

    return angulo_para_direcao(
        info,
        segmento_novo["dx"],
        segmento_novo["dy"],
    )


def largura_por_angulo(angulo):
    if angulo in (90.0, -90.0):
        return LARGURA_PECA_EM_PE

    return LARGURA_PECA_DEITADA


def altura_por_angulo(angulo):
    if angulo in (90.0, -90.0):
        return ALTURA_PECA_EM_PE

    return ALTURA_PECA_DEITADA


def dimensoes_por_angulo(angulo):
    return largura_por_angulo(angulo), altura_por_angulo(angulo)


def extensao_no_eixo(angulo, dx, dy):
    """
    Devolve o tamanho da peça no eixo em que ela vai andar.

    Se anda horizontalmente, usa largura.
    Se anda verticalmente, usa altura.
    """
    largura, altura = dimensoes_por_angulo(angulo)

    if movimento_horizontal(dx, dy):
        return largura

    if movimento_vertical(dx, dy):
        return altura

    raise ValueError(f"Direção inválida: dx={dx}, dy={dy}")


# ============================================================
# Ordem dos valores desenhados
# ============================================================

def valores_para_direcao(info_anterior, info_atual, dx, dy):
    """
    Escolhe a ordem dos valores de info_atual para que o valor conectado
    fique do lado de entrada da peça.

    dx =  1, dy =  0: anda para direita
        conectado fica à esquerda.

    dx = -1, dy =  0: anda para esquerda
        conectado fica à direita.

    dx =  0, dy = -1: desce
        conectado fica em cima.

    dx =  0, dy =  1: sobe
        conectado fica embaixo.
    """
    conectado = valor_comum(info_anterior, info_atual)
    outro = outro_valor(info_atual, conectado)

    if dx == 1 and dy == 0:
        return conectado, outro

    if dx == -1 and dy == 0:
        return outro, conectado

    if dx == 0 and dy == -1:
        # Com rotação de 90 graus:
        # primeiro valor fica embaixo;
        # segundo valor fica em cima.
        return outro, conectado

    if dx == 0 and dy == 1:
        # Subindo:
        # conectado precisa ficar embaixo.
        return conectado, outro

    raise ValueError(f"Direção inválida: dx={dx}, dy={dy}")


# ============================================================
# Caixas e limites
# ============================================================

def cabe_na_area(pos_x, pos_y, angulo):
    """
    Verifica se a caixa ocupada pela peça cabe nos limites da mesa.

    Usa largura e altura reais da peça depois da orientação.
    """
    largura, altura = dimensoes_por_angulo(angulo)

    esquerda = pos_x - largura / 2.0
    direita = pos_x + largura / 2.0
    baixo = pos_y - altura / 2.0
    cima = pos_y + altura / 2.0

    if esquerda < -LIMITE_X:
        return False

    if direita > LIMITE_X:
        return False

    if baixo < LIMITE_Y_INFERIOR:
        return False

    if cima > LIMITE_Y_SUPERIOR:
        return False

    return True


# ============================================================
# Posição da próxima peça
# ============================================================

def posicao_em_linha(
    pos_x_atual,
    pos_y_atual,
    segmento,
    angulo_anterior,
    angulo_atual,
):
    """
    Calcula a posição da próxima peça continuando no mesmo segmento.
    """
    dx = segmento["dx"]
    dy = segmento["dy"]

    passo = (
        extensao_no_eixo(angulo_anterior, dx, dy) / 2.0
        + extensao_no_eixo(angulo_atual, dx, dy) / 2.0
        + GAP_ENTRE_PECAS
    )

    return (
        pos_x_atual + dx * passo,
        pos_y_atual + dy * passo,
    )


def deslocamento_saida_corner(info_anterior, angulo_anterior, segmento_antigo):
    """
    Ajuste fino para o ponto de saída no corner.

    Se a peça anterior não é bucha, a conexão sai aproximadamente do centro
    da metade externa da peça, não do centro da peça inteira.

    Se é bucha, usamos o centro.
    """
    if eh_bucha(info_anterior):
        return 0.0

    dx = segmento_antigo["dx"]
    dy = segmento_antigo["dy"]

    if movimento_horizontal(dx, dy):
        return largura_por_angulo(angulo_anterior) / 4.0

    if movimento_vertical(dx, dy):
        return altura_por_angulo(angulo_anterior) / 4.0

    raise ValueError(f"Segmento inválido: {segmento_antigo}")


def posicao_em_corner(
    pos_x_atual,
    pos_y_atual,
    info_anterior,
    segmento_antigo,
    segmento_novo,
    angulo_anterior,
    angulo_atual,
):
    """
    Calcula a posição da primeira peça depois de virar o corner.

    Casos:
    - horizontal -> vertical:
        ajusta x pela metade externa da peça anterior;
        desce/sobe conforme o novo segmento.

    - vertical -> horizontal:
        ajusta y pela metade externa da peça anterior;
        anda para o lado conforme o novo segmento.
    """
    dx_antigo = segmento_antigo["dx"]
    dy_antigo = segmento_antigo["dy"]

    dx_novo = segmento_novo["dx"]
    dy_novo = segmento_novo["dy"]

    largura_atual, altura_atual = dimensoes_por_angulo(angulo_atual)

    saida = deslocamento_saida_corner(
        info_anterior,
        angulo_anterior,
        segmento_antigo,
    )

    # Horizontal -> vertical.
    if movimento_horizontal(dx_antigo, dy_antigo) and movimento_vertical(dx_novo, dy_novo):
        novo_x = pos_x_atual + dx_antigo * saida

        passo_y = (
            altura_por_angulo(angulo_anterior) / 2.0
            + altura_atual / 2.0
            + GAP_ENTRE_PECAS
        )

        novo_y = pos_y_atual + dy_novo * passo_y

        return novo_x, novo_y

    # Vertical -> horizontal.
    if movimento_vertical(dx_antigo, dy_antigo) and movimento_horizontal(dx_novo, dy_novo):
        passo_x = (
            largura_por_angulo(angulo_anterior) / 2.0
            + largura_atual / 2.0
            + GAP_ENTRE_PECAS
        )

        novo_x = pos_x_atual + dx_novo * passo_x
        novo_y = pos_y_atual + dy_antigo * saida

        return novo_x, novo_y

    # Fallback: se por algum motivo o percurso virar de modo estranho,
    # usa a regra simples de linha no segmento novo.
    return posicao_em_linha(
        pos_x_atual,
        pos_y_atual,
        segmento_novo,
        angulo_anterior,
        angulo_atual,
    )


# ============================================================
# Slots
# ============================================================

def criar_slot(
    pos_x,
    pos_y,
    info_anterior,
    info_atual,
    segmento,
    indice_segmento,
    tipo,
    subtipo,
    angulo,
):
    dx = segmento["dx"]
    dy = segmento["dy"]

    return {
        "tipo": tipo,
        "subtipo": subtipo,
        "segmento": segmento["nome"],
        "indice_segmento": indice_segmento,
        "dx": dx,
        "dy": dy,
        "pos_x": pos_x,
        "pos_y": pos_y,
        "angulo": angulo,
        "valores": valores_para_direcao(info_anterior, info_atual, dx, dy),
        "proximo_segmento": indice_segmento,
    }


def tipo_slot_para_segmento(segmento):
    if movimento_vertical(segmento["dx"], segmento["dy"]):
        return "vertical"

    return "linha"


def calcular_proximo_slot(
    pos_x_atual,
    pos_y_atual,
    info_anterior,
    info_atual,
    angulo_anterior,
    percurso,
    indice_segmento_atual,
):
    """
    Tenta colocar a peça no segmento atual.

    Se a caixa da nova peça extrapola os limites da mesa,
    avança para o próximo segmento e cria um corner.
    """
    segmento_atual = percurso[indice_segmento_atual]

    angulo_atual = angulo_para_direcao(
        info_atual,
        segmento_atual["dx"],
        segmento_atual["dy"],
    )

    candidato_x, candidato_y = posicao_em_linha(
        pos_x_atual,
        pos_y_atual,
        segmento_atual,
        angulo_anterior,
        angulo_atual,
    )

    if cabe_na_area(candidato_x, candidato_y, angulo_atual):
        tipo = tipo_slot_para_segmento(segmento_atual)

        return criar_slot(
            candidato_x,
            candidato_y,
            info_anterior,
            info_atual,
            segmento_atual,
            indice_segmento_atual,
            tipo=tipo,
            subtipo=None,
            angulo=angulo_atual,
        )

    # Não coube no segmento atual: vira para o próximo segmento.
    proximo_indice = min(
        indice_segmento_atual + 1,
        len(percurso) - 1,
    )

    segmento_novo = percurso[proximo_indice]

    angulo_corner = angulo_para_corner(
        info_atual,
        segmento_atual,
        segmento_novo,
    )

    corner_x, corner_y = posicao_em_corner(
        pos_x_atual,
        pos_y_atual,
        info_anterior,
        segmento_atual,
        segmento_novo,
        angulo_anterior,
        angulo_corner,
    )

    return criar_slot(
        corner_x,
        corner_y,
        info_anterior,
        info_atual,
        segmento_novo,
        proximo_indice,
        tipo="corner",
        subtipo=f"{segmento_atual['nome']}_para_{segmento_novo['nome']}",
        angulo=angulo_corner,
    )


# ============================================================
# Cadeia e ramos
# ============================================================

def quebrar_cadeia_no_pivo(cadeia_visual, indice_pivo):
    """
    cadeia_visual vem na ordem esquerda -> direita.

    Antes do pivô: lado esquerdo.
    Depois do pivô: lado direito.

    Para desenhar a partir do pivô para fora:
    - lado esquerdo precisa ser invertido;
    - lado direito já está na ordem correta.
    """
    pivo = cadeia_visual[indice_pivo]

    lado_esquerdo = list(reversed(cadeia_visual[:indice_pivo]))
    lado_direito = cadeia_visual[indice_pivo + 1:]

    return lado_esquerdo, pivo, lado_direito


def calcular_slots_ramo(
    pecas,
    peca_anterior,
    direcao,
    pos_x_inicial,
    pos_y_inicial,
):
    """
    Calcula os slots de um ramo a partir do pivô.

    Não desenha nada.

    Retorna lista de pares:
        (info_da_peca, slot)

    direcao =  1 -> lado direito
    direcao = -1 -> lado esquerdo.
    """
    resultado = []

    percurso = obter_percurso(direcao)

    pos_x_atual = pos_x_inicial
    pos_y_atual = pos_y_inicial

    info_anterior = peca_anterior
    angulo_anterior = angulo_em_linha(peca_anterior)

    indice_segmento_atual = 0

    for info_atual in pecas:
        slot = calcular_proximo_slot(
            pos_x_atual,
            pos_y_atual,
            info_anterior,
            info_atual,
            angulo_anterior,
            percurso,
            indice_segmento_atual,
        )

        resultado.append((info_atual, slot))

        pos_x_atual = slot["pos_x"]
        pos_y_atual = slot["pos_y"]
        angulo_anterior = slot["angulo"]
        indice_segmento_atual = slot["proximo_segmento"]

        info_anterior = info_atual

    return resultado
