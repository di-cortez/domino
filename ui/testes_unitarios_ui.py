"""
Testes unitários da camada de UI/controlador.

O arquivo é propositalmente executável direto, sem pytest, para facilitar em
aula: `python ui/testes_unitarios_ui.py`. Os testes evitam abrir janela OpenGL
e exercitam a parte determinística do controlador.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.heuristic_agent import AgenteEstrategico
from middleware.middleware import GerenciadorPartida
from middleware.motor_domino import MotorDomino
from ui.controle_partida import ControladorPartida


def _novo_controlador(tipos=None, intervalo_ms=1000):
    motor = MotorDomino(num_jogadores=2)
    agentes = [AgenteEstrategico(), AgenteEstrategico()]
    gerenciador = GerenciadorPartida(motor, agentes)

    if tipos is None:
        tipos = ["heuristico", "heuristico"]

    controlador = ControladorPartida(
        gerenciador,
        motor,
        intervalo_ms=intervalo_ms,
        tipos_agentes=tipos,
    )

    return motor, controlador


def _novo_controlador_com_humano_da_vez():
    motor = MotorDomino(num_jogadores=2)
    tipos = ["heuristico", "heuristico"]
    tipos[motor.jogador_atual] = "humano"

    agentes = [AgenteEstrategico(), AgenteEstrategico()]
    gerenciador = GerenciadorPartida(motor, agentes)

    controlador = ControladorPartida(
        gerenciador,
        motor,
        intervalo_ms=1000,
        tipos_agentes=tipos,
    )

    return motor, controlador


def _forcar_turno_humano(controlador, motor, jogador=0):
    controlador.tipos_agentes = ["heuristico", "heuristico"]
    controlador.tipos_agentes[jogador] = "humano"
    motor.jogador_atual = jogador
    controlador._configurar_visibilidade_maos_por_modo()
    controlador.historico = []
    controlador.historico_info = []
    controlador.indice = 0
    controlador._chave_selecao_humana = None
    controlador._capturar_estado()
    controlador._sincronizar_selecao_humana()


def _preparar_estado_humano(controlador, motor, pontas, mao, jogador=0):
    controlador.tipos_agentes = ["heuristico", "heuristico"]
    controlador.tipos_agentes[jogador] = "humano"
    motor.jogador_atual = jogador
    motor.pontas = list(pontas)
    motor.maos[jogador] = list(mao)
    motor.primeira_peca_obrigatoria = None
    motor.comprou_neste_turno[jogador] = False

    controlador._configurar_visibilidade_maos_por_modo()
    controlador.historico = []
    controlador.historico_info = []
    controlador.indice = 0
    controlador._chave_selecao_humana = None
    controlador._capturar_estado()
    controlador._sincronizar_selecao_humana()


def _validas(controlador, jogador):
    return controlador._acoes_validas_set(jogador)


def _rodar(nome, fn):
    fn()
    print(f"OK - {nome}")


def teste_ia_avanca_automaticamente():
    motor, controlador = _novo_controlador(intervalo_ms=1)
    turno = motor.turno

    controlador.atualizar(1000)

    assert motor.turno > turno


def teste_humano_nao_avanca_automaticamente():
    motor, controlador = _novo_controlador_com_humano_da_vez()
    turno = motor.turno

    controlador.atualizar(5000)

    assert motor.turno == turno
    assert controlador.jogador_humano_ativo()


def teste_selecao_inicial_humana_e_valida_quando_possivel():
    motor, controlador = _novo_controlador_com_humano_da_vez()
    jogador = motor.jogador_atual
    acao = controlador._acao_humana_selecionada()

    assert controlador.indice_peca_selecionada < len(motor.maos[jogador])

    if acao is not None:
        assert acao in _validas(controlador, jogador)


def teste_navegacao_humana_circular():
    motor, controlador = _novo_controlador_com_humano_da_vez()
    jogador = motor.jogador_atual
    n = len(motor.maos[jogador])

    controlador.indice_peca_selecionada = 0
    controlador._navegar_peca_humana(-1)

    assert controlador.indice_peca_selecionada == n - 1

    controlador._navegar_peca_humana(1)

    assert controlador.indice_peca_selecionada == 0


def teste_tab_alterna_ponta():
    motor, controlador = _novo_controlador()
    _preparar_estado_humano(controlador, motor, pontas=[1, 2], mao=[(1, 2)])

    controlador._alternar_ponta_humana()

    assert controlador.ponta_selecionada == "esquerda"

    controlador._alternar_ponta_humana()

    assert controlador.ponta_selecionada == "direita"


def teste_navegacao_atualiza_ponta_valida():
    motor, controlador = _novo_controlador()
    _preparar_estado_humano(
        controlador,
        motor,
        pontas=[1, 2],
        mao=[(1, 3), (4, 2), (1, 2), (4, 5)],
    )

    assert controlador.indice_peca_selecionada == 0
    assert controlador.ponta_selecionada == "esquerda"

    controlador._navegar_peca_humana(1)

    assert controlador.indice_peca_selecionada == 1
    assert controlador.ponta_selecionada == "direita"

    controlador._navegar_peca_humana(1)

    assert controlador.indice_peca_selecionada == 2
    assert controlador.ponta_selecionada == "direita"

    controlador._alternar_ponta_humana()

    assert controlador.ponta_selecionada == "esquerda"


def teste_pontas_iguais_indicam_direita_mas_acao_do_motor_e_valida():
    motor, controlador = _novo_controlador()
    _preparar_estado_humano(
        controlador,
        motor,
        pontas=[3, 3],
        mao=[(3, 5)],
    )

    assert controlador.ponta_selecionada == "direita"
    assert controlador._acao_humana_selecionada() == ((3, 5), 0)


def teste_seta_indica_metade_de_baixo_da_peca():
    motor, controlador = _novo_controlador()
    _preparar_estado_humano(
        controlador,
        motor,
        pontas=[2, 3],
        mao=[(1, 2)],
    )

    assert controlador.ponta_selecionada == "esquerda"
    assert controlador.posicao_seta_peca_selecionada() == "baixo"


def teste_seta_indica_metade_de_cima_da_peca():
    motor, controlador = _novo_controlador()
    _preparar_estado_humano(
        controlador,
        motor,
        pontas=[2, 3],
        mao=[(2, 1)],
    )

    assert controlador.ponta_selecionada == "esquerda"
    assert controlador.posicao_seta_peca_selecionada() == "cima"


def teste_seta_peca_que_joga_dos_dois_lados_usa_valor_da_ponta():
    motor, controlador = _novo_controlador()
    _preparar_estado_humano(
        controlador,
        motor,
        pontas=[2, 3],
        mao=[(2, 3)],
    )

    assert controlador.ponta_selecionada == "direita"
    assert controlador.posicao_seta_peca_selecionada() == "baixo"


def teste_enter_humano_executa_jogada_valida():
    motor, controlador = _novo_controlador_com_humano_da_vez()
    turno = motor.turno

    controlador._jogar_peca_humana()

    assert motor.turno > turno
    assert len(controlador.historico) == 2


def teste_velocidades_tem_limites():
    _motor, controlador = _novo_controlador()

    assert controlador._texto_velocidade() == "1x"

    for _ in range(10):
        controlador._alterar_velocidade(1)

    assert controlador._texto_velocidade() == "4x"
    assert controlador._intervalo_atual_ms() == 250.0

    for _ in range(10):
        controlador._alterar_velocidade(-1)

    assert controlador._texto_velocidade() == "1/4x"
    assert controlador._intervalo_atual_ms() == 4000.0


def teste_reinicio_pede_confirmacao_e_expira():
    motor, controlador = _novo_controlador(intervalo_ms=1)
    turno = motor.turno

    controlador._atalho_reiniciar()

    assert controlador._confirmacao_reinicio_ativa()
    assert controlador.pausado
    assert motor.turno == turno

    controlador.atualizar(2100)

    assert not controlador._confirmacao_reinicio_ativa()
    assert not controlador.pausado
    assert motor.turno > turno


def teste_reinicio_segundo_r_confirma():
    motor, controlador = _novo_controlador()

    controlador.avancar()
    assert motor.turno > 0

    controlador._atalho_reiniciar()
    controlador._atalho_reiniciar()

    assert motor.turno == 0
    assert len(controlador.historico) == 1
    assert not controlador._confirmacao_reinicio_ativa()


def teste_reinicio_fim_de_jogo_e_direto():
    motor, controlador = _novo_controlador()
    controlador.fim_de_jogo = True
    controlador.info_final = {"vencedor": 0}

    controlador._atalho_reiniciar()

    assert motor.turno == 0
    assert not controlador.fim_de_jogo
    assert not controlador._confirmacao_reinicio_ativa()


def teste_visibilidade_ia_vs_ia_nao_oculta():
    _motor, controlador = _novo_controlador()

    assert not controlador.mao_oculta(0)
    assert not controlador.mao_oculta(1)

    controlador._alternar_visibilidade_mao(0)

    assert not controlador.mao_oculta(0)
    assert controlador.notificacao["texto"] == "IA vs IA: mãos sempre visíveis"


def teste_visibilidade_humano_vs_ia():
    motor, controlador = _novo_controlador()
    _forcar_turno_humano(controlador, motor, jogador=0)

    assert not controlador.mao_oculta(0)
    assert controlador.mao_oculta(1)

    controlador._alternar_visibilidade_mao(1)

    assert not controlador.mao_oculta(1)

    controlador._alternar_visibilidade_mao(1)

    assert controlador.mao_oculta(1)

    controlador._alternar_visibilidade_mao(0)

    assert not controlador.mao_oculta(0)
    assert controlador.notificacao["texto"] == "A mão do humano fica sempre visível"


def teste_visibilidade_humano_vs_humano_so_mao_da_vez():
    motor, controlador = _novo_controlador(tipos=["humano", "humano"])
    motor.jogador_atual = 0
    controlador.historico = []
    controlador.historico_info = []
    controlador.indice = 0
    controlador._capturar_estado()

    assert not controlador.mao_oculta(0)
    assert controlador.mao_oculta(1)

    motor.jogador_atual = 1
    controlador.historico = []
    controlador.historico_info = []
    controlador.indice = 0
    controlador._capturar_estado()

    assert controlador.mao_oculta(0)
    assert not controlador.mao_oculta(1)

    controlador._alternar_visibilidade_mao(0)

    assert controlador.mao_oculta(0)
    assert controlador.notificacao["texto"] == "Humano vs humano: só a mão da vez aparece"


def main():
    testes = [
        ("IA avanca automaticamente", teste_ia_avanca_automaticamente),
        ("humano nao avanca automaticamente", teste_humano_nao_avanca_automaticamente),
        ("selecao inicial humana", teste_selecao_inicial_humana_e_valida_quando_possivel),
        ("navegacao humana circular", teste_navegacao_humana_circular),
        ("Tab alterna ponta", teste_tab_alterna_ponta),
        ("navegacao atualiza ponta valida", teste_navegacao_atualiza_ponta_valida),
        ("pontas iguais usam direita visual", teste_pontas_iguais_indicam_direita_mas_acao_do_motor_e_valida),
        ("seta indica metade de baixo", teste_seta_indica_metade_de_baixo_da_peca),
        ("seta indica metade de cima", teste_seta_indica_metade_de_cima_da_peca),
        ("seta usa valor da ponta selecionada", teste_seta_peca_que_joga_dos_dois_lados_usa_valor_da_ponta),
        ("Enter humano executa jogada valida", teste_enter_humano_executa_jogada_valida),
        ("velocidades", teste_velocidades_tem_limites),
        ("reinicio pede confirmacao e expira", teste_reinicio_pede_confirmacao_e_expira),
        ("reinicio confirma no segundo R", teste_reinicio_segundo_r_confirma),
        ("reinicio no fim de jogo", teste_reinicio_fim_de_jogo_e_direto),
        ("visibilidade IA vs IA", teste_visibilidade_ia_vs_ia_nao_oculta),
        ("visibilidade humano vs IA", teste_visibilidade_humano_vs_ia),
        ("visibilidade humano vs humano", teste_visibilidade_humano_vs_humano_so_mao_da_vez),
    ]

    for nome, fn in testes:
        _rodar(nome, fn)

    print(f"\n{len(testes)} testes passaram.")


if __name__ == "__main__":
    main()
