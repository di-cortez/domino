# UI

Interface visual do simulador (Pygame + PyOpenGL). Nenhum arquivo aqui decide
se uma jogada é válida — isso é sempre consultado ao `MotorDomino`; a UI apenas
organiza entrada de teclado, layout visual e desenho.

> Para o passo a passo detalhado de como os arquivos se encaixam (fluxo de uma
> jogada automática, fluxo de uma jogada humana, diagramas), veja
> [`funcionamento_ui.md`](funcionamento_ui.md). Este README foca em "o que cada
> arquivo faz"; aquele documento foca em "como eles conversam entre si".

## Arquivos

| Arquivo | Responsabilidade |
|---|---|
| `main_visual.py` | Ponto de entrada (`python -m ui.main_visual`). Cria janela, motor, agentes, `GerenciadorPartida`, `ControladorPartida`, `HudRenderer` e roda o laço principal a 60 FPS. |
| `controle_partida.py` | `ControladorPartida` — orquestrador da UI: histórico de snapshots, pausa/velocidade, turnos automáticos, menu, reinício com confirmação e notificações temporárias para a HUD. |
| `controle_humano.py` | `ControleHumanoMixin` — interação do turno humano: seleção de peça, escolha de ponta, jogar/comprar/passar e cálculo da seta de seleção na HUD. |
| `visibilidade_maos.py` | `VisibilidadeMaosMixin` — regras de quais mãos aparecem, conforme o modo: IA vs. IA, humano vs. IA (alternável com `J`/`K`) ou humano vs. humano (só a mão da vez). |
| `agentes_ui.py` | Fábrica central de agentes da UI (`criar_agente_por_tipo`) e nomes amigáveis (`nome_tipo_agente`) para os cinco tipos do menu. Define `AgenteHumanoBloqueado`, sentinela do turno humano. |
| `hud.py` | `HudRenderer` — camada 2D sobre a mesa: barra superior, mãos e monte, seta de seleção, notificações, fim de jogo, menu e barra de atalhos. Só lê o controlador; nunca altera a partida. |
| `interface.py` | `renderizar_cena(estado)` — desenha a mesa 3D a partir de um pivô estável, separando a cadeia em ramo esquerdo e direito para peças já colocadas não deslizarem. |
| `layout_domino.py` | Geometria pura (não desenha nada): direção dos ramos, ângulo de cada peça, ordem dos valores, posição em linha e posição ao virar um canto perto do limite da mesa. |
| `desenho_domino.py` | `desenhar_peca(...)` — aplica a transformação OpenGL de uma peça 3D a partir da posição/ângulo calculados por `layout_domino.py`. |
| `primitivas.py` | Funções básicas de desenho OpenGL/Pygame: retângulo, contorno, linha, triângulo, círculo, texto, modo 2D da HUD e dominó (3D e 2D, incluindo o verso). |
| `renderizador_estado.py` | `RenderizadorEstado` — guarda a peça-pivô da mesa e a reencontra em cada snapshot, para a cadeia não "pular" visualmente entre frames. |
| `config_visual.py` | Constantes de geometria da mesa: posição inicial, escala da peça, espaçamento e limites que o layout usa para decidir quando virar um ramo. |
| `testes_unitarios_ui.py` | Testes sequenciais sem pytest (`python ui/testes_unitarios_ui.py`): avanço automático, turno humano, seleção/ponta, velocidade, reinício e visibilidade das mãos. |
| `funcionamento_ui.md` | Documento narrativo (não é código) com o passo a passo de como os arquivos se encaixam, incluindo os fluxos de jogada automática/humana e a lista de atalhos. |

## Atalhos e menu

Ver as seções **Controles da simulação visual** e **HUD** no `README.md` da
raiz do projeto para a lista completa de teclas e o comportamento do menu de
configuração.
