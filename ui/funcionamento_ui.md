# Funcionamento da UI de dominó

Este documento resume como os arquivos da pasta `ui` se encaixam.

## Visão geral

A interface visual é dividida em três camadas:

1. **Entrada e controle da partida**: teclado, pausa, histórico, reinício, humano e IA.
2. **Layout visual**: cálculo de posições, ângulos e ramos das peças na mesa.
3. **Desenho**: chamadas OpenGL/Pygame para mesa, peças, HUD, menus e textos.

O motor de dominó continua sendo a fonte das regras do jogo. A UI não decide se
uma jogada é válida; ela consulta o motor e só organiza a interação visual.

## Arquivos principais

### `main_visual.py`

É o ponto de entrada.

Ele cria:

- `MotorDomino`;
- agentes iniciais;
- `GerenciadorPartida`;
- `ControladorPartida`;
- `HudRenderer`;
- janela Pygame/OpenGL.

O laço principal faz sempre a mesma sequência:

1. calcula `dt_ms`;
2. chama `controlador.processar_entrada()`;
3. chama `controlador.atualizar(dt_ms)`;
4. desenha a mesa com `renderizar_cena(...)`;
5. desenha a HUD com `hud.renderizar(...)`;
6. troca o frame com `pygame.display.flip()`.

### `controle_partida.py`

É o orquestrador da UI.

Responsabilidades:

- manter histórico de snapshots para avançar e voltar;
- controlar pausa e velocidade;
- executar turnos automáticos via `GerenciadorPartida`;
- parar quando o jogador da vez é humano;
- abrir e fechar o menu;
- reiniciar a partida;
- guardar notificações temporárias para a HUD.

O controlador herda dois mixins:

- `ControleHumanoMixin`;
- `VisibilidadeMaosMixin`.

Assim, a HUD e os testes continuam usando `ControladorPartida`, mas a lógica
interna fica em arquivos menores.

### `agentes_ui.py`

Centraliza os tipos de agente usados pela interface:

- `neural`;
- `heuristico`;
- `aleatorio`;
- `humano`.

Também fornece:

- `criar_agente_por_tipo(tipo)`;
- `nome_tipo_agente(tipo)`.

O agente humano é uma sentinela. Ele não deve jogar pelo `GerenciadorPartida`,
porque a jogada humana é executada diretamente pelo controlador depois do
teclado.

### `controle_humano.py`

Cuida da interação quando o jogador da vez é humano:

- selecionar peça da mão;
- navegar com esquerda/direita;
- alternar ponta com cima/baixo/Tab;
- calcular se a seta amarela fica acima ou abaixo da peça;
- jogar com Enter;
- comprar com C;
- passar com P.

A regra importante é que a seta amarela aponta para a metade da peça que será
conectada na mesa. Por exemplo, se a ponta selecionada exige o valor 2, a seta
aparece no lado da peça onde está o 2.

### `visibilidade_maos.py`

Define quando cada mão aparece aberta ou oculta:

- IA vs IA: as duas mãos ficam sempre visíveis;
- humano vs IA: a mão humana fica visível e a mão da IA pode ser alternada;
- humano vs humano: só aparece a mão do jogador da vez.

As teclas `J` e `K` tentam alternar a mão de J0 e J1. Quando a regra do modo
não permite alternar, o controlador mostra uma notificação curta.

### `hud.py`

Desenha a camada 2D por cima da mesa:

- barra superior com jogador da vez e turno;
- barra das mãos e monte;
- seta amarela da peça selecionada;
- notificações;
- mensagem de fim de jogo;
- menu de configuração;
- barra inferior de atalhos.

A HUD só lê o controlador e o snapshot atual. Ela não altera o jogo.

### `interface.py`

Desenha a mesa 3D.

O arquivo recebe `cadeia_visual`, escolhe um pivô estável e separa a cadeia em:

- ramo esquerdo;
- pivô;
- ramo direito.

Isso evita que todas as peças andem quando uma peça nova entra em uma ponta.

### `layout_domino.py`

Calcula a geometria das peças na mesa:

- direção dos ramos;
- ângulo de cada peça;
- ordem dos valores desenhados;
- posição em linha;
- posição quando o ramo vira;
- limites da mesa.

Este arquivo não desenha nada. Ele só devolve slots com posição, ângulo e
valores.

### `desenho_domino.py`

Desenha uma peça individual em 3D.

Ele recebe posição, ângulo e valores já calculados pelo layout.

### `primitivas.py`

Contém funções simples de desenho:

- retângulo;
- contorno;
- linha;
- triângulo;
- círculo;
- texto;
- dominó 3D;
- dominó 2D da HUD;
- verso de dominó.

### `renderizador_estado.py`

Guarda a peça escolhida como pivô da mesa.

Quando o histórico muda ou novas peças entram, ele tenta reencontrar o mesmo
pivô para manter a mesa visualmente estável.

### `config_visual.py`

Reúne constantes visuais:

- posição inicial;
- escala da peça;
- espaçamento;
- limites horizontais e verticais da mesa;
- dimensões de peças em pé e deitadas.

### `testes_unitarios_ui.py`

Arquivo executável com testes sequenciais da UI.

Ele testa principalmente:

- avanço automático de IA;
- parada em turno humano;
- seleção e navegação da mão;
- alternância de ponta;
- posição da seta amarela;
- execução de jogada humana;
- velocidade;
- reinício rápido;
- visibilidade das mãos.

Para rodar:

```bash
python ui/testes_unitarios_ui.py
```

No ambiente atual do projeto, normalmente:

```bash
.venv/bin/python ui/testes_unitarios_ui.py
```

## Fluxo de uma jogada automática

1. `main_visual.py` chama `controlador.atualizar(dt_ms)`.
2. O controlador acumula tempo.
3. Quando atinge o intervalo da velocidade atual, chama `avancar()`.
4. Se está na ponta viva do histórico, chama `_jogar_proximo_turno()`.
5. Se o jogador é IA, o `GerenciadorPartida` escolhe e executa a jogada.
6. O controlador captura um novo snapshot.
7. A mesa e a HUD desenham esse snapshot.

## Fluxo de uma jogada humana

1. O controlador detecta que o jogador da vez é humano.
2. O avanço automático para somente naquele turno.
3. A HUD mostra a mão do humano e a seta da peça selecionada.
4. O usuário escolhe peça e ponta.
5. Enter, C ou P executa a ação.
6. O motor valida e aplica a ação.
7. O controlador captura o novo snapshot.
8. Se a próxima vez for IA, o automático volta a andar.

## Atalhos

Modo automático/observação:

- `M`: menu;
- `R`: reiniciar;
- `Espaço`: pausa;
- `Setas`: passo/histórico;
- `+` e `-`: velocidade;
- `J` e `K`: visibilidade das mãos quando permitido;
- `ESC`: sair.

Turno humano:

- `Esquerda/Direita`: seleciona peça;
- `Cima/Baixo/Tab`: alterna ponta quando a peça pode jogar nos dois lados;
- `Enter`: joga a peça;
- `C`: compra;
- `P`: passa;
- `M`: menu;
- `ESC`: sair.
