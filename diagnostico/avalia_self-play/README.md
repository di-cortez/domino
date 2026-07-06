# Avaliação: self-play puro vs. treino direto contra o heurístico

Compara dois checkpoints de RL treinados por regimes diferentes de
`training/self_play.py` (parâmetro `oponente_treino`):

- **`self_play`** — self-play puro contra um pool de snapshots congelados de
  iterações passadas de si mesma (padrão atual; ver
  `references/fundamentos_rl.pdf`, seção "Self-play e o paralelo com o
  AlphaGo").
- **`heuristico`** — toda partida de treino é direto contra o
  `AgenteEstrategico` fixo (sem self-play).

Existe para responder uma pergunta concreta: **o self-play puro produz uma
política melhor do que treinar direto contra o professor heurístico?** Os
testes já rodados (1.000 e 5.000 iterações, mesmo warm-start de SL) não
encontraram diferença estatisticamente significativa entre os dois regimes
— ambos ficam em ~47–50% de vitórias vs. o heurístico. Este script existe
para reproduzir essa comparação (ou repeti-la com orçamentos de treino
maiores) e gerar os artefatos (CSV/JSON/PNGs) prontos para apresentação.

## 1. Gerar os dois checkpoints

A partir da raiz do repositório:

```bash
python -c "from training.self_play import treinar; \
    treinar(oponente_treino='self_play', caminho_pesos_rl='models/pesos_domino_rl_self_play.npz')"

python -c "from training.self_play import treinar; \
    treinar(oponente_treino='heuristico', caminho_pesos_rl='models/pesos_domino_rl_heuristico.npz')"
```

Ambas as chamadas fazem warm-start a partir do mesmo
`models/pesos_domino_sl.npz` (o `treinar()` só faz warm-start quando o
`caminho_pesos_rl` indicado ainda não existe) — importante para a
comparação ser justa. Ajuste `iteracoes`, `partidas_por_iteracao` etc.
conforme necessário; veja `training/README.md`.

## 2. Rodar a comparação

```bash
python "diagnostico/avalia_self-play/comparar_regimes.py"
```

(Precisa ser executado como script direto — `python arquivo.py`, não
`python -m pacote.modulo` — porque o nome desta pasta tem um hífen, que não
é um identificador Python válido.)

Opções (`--help` lista todas): `--pesos-self-play`, `--pesos-heuristico`,
`-n/--partidas` (padrão 1000), `--seed`, `--saida`.

## Saídas

Em `diagnostico/resultados/self_play_vs_heuristico_regimes/` (ou
`--saida`), três subpastas — uma por confronto — cada uma com
`resumo.json`, `partidas.csv` e os quatro PNGs de sempre (ver
`diagnostico/README.md`):

- `self_play_vs_heuristico/` — checkpoint self-play vs. `AgenteEstrategico`
- `heuristico_vs_heuristico/` — checkpoint treinado-vs-heurístico vs. `AgenteEstrategico`
- `self_play_vs_heuristico_direto/` — os dois checkpoints entre si

No console, o script também imprime uma conclusão em texto simples: se os
IC 95% de vitória contra o heurístico se sobrepõem entre os dois regimes
(sem diferença estatística) ou não, e se o confronto direto exclui os 50%
(um regime realmente supera o outro) ou não.

## Como interpretar

Veja também "Como interpretar" em `diagnostico/README.md` — os mesmos
princípios (IC 95%, cuidado com `n` pequeno) valem aqui. Em particular: se
o confronto direto entre os dois checkpoints ficar por volta de 50% com um
IC amplo cobrindo esse valor, isso **não** significa que o self-play "não
funciona" — pode só significar que o orçamento de treino usado para gerar
os checkpoints ainda é pequeno demais para separar os dois regimes (é
exatamente o que observamos em 1.000–5.000 iterações). Rodar com mais
iterações antes de comparar é o próximo passo natural, não uma mudança
neste script.
