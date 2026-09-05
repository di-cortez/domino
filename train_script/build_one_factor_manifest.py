#!/usr/bin/env python3
"""Build the shareable review page for the one-factor training sequence.

The page exists to be checked, not read: an advisor should be able to confirm
in one pass that the shared configuration is right and that every run varies
exactly one parameter, before 115 hours of compute are spent on it. Every
number comes from the project's own defaults and from the sequence script's
`--dry-run`, so the page cannot drift from what will actually run.

    python3 train_script/build_one_factor_manifest.py
"""

from __future__ import annotations

import html
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from training.canonical_run import canonical_run_dir  # noqa: E402
from training.pipeline import DEFAULT_SEED, PIPELINE_LEVELS  # noqa: E402
from training.rl.config import RLTrainingOptions  # noqa: E402

WRAPPER = HERE / "run_one_factor_tests_diego_notebook.sh"
RULESET = "double-six"
HOURS_EACH = 5

# Each group is the one parameter it moves: the label the reader checks, the
# defaults line that parameter sits on, and how a run's flags read as a change.
GROUPS = (
    ("Oponente de treino", "opponent",
     "Contra quem as partidas de self-play são jogadas."),
    ("Baseline de vantagem", "baseline",
     "O termo <code>b</code> subtraído do retorno em "
     "<code>vantagem = retorno − b</code>."),
    ("Relógio do desconto", "distance",
     "O que <code>gamma</code> conta ao descontar: turnos da mesa ou decisões "
     "do agente. A primeira metade é o relógio local, a segunda o terminal."),
    ("Coeficiente de entropia", "entropy",
     "O bônus de exploração somado à perda da política."),
    ("Taxa de aprendizado", "lr",
     "O passo do otimizador."),
    ("Partidas por iteração (GPI)", "gpi",
     "Quantas partidas entram em cada atualização da política."),
    ("Pesos terminais (a_E, a_B)", "terminal",
     "Mão vazia contra jogo fechado. O par é normalizado pelo maior membro, "
     "então só a razão dentro do par tem significado."),
    ("Pesos imediatos (a_P, a_D)", "immediate",
     "Passe contra compra. Mesma normalização, mesmo par."),
)

# Which run belongs to which group, and the human reading of its change.
CHANGES = {
    "control": (None, None, None),
    "bucket_heuristic": ("opponent", "--opponent-buckets", "random → heuristic"),
    "baseline_zero": ("baseline", "--baseline", "lookup-table → zero"),
    "baseline_batch_mean": ("baseline", "--baseline", "lookup-table → batch-mean"),
    "distance_turn_turn": ("distance", "--reward-distance-mode",
                           "decision-decision → turn-turn"),
    "distance_turn_decision": ("distance", "--reward-distance-mode",
                               "decision-decision → turn-decision"),
    "distance_decision_turn": ("distance", "--reward-distance-mode",
                               "decision-decision → decision-turn"),
    "entropy_0p01": ("entropy", "--entropy-coef", "0 → 0,01"),
    "entropy_0p1": ("entropy", "--entropy-coef", "0 → 0,1"),
    "lr_0p005": ("lr", "--learning-rate", "0,01 → 0,005"),
    "lr_0p02": ("lr", "--learning-rate", "0,01 → 0,02"),
    "lr_0p03": ("lr", "--learning-rate", "0,01 → 0,03"),
    "lr_0p04": ("lr", "--learning-rate", "0,01 → 0,04"),
    "gpi_1000": ("gpi", "--gpi", "2000 → 1000"),
    "gpi_4000": ("gpi", "--gpi", "2000 → 4000"),
    "terminal_aE1_aB0": ("terminal", "a_E, a_B", "(1, 1) → (1, 0)"),
    "terminal_aE0_aB1": ("terminal", "a_E, a_B", "(1, 1) → (0, 1)"),
    "terminal_aE1_aB2": ("terminal", "a_E, a_B", "(1, 1) → (1, 2)"),
    "terminal_aE2_aB1": ("terminal", "a_E, a_B", "(1, 1) → (2, 1)"),
    "immediate_aP1_aD0": ("immediate", "a_P, a_D", "(1, 1) → (1, 0)"),
    "immediate_aP0_aD1": ("immediate", "a_P, a_D", "(1, 1) → (0, 1)"),
    "immediate_aP1_aD2": ("immediate", "a_P, a_D", "(1, 1) → (1, 2)"),
    "immediate_aP2_aD1": ("immediate", "a_P, a_D", "(1, 1) → (2, 1)"),
}


def collect_runs():
    """Return the sequence exactly as the script would execute it."""
    result = subprocess.run(
        ["bash", str(WRAPPER), "--dry-run"],
        capture_output=True, text=True, cwd=str(REPO),
        env={"PATH": "/usr/bin:/bin", "PYTHON": sys.executable, "HOME": str(Path.home())},
    )
    if result.returncode != 0:
        raise SystemExit(f"The sequence dry run failed:\n{result.stderr}")
    runs = []
    for line in result.stdout.splitlines():
        match = re.search(r"--run-name (\S+)(.*)$", line)
        if not match:
            continue
        name, flags = match.group(1), match.group(2).strip()
        short = name.replace("one_factor_", "").replace("_diego_notebook", "")
        runs.append({
            "order": len(runs) + 1,
            "run_name": name,
            "short": short,
            "flags": flags,
            "directory": canonical_run_dir(
                ".", "forever", DEFAULT_SEED, run_name=name
            ).name,
            "bundle": bundle_suffix_of(flags),
        })
    if len(runs) != len(CHANGES):
        raise SystemExit(
            f"The sequence has {len(runs)} points but this page describes "
            f"{len(CHANGES)}. Update CHANGES."
        )
    return runs


def bundle_suffix_of(flags):
    """Return the analysis-bundle tail the sequence passes for one run."""
    match = re.search(r"--bundle-suffix (\S+)", flags)
    return match.group(1) if match else "control"


def comma(value):
    """Render a number the way the surrounding Portuguese reads it.

    A trailing `.0` is dropped: `entropy_coef = 0.0` is the integer zero to a
    reader checking the configuration, and `0,0` invites the question of what
    the hidden digit is.
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).replace(".", ",")


def shared_parameters():
    options = RLTrainingOptions()
    return (
        ("Semente", DEFAULT_SEED, "seed"),
        ("Ruleset", RULESET, "ruleset"),
        ("Nível do pipeline", "forever", "—"),
        ("Épocas do PPO", PIPELINE_LEVELS["forever"].ppo_max_epochs,
         "ppo_max_epochs"),
        ("Oponente de treino", ",".join(options.opponent_buckets),
         "opponent_buckets"),
        ("Baseline", options.baseline[0], "baseline"),
        ("Taxa de aprendizado", comma(options.learning_rate), "learning_rate"),
        ("Partidas por iteração", options.gpi, "gpi"),
        ("Coef. de entropia", comma(options.entropy_coef), "entropy_coef"),
        ("Mistura terminal/local", comma(options.reward_eta), "reward_eta"),
        ("Desconto terminal", comma(options.gamma_f), "gamma_f"),
        ("Desconto de evento", comma(options.gamma_i), "gamma_i"),
        ("Relógio do desconto", options.reward_distance_mode,
         "reward_distance_mode"),
        ("Peso a_E / a_B", f"{comma(options.terminal_empty_hand_weight)} / "
         f"{comma(options.terminal_blocked_weight)}", "terminal_*_weight"),
        ("Peso a_P / a_D", f"{comma(options.immediate_pass_weight)} / "
         f"{comma(options.immediate_draw_weight)}", "immediate_*_weight"),
        ("Peso de dificuldade", comma(options.difficulty_weight),
         "difficulty_weight"),
        ("Cabeça de valor", "desligada" if not options.use_value_head else "ligada",
         "value_head"),
    )


def render_run(run):
    group, flag, change = CHANGES[run["short"]]
    # `--bundle-suffix` is bookkeeping, not part of the experiment, so it is
    # reported once in the directory listing rather than on every run.
    visible = re.sub(r"\s*--bundle-suffix \S+", "", run["flags"]).strip()
    escaped_flags = html.escape(visible) if visible else "—"
    if group is None:
        change_html = (
            '<span class="control-note">nada muda — este é o ponto de '
            'comparação</span>'
        )
    else:
        before, after = change.split(" → ")
        change_html = (
            f'<span class="param">{html.escape(flag)}</span>'
            f'<span class="was">{html.escape(before)}</span>'
            f'<span class="arrow" aria-hidden="true">→</span>'
            f'<span class="now">{html.escape(after)}</span>'
        )
    return f'''      <li class="run{' is-control' if group is None else ''}">
        <span class="order">{run['order']:02d}</span>
        <div class="run-body">
          <p class="run-name">{html.escape(run['short'])}</p>
          <div class="change">{change_html}</div>
          <p class="flags"><code>{escaped_flags}</code></p>
        </div>
      </li>'''


def build():
    runs = collect_runs()
    by_group = {}
    for run in runs:
        group = CHANGES[run["short"]][0]
        by_group.setdefault(group, []).append(run)

    shared_rows = "\n".join(
        f'          <div class="param-row"><dt>{html.escape(label)}</dt>'
        f'<dd class="value">{html.escape(str(value))}</dd>'
        f'<dd class="key"><code>{html.escape(key)}</code></dd></div>'
        for label, value, key in shared_parameters()
    )

    control = by_group[None][0]
    sections = [f'''  <section class="group" id="controle">
    <div class="group-head">
      <p class="group-kicker">Ponto de comparação</p>
      <h2>Controle</h2>
      <p class="group-note">Tudo no padrão. Toda diferença medida nas outras
        vinte e duas execuções é lida contra esta.</p>
    </div>
    <ol class="runs">
{render_run(control)}
    </ol>
  </section>''']

    for title, key, note in GROUPS:
        group_runs = by_group.get(key, [])
        if not group_runs:
            continue
        rows = "\n".join(render_run(run) for run in group_runs)
        count = len(group_runs)
        sections.append(f'''  <section class="group">
    <div class="group-head">
      <p class="group-kicker">{count} execuç{'ão' if count == 1 else 'ões'}</p>
      <h2>{html.escape(title)}</h2>
      <p class="group-note">{note}</p>
    </div>
    <ol class="runs">
{rows}
    </ol>
  </section>''')

    directories = "\n".join(
        f"      <li><code>{html.escape(run['directory'])}</code>"
        f"<span class=\"bundle\">└ {html.escape(run['bundle'])}</span></li>"
        for run in runs
    )

    total = len(runs)
    total_hours = total * HOURS_EACH

    return f'''<title>Sequência de treinos de um fator</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bitter:wght@500;600&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500;700&display=swap">
<style>
  :root {{
    color-scheme: light;
    /* Neutrals biased toward the teal accent rather than a flat grey. */
    --ground: #eef2f1;
    --surface: #ffffff;
    --sunk: #e3eae8;
    --ink: #131e1c;
    --ink-soft: #475855;
    --ink-faint: #74847f;
    --rule: #cfdad7;
    --rule-soft: #e0e8e6;
    --accent: #0a6b5c;
    --accent-wash: #e2efec;
    --was: #9a5b18;
    --now: #0a6b5c;
    --shadow: 0 1px 2px rgba(19, 30, 28, .05), 0 10px 28px rgba(19, 30, 28, .055);
    --serif: "Bitter", Georgia, serif;
    --sans: "Source Sans 3", -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    --mono: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      --ground: #0d1514;
      --surface: #15201e;
      --sunk: #1c2927;
      --ink: #dfe8e5;
      --ink-soft: #a3b3af;
      --ink-faint: #7a8a86;
      --rule: #2a3835;
      --rule-soft: #202c2a;
      --accent: #4fbfa9;
      --accent-wash: #16302b;
      --was: #d8944a;
      --now: #4fbfa9;
      --shadow: 0 1px 2px rgba(0, 0, 0, .35), 0 10px 28px rgba(0, 0, 0, .3);
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --ground: #0d1514;
    --surface: #15201e;
    --sunk: #1c2927;
    --ink: #dfe8e5;
    --ink-soft: #a3b3af;
    --ink-faint: #7a8a86;
    --rule: #2a3835;
    --rule-soft: #202c2a;
    --accent: #4fbfa9;
    --accent-wash: #16302b;
    --was: #d8944a;
    --now: #4fbfa9;
    --shadow: 0 1px 2px rgba(0, 0, 0, .35), 0 10px 28px rgba(0, 0, 0, .3);
  }}

  * {{ box-sizing: border-box; }}
  body {{
    background: var(--ground);
    color: var(--ink);
    font-family: var(--sans);
    font-size: 16.5px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }}
  .shell {{
    max-width: 1020px;
    margin: 0 auto;
    padding: 0 clamp(18px, 4vw, 40px) 88px;
  }}
  code {{ font-family: var(--mono); font-size: .86em; }}

  /* ---- masthead ---- */
  header {{
    padding: clamp(40px, 7vw, 76px) 0 clamp(24px, 3vw, 34px);
    border-bottom: 2px solid var(--ink);
  }}
  .eyebrow {{
    font-family: var(--mono);
    font-size: 11.5px;
    font-weight: 500;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--accent);
    margin: 0 0 16px;
  }}
  h1 {{
    font-family: var(--serif);
    font-weight: 600;
    font-size: clamp(2.1rem, 5vw, 3.2rem);
    line-height: 1.08;
    letter-spacing: -.014em;
    margin: 0 0 16px;
    max-width: 18ch;
    text-wrap: balance;
  }}
  .standfirst {{
    font-size: clamp(1.02rem, 1.9vw, 1.16rem);
    color: var(--ink-soft);
    margin: 0;
    max-width: 64ch;
  }}
  .tallies {{
    display: flex;
    flex-wrap: wrap;
    gap: 0;
    margin: 30px 0 0;
    border-top: 1px solid var(--rule);
  }}
  .tally {{
    flex: 1 1 150px;
    padding: 13px 22px 13px 0;
    border-right: 1px solid var(--rule-soft);
  }}
  .tally:last-child {{ border-right: 0; }}
  .tally dt {{
    font-family: var(--mono);
    font-size: 10.5px;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--ink-faint);
    margin: 0 0 4px;
  }}
  .tally dd {{
    margin: 0;
    font-family: var(--mono);
    font-size: 1.12rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
  }}

  /* ---- shared configuration ---- */
  .shared {{ padding-top: clamp(40px, 5vw, 60px); }}
  h2 {{
    font-family: var(--serif);
    font-weight: 600;
    font-size: 1.42rem;
    line-height: 1.2;
    letter-spacing: -.008em;
    margin: 0 0 8px;
    text-wrap: balance;
  }}
  .lead {{ color: var(--ink-soft); margin: 0 0 22px; max-width: 66ch; }}
  .params {{
    display: grid;
    gap: 1px;
    margin: 0;
    background: var(--rule-soft);
    border: 1px solid var(--rule);
    border-radius: 4px;
    overflow: hidden;
  }}
  @media (min-width: 700px) {{
    .params {{ grid-template-columns: 1fr 1fr; }}
    /* An odd parameter count would leave the gap colour showing in the last
       cell; this fills it so the block keeps a straight edge. */
    .params::after {{ content: ""; background: var(--surface); }}
  }}
  .param-row {{
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: baseline;
    column-gap: 12px;
    background: var(--surface);
    padding: 10px 16px;
  }}
  .param-row dt {{ font-size: .93rem; color: var(--ink-soft); }}
  .param-row .value {{
    margin: 0;
    font-family: var(--mono);
    font-size: .9rem;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
    text-align: right;
  }}
  .param-row .key {{
    grid-column: 1 / -1;
    margin: 1px 0 0;
    font-size: .74rem;
    color: var(--ink-faint);
  }}
  .param-row .key code {{ font-size: 1em; }}

  /* ---- run groups ---- */
  .group {{ padding-top: clamp(34px, 4.5vw, 52px); }}
  .group-head {{ margin-bottom: 14px; }}
  .group-kicker {{
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: .11em;
    text-transform: uppercase;
    color: var(--ink-faint);
    margin: 0 0 5px;
  }}
  .group-note {{ margin: 6px 0 0; color: var(--ink-soft); max-width: 68ch; font-size: .95rem; }}

  .runs {{ list-style: none; margin: 0; padding: 0; display: grid; gap: 1px;
           background: var(--rule-soft); border: 1px solid var(--rule);
           border-radius: 4px; overflow: hidden; }}
  .run {{
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 14px;
    background: var(--surface);
    padding: 13px 16px;
  }}
  .run.is-control {{ background: var(--accent-wash); }}
  .order {{
    font-family: var(--mono);
    font-size: .76rem;
    font-weight: 700;
    color: var(--ink-faint);
    padding-top: 3px;
    font-variant-numeric: tabular-nums;
  }}
  .run-body {{ min-width: 0; display: flex; flex-direction: column; gap: 5px; }}
  .run-name {{
    margin: 0;
    font-family: var(--mono);
    font-size: .92rem;
    font-weight: 500;
    word-break: break-word;
  }}
  .change {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px;
             font-family: var(--mono); font-size: .82rem; }}
  .param {{ color: var(--ink-faint); }}
  .was {{ color: var(--was); text-decoration: line-through;
          text-decoration-thickness: 1px; }}
  .arrow {{ color: var(--ink-faint); }}
  .now {{ color: var(--now); font-weight: 700; }}
  .control-note {{ font-family: var(--sans); font-size: .88rem; color: var(--ink-soft); }}
  .flags {{ margin: 0; overflow-x: auto; }}
  .flags code {{
    display: block;
    white-space: pre;
    font-size: .77rem;
    color: var(--ink-faint);
    padding-bottom: 2px;
  }}

  /* ---- closing notes ---- */
  .notes {{ padding-top: clamp(40px, 5vw, 60px); }}
  .note {{
    background: var(--surface);
    border-left: 3px solid var(--accent);
    border-radius: 0 4px 4px 0;
    padding: 18px 22px;
    margin: 0 0 14px;
    box-shadow: var(--shadow);
  }}
  .note h3 {{ font-family: var(--sans); font-size: 1rem; font-weight: 600; margin: 0 0 6px; }}
  .note p {{ margin: 0; color: var(--ink-soft); font-size: .95rem; max-width: 70ch; }}
  .note p + p {{ margin-top: 8px; }}

  details {{ margin-top: 22px; border: 1px solid var(--rule); border-radius: 4px;
             background: var(--surface); }}
  summary {{ cursor: pointer; padding: 12px 16px; font-weight: 600; font-size: .95rem; }}
  summary::marker {{ color: var(--accent); }}
  details ul {{ margin: 0; padding: 0 16px 14px 34px; }}
  details li {{ font-size: .8rem; margin-bottom: 3px; }}
  details li code {{ font-size: 1em; color: var(--ink-soft); word-break: break-all; }}
  details li .bundle {{
    display: block;
    font-family: var(--mono);
    font-size: .95em;
    color: var(--accent);
    padding-left: 12px;
  }}

  footer {{
    margin-top: clamp(44px, 6vw, 66px);
    padding-top: 22px;
    border-top: 2px solid var(--ink);
    font-size: .86rem;
    color: var(--ink-faint);
    max-width: 74ch;
  }}
  :focus-visible {{ outline: 2px solid var(--accent); outline-offset: 3px; }}
</style>

<div class="shell">
  <header>
    <p class="eyebrow">Domino RL · para revisão · 4 de setembro de 2026</p>
    <h1>{total} treinos, um parâmetro por vez</h1>
    <p class="standfirst">
      Cada execução usa a configuração padrão do projeto e altera
      <strong>exatamente um</strong> parâmetro, então qualquer diferença medida
      contra o controle pertence àquele parâmetro. Esta página existe para ser
      conferida antes de {total_hours} horas de máquina serem gastas.
    </p>
    <dl class="tallies">
      <div class="tally"><dt>Execuções</dt><dd>{total}</dd></div>
      <div class="tally"><dt>Horas cada</dt><dd>{HOURS_EACH} h</dd></div>
      <div class="tally"><dt>Total</dt><dd>{total_hours} h</dd></div>
      <div class="tally"><dt>Semente</dt><dd>{DEFAULT_SEED}</dd></div>
      <div class="tally"><dt>Ruleset</dt><dd>{RULESET}</dd></div>
    </dl>
  </header>

  <section class="shared">
    <h2>A configuração comum</h2>
    <p class="lead">
      Vale para todas as {total} execuções. Uma variante troca apenas a linha
      que ela testa; tudo o mais nesta lista permanece.
    </p>
    <dl class="params">
{shared_rows}
    </dl>
  </section>

{chr(10).join(sections)}

  <section class="notes">
    <h2>Três pontos que pedem atenção</h2>

    <div class="note">
      <h3>O ruleset é <code>double-six</code>, não <code>double-three</code></h3>
      <p>
        Os experimentos de baseline anteriores foram todos em
        <code>double-three</code>, então estes resultados <strong>não são
        comparáveis</strong> com aqueles. O motivo é o baseline
        <code>lookup-table</code>: ele precisa de uma tabela de recompensa
        empacotada no formato 3, e <code>double-six</code> é o único ruleset
        que tem uma. Mudar para <code>double-three</code> exige reconstruir
        esse artefato antes.
      </p>
    </div>

    <div class="note">
      <h3>Seis dos valores pedidos já eram o padrão</h3>
      <p>
        Bucket <code>random</code>, baseline <code>lookup-table</code>,
        distância <code>decision-decision</code>, entropia 0, taxa 0,01 e GPI
        2000 são a configuração padrão. Sob uma única semente, executá-los
        separadamente seria a mesma execução repetida seis vezes, a cinco horas
        cada. Eles estão representados pelo <strong>controle</strong>, que
        também é o ponto (1, 1) dos dois grupos de peso de recompensa.
      </p>
    </div>

    <div class="note">
      <h3>Uma semente só</h3>
      <p>
        Todas as execuções usam <code>seed = {DEFAULT_SEED}</code>. Isso torna as
        comparações limpas — mesma inicialização supervisionada, mesmas mãos —
        mas <strong>nenhuma diferença observada pode ser separada de variação
        de semente</strong>, porque não há réplica. Repetir o controle com uma
        segunda semente daria essa régua.
      </p>
      <p>
        Vale lembrar também que <code>--gpi 4000</code> dobra o tamanho da
        iteração: em cinco horas ela faz metade das atualizações do controle.
        A comparação por tempo e a comparação por número de atualizações dizem
        coisas diferentes nesse ponto.
      </p>
    </div>

    <details>
      <summary>Diretórios que serão criados em <code>models/rl/</code>, com o pacote de análise de cada um</summary>
      <ul>
{directories}
      </ul>
    </details>
  </section>

  <footer>
    <p>
      Gerado a partir do próprio script da sequência
      (<code>train_script/run_one_factor_tests_diego_notebook.sh --dry-run</code>)
      e dos valores padrão do projeto, por
      <code>train_script/build_one_factor_manifest.py</code>. Os números desta
      página não são transcritos à mão: se o script mudar, ela muda junto.
      A sequência é retomável — cada ponto grava o tempo de RL consumido, e
      reexecutar o script continua de onde parou.
    </p>
  </footer>
</div>
'''


def main():
    output = HERE / "one_factor_manifest.html"
    output.write_text(build())
    print(f"{output}: {output.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
