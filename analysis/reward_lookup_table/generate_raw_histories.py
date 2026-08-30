"""Generate deterministic full neural-vs-heuristic histories for every ruleset."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import sys
import time


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from reward_lookup_common import (
    RAW_FORMAT,
    RAW_FORMAT_VERSION,
    action_kind,
    atomic_write_json,
    canonical_json_sha256,
    decision_class,
    discover_checkpoints,
    file_sha256,
    local_event_for_action,
    parse_ruleset_selection,
    raw_reward_schema,
    serialize_action,
    terminal_reward_for_neural,
    tile_option_count,
    write_deterministic_gzip_lines,
)


DEFAULT_GAMES = 100_000
DEFAULT_WORKERS = 10
DEFAULT_CHUNK_GAMES = 1_000
DEFAULT_JOB_GAMES = 16
DEFAULT_BASE_SEED = 20_260_829
DEFAULT_RAW_ROOT = SCRIPT_DIR / "raw"
MANIFEST_NAME = "manifest.json"

_WORKER_RULESET = None
_WORKER_SEED_PLAN = None
_WORKER_NEURAL = None
_WORKER_HEURISTIC = None
_WORKER_REWARD_SCHEMA = None


def _force_cpu_environment():
    """Keep analysis workers away from CUDA and nested BLAS thread pools."""
    os.environ["DOMINO_FORCE_CPU"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = "1"


@contextlib.contextmanager
def _cpu_only_children():
    """Temporarily set the environment inherited by spawned workers."""
    names = (
        "DOMINO_FORCE_CPU",
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    )
    previous = {name: os.environ.get(name) for name in names}
    _force_cpu_environment()
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _worker_initializer(weights_path, ruleset_name, base_seed):
    """Load one immutable CPU policy and fresh stateful agents per worker."""
    global _WORKER_RULESET, _WORKER_SEED_PLAN
    global _WORKER_NEURAL, _WORKER_HEURISTIC, _WORKER_REWARD_SCHEMA

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _force_cpu_environment()

    from agents.heuristic_agent import StrategicAgent
    from agents.rl_agent import RLAgent
    from agents.rl_nn import PolicyNetwork
    from middleware.rulesets import resolve_ruleset
    from utils.myrandom import SeedPlan

    _WORKER_RULESET = resolve_ruleset(ruleset_name)
    _WORKER_SEED_PLAN = SeedPlan(root_seed=int(base_seed))
    network = PolicyNetwork.load(
        str(weights_path),
        use_value_head=False,
        device="cpu",
    )
    _WORKER_NEURAL = RLAgent(
        network,
        mode="evaluation",
        ruleset=_WORKER_RULESET,
        use_opponent_suit_features=True,
        use_opponent_bucket_features=False,
    )
    _WORKER_HEURISTIC = StrategicAgent(ruleset=_WORKER_RULESET)
    _WORKER_REWARD_SCHEMA = raw_reward_schema()


def _compact_position(engine):
    """Return the nonredundant public counts needed by the lookup analysis."""
    return {
        "hand_sizes": [int(len(hand)) for hand in engine.hands],
        "stock_size": int(len(engine.stock)),
        "ends": [int(value) for value in engine.ends],
    }


def _play_game(game_index):
    """Play and serialize one complete, event-sourced game history."""
    from middleware.domino_engine import DominoEngine
    from utils.myrandom import RandomNamespace

    if any(
        value is None
        for value in (
            _WORKER_RULESET,
            _WORKER_SEED_PLAN,
            _WORKER_NEURAL,
            _WORKER_HEURISTIC,
            _WORKER_REWARD_SCHEMA,
        )
    ):
        raise RuntimeError("Reward lookup worker was not initialized")

    game_index = int(game_index)
    game_rng = _WORKER_SEED_PLAN.generator(
        RandomNamespace.DIAGNOSTIC_GAME,
        _WORKER_RULESET.name,
        game_index,
    )
    engine = DominoEngine(
        player_count=2,
        rng=game_rng,
        ruleset=_WORKER_RULESET,
    )
    engine.game_id = game_index + 1
    neural_seat = game_index % 2
    agents = [None, None]
    agents[neural_seat] = _WORKER_NEURAL
    agents[1 - neural_seat] = _WORKER_HEURISTIC

    initial_state = engine.to_dict()
    turns = []
    neural_decision_count = 0
    local_event_sum = 0.0

    while not engine.game_over:
        state = engine._get_state()
        acting_seat = int(state["current_player"])
        actor = "neural" if acting_seat == neural_seat else "heuristic"
        legal_actions = engine.valid_actions(acting_seat)
        option_count = tile_option_count(legal_actions)
        is_neural_decision = (
            acting_seat == neural_seat and option_count >= 2
        )
        decision_index = neural_decision_count if is_neural_decision else None
        decisions_before = neural_decision_count
        pre_position = _compact_position(engine)
        drawn_tile = (
            [int(value) for value in engine.stock[0]]
            if legal_actions == [("DRAW", None)] and engine.stock
            else None
        )

        chosen_action = agents[acting_seat].choose_move(state, legal_actions)
        if is_neural_decision:
            neural_decision_count += 1
        event = local_event_for_action(
            chosen_action,
            acting_seat,
            neural_seat,
            _WORKER_REWARD_SCHEMA,
        )
        if event is not None:
            local_event_sum += float(event["base_reward"])

        turn_index = int(engine.turn)
        engine.step(
            chosen_action,
            return_state=False,
            legal_actions=legal_actions,
        )
        turns.append({
            "turn": turn_index,
            "acting_seat": acting_seat,
            "actor": actor,
            "pre": pre_position,
            "legal_actions": [
                serialize_action(action) for action in legal_actions
            ],
            "tile_option_count": int(option_count),
            "decision_class": decision_class(chosen_action, option_count),
            "neural_decisions_before": int(decisions_before),
            "neural_decision_index": decision_index,
            "action": serialize_action(chosen_action),
            "action_kind": action_kind(chosen_action),
            "drawn_tile": drawn_tile,
            "local_event_for_neural": event,
            "post": {
                **_compact_position(engine),
                "game_over": bool(engine.game_over),
                "winner": engine.winner,
            },
        })

    final_state = engine.to_dict()
    terminal = terminal_reward_for_neural(
        final_state,
        neural_seat,
        _WORKER_REWARD_SCHEMA,
    )
    neural_won = int(final_state["winner"]) == neural_seat
    record = {
        "game": game_index + 1,
        "game_index": game_index,
        "neural_seat": int(neural_seat),
        "result": "win" if neural_won else "loss",
        "initial_state": initial_state,
        "turns": turns,
        "final_state": final_state,
        "neural_real_decisions": int(neural_decision_count),
        "raw_rewards_for_neural": {
            "local_event_sum": float(local_event_sum),
            "terminal": terminal,
        },
    }
    line = json.dumps(
        record,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return {
        "game_index": game_index,
        "line": line,
        "turns": len(turns),
        "neural_real_decisions": int(neural_decision_count),
        "neural_win": int(neural_won),
        "local_events": sum(
            turn["local_event_for_neural"] is not None for turn in turns
        ),
    }


def _play_block(game_indices):
    """Return one ordered block so each worker transfer remains bounded."""
    return [_play_game(game_index) for game_index in game_indices]


def _chunk_path(ruleset_dir, chunk_index, start_game, stop_game):
    """Return the stable filename for one half-open game interval."""
    return ruleset_dir / (
        f"part-{chunk_index:05d}-games-{start_game:06d}-{stop_game - 1:06d}"
        ".jsonl.gz"
    )


def _configuration(args, ruleset_name, checkpoint):
    """Return everything that can affect decompressed raw bytes."""
    return {
        "format": RAW_FORMAT,
        "format_version": RAW_FORMAT_VERSION,
        "ruleset_name": ruleset_name,
        "matchup": "neural_vs_heuristic",
        "games": int(args.games),
        "chunk_games": int(args.chunk_games),
        "base_seed": int(args.base_seed),
        "random_stream": {
            "bit_generator": "PCG64",
            "derivation": "blake2b-128-namespace-coordinates-v1",
            "namespace": "diagnostic.game",
            "coordinates": ["ruleset_name", "zero_based_game_index"],
        },
        "seat_policy": "neural seat is zero_based_game_index modulo 2",
        "policy_mode": "deterministic_evaluation",
        "history_encoding": (
            "full initial state plus every legal-action set, chosen action, "
            "drawn tile, compact pre/post position, and full final state"
        ),
        "reward_schema": raw_reward_schema(),
        "neural_policy": {
            key: value
            for key, value in checkpoint.items()
            if key != "path"
        },
    }


def _new_manifest(configuration):
    return {
        "configuration": configuration,
        "configuration_sha256": canonical_json_sha256(configuration),
        "status": "partial",
        "chunks": [],
        "summary": {
            "games": 0,
            "turns": 0,
            "neural_real_decisions": 0,
            "neural_wins": 0,
            "neural_losses": 0,
            "local_events": 0,
        },
    }


def _load_or_create_manifest(ruleset_dir, configuration):
    manifest_path = ruleset_dir / MANIFEST_NAME
    expected_hash = canonical_json_sha256(configuration)
    if not manifest_path.exists():
        unexpected = list(ruleset_dir.glob("part-*.jsonl.gz"))
        if unexpected:
            raise ValueError(
                f"Raw chunks exist without {manifest_path}; move or remove "
                "that incomplete directory explicitly."
            )
        return _new_manifest(configuration)
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("configuration_sha256") != expected_hash:
        raise ValueError(
            f"Existing raw configuration differs in {ruleset_dir}. "
            "Use --force only if replacing it is intentional."
        )
    if manifest.get("configuration") != configuration:
        raise ValueError(f"Configuration hash collision in {manifest_path}")
    _validate_existing_chunks(ruleset_dir, manifest)
    return manifest


def _validate_existing_chunks(ruleset_dir, manifest):
    """Validate every retained atomic chunk before resuming."""
    expected_start = 0
    referenced = set()
    summary = {
        "games": 0,
        "turns": 0,
        "neural_real_decisions": 0,
        "neural_wins": 0,
        "neural_losses": 0,
        "local_events": 0,
    }
    for chunk in manifest.get("chunks", []):
        if int(chunk["start_game_index"]) != expected_start:
            raise ValueError("Raw chunk intervals are not contiguous")
        expected_start = int(chunk["stop_game_index"])
        path = ruleset_dir / chunk["file"]
        referenced.add(path.name)
        if not path.is_file():
            raise FileNotFoundError(f"Missing raw chunk: {path}")
        if file_sha256(path) != chunk["compressed_sha256"]:
            raise ValueError(f"Raw chunk checksum mismatch: {path}")
        for key in summary:
            summary[key] += int(chunk[key])
    if manifest.get("summary") != summary:
        raise ValueError("Raw manifest summary does not match its chunks")
    unreferenced = {
        path.name for path in ruleset_dir.glob("part-*.jsonl.gz")
    } - referenced
    if unreferenced:
        raise ValueError(
            f"Unreferenced raw chunks in {ruleset_dir}: {sorted(unreferenced)}"
        )


def _job_blocks(start_game, stop_game, job_games):
    return [
        tuple(range(block_start, min(block_start + job_games, stop_game)))
        for block_start in range(start_game, stop_game, job_games)
    ]


def _generate_chunk(executor, start_game, stop_game, job_games):
    futures = [
        executor.submit(_play_block, block)
        for block in _job_blocks(start_game, stop_game, job_games)
    ]
    records = []
    for future in as_completed(futures):
        records.extend(future.result())
    records.sort(key=lambda item: item["game_index"])
    expected = list(range(start_game, stop_game))
    actual = [item["game_index"] for item in records]
    if actual != expected:
        raise AssertionError("Parallel raw results lost or reordered games")
    return records


def _chunk_manifest(path, start_game, stop_game, records, gzip_metadata):
    wins = sum(item["neural_win"] for item in records)
    return {
        "file": path.name,
        "start_game_index": int(start_game),
        "stop_game_index": int(stop_game),
        "games": len(records),
        "turns": sum(item["turns"] for item in records),
        "neural_real_decisions": sum(
            item["neural_real_decisions"] for item in records
        ),
        "neural_wins": int(wins),
        "neural_losses": int(len(records) - wins),
        "local_events": sum(item["local_events"] for item in records),
        **gzip_metadata,
    }


def _append_chunk(manifest, chunk):
    manifest["chunks"].append(chunk)
    for key in manifest["summary"]:
        manifest["summary"][key] += int(chunk[key])


def _progress(total, initial, description):
    try:
        from tqdm import tqdm

        return tqdm(
            total=total,
            initial=initial,
            unit="game",
            desc=description,
            dynamic_ncols=True,
        )
    except ImportError:
        class PlainProgress:
            def update(self, amount):
                del amount

            def set_postfix(self, **values):
                del values

            def close(self):
                return None

        return PlainProgress()


def generate_ruleset(args, ruleset_name, checkpoint):
    """Generate or resume every atomic chunk for one ruleset."""
    ruleset_dir = args.raw_root / ruleset_name
    if args.force and ruleset_dir.exists():
        shutil.rmtree(ruleset_dir)
    ruleset_dir.mkdir(parents=True, exist_ok=True)
    configuration = _configuration(args, ruleset_name, checkpoint)
    manifest = _load_or_create_manifest(ruleset_dir, configuration)
    if manifest["status"] == "complete":
        print(f"{ruleset_name}: raw already complete; validated and skipped.")
        return manifest

    completed = int(manifest["summary"]["games"])
    if completed >= args.games:
        manifest["status"] = "complete"
        atomic_write_json(ruleset_dir / MANIFEST_NAME, manifest)
        return manifest

    progress = _progress(args.games, completed, ruleset_name)
    started = time.perf_counter()
    context = mp.get_context("spawn")
    try:
        with _cpu_only_children():
            with ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=context,
                initializer=_worker_initializer,
                initargs=(
                    str(checkpoint["path"]),
                    ruleset_name,
                    args.base_seed,
                ),
            ) as executor:
                for start_game in range(completed, args.games, args.chunk_games):
                    stop_game = min(start_game + args.chunk_games, args.games)
                    chunk_index = start_game // args.chunk_games
                    path = _chunk_path(
                        ruleset_dir,
                        chunk_index,
                        start_game,
                        stop_game,
                    )
                    records = _generate_chunk(
                        executor,
                        start_game,
                        stop_game,
                        args.job_games,
                    )
                    gzip_metadata = write_deterministic_gzip_lines(
                        path,
                        (item["line"] for item in records),
                        compresslevel=args.compresslevel,
                    )
                    chunk = _chunk_manifest(
                        path,
                        start_game,
                        stop_game,
                        records,
                        gzip_metadata,
                    )
                    _append_chunk(manifest, chunk)
                    manifest["status"] = (
                        "complete"
                        if manifest["summary"]["games"] == args.games
                        else "partial"
                    )
                    atomic_write_json(ruleset_dir / MANIFEST_NAME, manifest)
                    progress.update(len(records))
                    games = manifest["summary"]["games"]
                    wins = manifest["summary"]["neural_wins"]
                    progress.set_postfix(
                        decisions=manifest["summary"]["neural_real_decisions"],
                        win=f"{100.0 * wins / games:.2f}%",
                    )
    finally:
        progress.close()
    elapsed = time.perf_counter() - started
    print(
        f"{ruleset_name}: {manifest['summary']['games']:,} games, "
        f"{manifest['summary']['neural_real_decisions']:,} decisions, "
        f"{elapsed:.1f}s this invocation."
    )
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--chunk-games", type=int, default=DEFAULT_CHUNK_GAMES)
    parser.add_argument("--job-games", type=int, default=DEFAULT_JOB_GAMES)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--compresslevel", type=int, default=6)
    parser.add_argument("--rulesets", default="all")
    parser.add_argument("--weights-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and regenerate selected raw ruleset directories.",
    )
    args = parser.parse_args(argv)
    if args.games < 1:
        parser.error("--games must be positive")
    if not 1 <= args.workers <= 20:
        parser.error("--workers must be between 1 and 20")
    if args.chunk_games < 1:
        parser.error("--chunk-games must be positive")
    if not 1 <= args.job_games <= args.chunk_games:
        parser.error("--job-games must be between 1 and --chunk-games")
    if args.base_seed < 0:
        parser.error("--base-seed must be non-negative")
    if not 1 <= args.compresslevel <= 9:
        parser.error("--compresslevel must be between 1 and 9")
    args.weights_dir = args.weights_dir.resolve()
    args.raw_root = args.raw_root.resolve()
    checkpoints = discover_checkpoints(args.weights_dir)
    try:
        args.selected_rulesets = parse_ruleset_selection(
            args.rulesets,
            available=checkpoints,
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.checkpoints = checkpoints
    return args


def main(argv=None):
    args = parse_args(argv)
    print(
        f"Generating {args.games:,} neural-vs-heuristic full histories for "
        f"{len(args.selected_rulesets)} ruleset(s), using {args.workers} "
        "CPU-only workers."
    )
    for ruleset_name in args.selected_rulesets:
        checkpoint = args.checkpoints[ruleset_name]
        print(
            f"\n{ruleset_name}: {checkpoint['file']} "
            f"({checkpoint['sha256'][:12]}...)"
        )
        generate_ruleset(args, ruleset_name, checkpoint)


if __name__ == "__main__":
    main()
