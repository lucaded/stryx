"""Terminal/web command -> OpenAI LLM -> per-drone orbit assignments -> crazyflow sim with AMSwarm."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr
import jax
import numpy as np
import yaml
from axswarm import SolverData, SolverSettings, solve
from crazyflow.control import Control
from crazyflow.sim import Physics, Sim
from openai import OpenAI

OPENAI_MODEL = "gpt-4o"
N_DRONES = 3
DEFAULT_DURATION_S = 15.0
WAYPOINT_DT_S = 0.25

RADIUS_RANGE = (0.3, 2.0)
HEIGHT_RANGE = (0.5, 1.8)
SPEED_RANGE = (0.1, 1.5)
CENTER_RANGE = (-2.0, 2.0)

REPO_ROOT = Path(__file__).parent
SETTINGS_PATH = REPO_ROOT / "swarm_gpt" / "data" / "settings.yaml"

SYSTEM_PROMPT = """You are a swarm-cinematography director controlling 3 drones (indices 0, 1, 2) flying in simulation. Output ONE JSON object only. No prose, no code fences.

Pick ONE of two modes based on the command:

(A) "primitive" — when the user asks for a simple orbit/circle and does NOT imply distinct camera angles:
{
  "mode": "primitive",
  "strategy": "single_orbit",
  "rationale": "<one sentence>",
  "assignments": [
    {"drone": 0, "role": "orbit", "primitive": "orbit", "params": {"radius": <m>, "height": <m>, "speed": <m/s>, "center": [<x>, <y>]}},
    {"drone": 1, "role": "orbit", "primitive": "orbit", "params": {"radius": <m>, "height": <m>, "speed": <m/s>, "center": [<x>, <y>]}},
    {"drone": 2, "role": "orbit", "primitive": "orbit", "params": {"radius": <m>, "height": <m>, "speed": <m/s>, "center": [<x>, <y>]}}
  ]
}
For single_orbit all three drones MUST share the same params; phase offsets are added by the runtime.

(B) "decompose" — when the user implies coverage of a subject (e.g. "cover this", "give me coverage", "film this cinematically", "shot reverse shot"). Pick ONE strategy and give each drone a distinct role with its own params:
{
  "mode": "decompose",
  "strategy": "<full_coverage|triangulation|lead_chase|hero_with_context|crowd_reaction|shot_reverse_shot|master_with_coverage|over_the_shoulder|high_low|bullet_time|reveal_arc|goal_to_goal>",
  "rationale": "<one sentence justifying why this strategy fits the command>",
  "assignments": [
    {"drone": 0, "role": "<role>", "primitive": "orbit", "params": {"radius": <m>, "height": <m>, "speed": <m/s>, "center": [<x>, <y>]}},
    {"drone": 1, "role": "<role>", "primitive": "orbit", "params": {"radius": <m>, "height": <m>, "speed": <m/s>, "center": [<x>, <y>]}},
    {"drone": 2, "role": "<role>", "primitive": "orbit", "params": {"radius": <m>, "height": <m>, "speed": <m/s>, "center": [<x>, <y>]}}
  ]
}

Decompose strategies:
- full_coverage: wide_establish + close_follow + side_angle (3 distinct framings of same subject).
- triangulation: ~120° angular separation around subject; vary heights for 3D capture.
- lead_chase: same radius and height; phase offsets so drones lead/middle/trail.
- hero_with_context: drone 0 hero low close; drones 1-2 wide and high, behind.
- crowd_reaction: drone 0 on subject; drones 1-2 turned outward, capturing audience.
- shot_reverse_shot: dialogue coverage. Drone 0 orbits left actor (center ≈ [-0.7, 0]); drone 1 orbits right actor (center ≈ [0.7, 0]); drone 2 wide (center origin, large radius).
- master_with_coverage: drone 0 wide master; drones 1-2 medium close-ups from different sides.
- over_the_shoulder: drone 0 OTS (close, height ≈ 1.5); drone 1 reverse on subject (close, low); drone 2 wide context.
- high_low: drone 0 overhead (height ≈ 1.8, slow); drone 1 eye-level (height ≈ 1.0); drone 2 low (height ≈ 0.5).
- bullet_time: same radius and height, coordinated phase sweep — Matrix-style synced arc.
- reveal_arc: drones start near center (small radius) — visually a slow expanding orbit.
- goal_to_goal: drones at distinct centers along x-axis ([-1.5, 0], [0, 0], [1.5, 0]), small radii, all facing center.

Role-to-parameter visual encoding (each drone executes as an orbit with its own radius/height/speed/center):
- wide_establish: radius 1.5–2.0m, height 1.5–1.8m, speed 0.2–0.4 m/s
- close_follow:   radius 0.4–0.8m, height 0.9–1.2m, speed 0.6–1.0 m/s
- side_angle:     radius 1.0–1.5m, height 0.5–0.8m, speed 0.4–0.7 m/s
- hero_low:       radius 0.4–0.6m, height 0.5–0.7m, speed 0.5–0.8 m/s
- context_back:   radius 1.7–2.0m, height 1.4–1.7m, speed 0.2–0.4 m/s
- audience:       radius 1.2–1.5m, height 0.9–1.1m, speed 0.3–0.5 m/s

Hard parameter bounds (clamp to these): radius in [0.3, 2.0]m, height in [0.5, 1.8]m, speed in [0.1, 1.5]m/s, center in [-2.0, 2.0]m on each axis.
Always emit exactly 3 assignments (drones 0, 1, 2) and ALWAYS include a non-empty rationale."""


def parse_command(text: str) -> dict:
    client = OpenAI()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=600,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return json.loads(resp.choices[0].message.content)


def _clip(v, lo, hi, default):
    try:
        return float(np.clip(float(v), lo, hi))
    except (TypeError, ValueError):
        return float(default)


def clamp_orbit_params(p: dict) -> dict:
    center = p.get("center") if isinstance(p.get("center"), (list, tuple)) else [0.0, 0.0]
    return {
        "radius": _clip(p.get("radius", 1.0), *RADIUS_RANGE, 1.0),
        "height": _clip(p.get("height", 1.0), *HEIGHT_RANGE, 1.0),
        "speed": _clip(p.get("speed", 0.5), *SPEED_RANGE, 0.5),
        "center": [
            _clip(center[0] if len(center) > 0 else 0.0, *CENTER_RANGE, 0.0),
            _clip(center[1] if len(center) > 1 else 0.0, *CENTER_RANGE, 0.0),
        ],
    }


def normalize(raw: dict) -> dict:
    """Produce {mode, strategy, rationale, assignments[3]} with clamped params."""
    if isinstance(raw.get("assignments"), list) and raw["assignments"]:
        out = {
            "mode": str(raw.get("mode", "primitive")),
            "strategy": str(raw.get("strategy", "single_orbit")),
            "rationale": str(raw.get("rationale", "") or ""),
            "assignments": [
                {
                    "drone": int(a["drone"]),
                    "role": str(a.get("role", "orbit")),
                    "primitive": str(a.get("primitive", "orbit")),
                    "params": clamp_orbit_params(a.get("params", {})),
                }
                for a in raw["assignments"]
            ],
        }
        out["assignments"].sort(key=lambda x: x["drone"])
        return out
    if raw.get("primitive") == "orbit" and isinstance(raw.get("params"), dict):
        p = clamp_orbit_params(raw["params"])
        return {
            "mode": "primitive",
            "strategy": "single_orbit",
            "rationale": "Legacy single-primitive command; replicated across drones.",
            "assignments": [
                {"drone": i, "role": "orbit", "primitive": "orbit", "params": p}
                for i in range(N_DRONES)
            ],
        }
    raise ValueError(f"Unrecognized command shape: {json.dumps(raw)[:200]}")


def build_waypoints(
    assignments: list[dict],
    duration: float = DEFAULT_DURATION_S,
    dt: float = WAYPOINT_DT_S,
) -> dict[str, np.ndarray]:
    n = len(assignments)
    n_samples = max(2, int(np.ceil(duration / dt)) + 1)
    times = np.linspace(0.0, duration, n_samples)
    pos = np.zeros((n, n_samples, 3), dtype=np.float64)
    vel = np.zeros((n, n_samples, 3), dtype=np.float64)
    by_idx = {int(a["drone"]): a for a in assignments}
    for i in range(n):
        a = by_idx[i]
        p = a["params"]
        radius = p["radius"]
        height = p["height"]
        speed = p["speed"]
        cx, cy = p["center"]
        omega = speed / max(radius, 1e-3)
        phase = 2.0 * np.pi * i / n
        angle = omega * times + phase
        pos[i, :, 0] = cx + radius * np.cos(angle)
        pos[i, :, 1] = cy + radius * np.sin(angle)
        pos[i, :, 2] = height
        vel[i, :, 0] = -radius * omega * np.sin(angle)
        vel[i, :, 1] = radius * omega * np.cos(angle)
    t = np.tile(times, (n, 1))
    return {"time": t, "pos": pos, "vel": vel, "acc": np.zeros_like(pos)}


def run_sim(waypoints: dict, settings: dict, gui: bool = True) -> None:
    sim = Sim(
        n_worlds=1,
        n_drones=waypoints["pos"].shape[0],
        physics=Physics.analytical,
        control=Control.state,
        freq=settings["sim_freq"],
        attitude_freq=settings["attitude_freq"],
        state_freq=settings["state_freq"],
        device="cpu",
    )
    sim.max_visual_geom = 100_000
    sim.reset()
    sim.state_control(np.zeros((sim.n_worlds, sim.n_drones, 13), dtype=np.float32))
    sim.step(sim.freq // sim.control_freq)
    sim.reset()

    solver_settings_kwargs = {
        k: (np.asarray(v) if isinstance(v, list) else v)
        for k, v in settings["axswarm"].items()
    }
    solver_settings = SolverSettings(**solver_settings_kwargs)
    dyn = settings["Dynamics"]
    A, B = np.asarray(dyn["A"]), np.asarray(dyn["B"])
    A_prime, B_prime = np.asarray(dyn["A_prime"]), np.asarray(dyn["B_prime"])
    solver_data = SolverData.init(
        waypoints=waypoints,
        K=solver_settings.K,
        N=solver_settings.N,
        A=A,
        B=B,
        A_prime=A_prime,
        B_prime=B_prime,
        freq=solver_settings.freq,
        smoothness_weight=solver_settings.smoothness_weight,
        input_smoothness_weight=solver_settings.input_smoothness_weight,
        input_continuity_weight=solver_settings.input_continuity_weight,
    )

    n_steps = int(waypoints["time"][0, -1] * sim.control_freq)
    solve_every_n_steps = sim.control_freq // solver_settings.freq

    control = np.zeros((sim.n_worlds, sim.n_drones, 13), dtype=np.float32)
    pos_init = sim.data.states.pos.at[0, ...].set(waypoints["pos"][:, 0])
    sim.data = sim.data.replace(states=sim.data.states.replace(pos=pos_init))
    pos = np.asarray(sim.data.states.pos[0])
    vel = np.asarray(sim.data.states.vel[0])

    fps = 60
    print(f"Running {n_steps} sim steps ({waypoints['time'][0, -1]:.1f}s) "
          f"with {sim.n_drones} drones; GUI={gui}")
    for step in range(n_steps):
        t = step / sim.control_freq
        if step % solve_every_n_steps == 0:
            state = np.concatenate((pos, vel), axis=-1)
            success, _, solver_data = solve(state, t, solver_data, solver_settings)
            jax.block_until_ready(solver_data)
            if not all(success):
                print(f"[t={t:.2f}s] AMSwarm solve failed", file=sys.stderr)
            solver_data = solver_data.step(solver_data)
            pos = np.asarray(solver_data.u_pos[:, 0])
            vel = np.asarray(solver_data.u_vel[:, 0])
            control[0, :, :3] = solver_data.u_pos[:, 0]
            control[0, :, 3:6] = solver_data.u_vel[:, 0]
        sim.state_control(control)
        sim.step(sim.freq // sim.control_freq)
        if gui and (step * fps) % sim.control_freq < fps:
            sim.render()
    sim.close()
    print("Done.")


def render_roles_markdown(parsed: dict) -> str:
    head = [
        f"**Mode:** `{parsed['mode']}` &nbsp;&nbsp; **Strategy:** `{parsed['strategy']}`",
        "",
        f"_{parsed.get('rationale', '') or '(no rationale)'}_",
        "",
        "| Drone | Role | Radius | Height | Speed | Center |",
        "|---|---|---|---|---|---|",
    ]
    for a in parsed["assignments"]:
        p = a["params"]
        head.append(
            f"| {a['drone']} | `{a['role']}` | {p['radius']:.2f} m | "
            f"{p['height']:.2f} m | {p['speed']:.2f} m/s | "
            f"[{p['center'][0]:.2f}, {p['center'][1]:.2f}] |"
        )
    return "\n".join(head)


def print_summary(parsed: dict) -> None:
    print(f"Mode: {parsed['mode']} | Strategy: {parsed['strategy']}")
    if parsed.get("rationale"):
        print(f"Rationale: {parsed['rationale']}")
    for a in parsed["assignments"]:
        p = a["params"]
        print(
            f"  drone {a['drone']}  role={a['role']:<18}  "
            f"r={p['radius']:.2f}m  h={p['height']:.2f}m  v={p['speed']:.2f}m/s  "
            f"c=[{p['center'][0]:.2f},{p['center'][1]:.2f}]"
        )


def argv_command() -> str:
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    return " ".join(positional).strip()


def execute(text: str, settings: dict, gui: bool) -> None:
    print(f"Parsing: {text!r}")
    raw = parse_command(text)
    parsed = normalize(raw)
    print_summary(parsed)
    waypoints = build_waypoints(parsed["assignments"])
    run_sim(waypoints, settings, gui=gui)


def launch_ui(settings: dict) -> None:
    def submit(text: str, show_viewer: bool) -> tuple[str, str, str]:
        text = text.strip()
        if not text:
            return "", "", "Type a command first."
        try:
            raw = parse_command(text)
            parsed = normalize(raw)
        except Exception as exc:
            return "", "", f"LLM/parse error: {exc}"
        roles_md = render_roles_markdown(parsed)
        parsed_json = json.dumps(parsed, indent=2)
        try:
            waypoints = build_waypoints(parsed["assignments"])
            run_sim(waypoints, settings, gui=show_viewer)
        except Exception as exc:
            return roles_md, parsed_json, f"Sim error: {exc}"
        return roles_md, parsed_json, f"Done. {N_DRONES} drones, {DEFAULT_DURATION_S:.0f}s flown."

    with gr.Blocks(title="dolly MVP") as ui:
        gr.Markdown("# dolly MVP — central-planner swarm cinematography")
        with gr.Row():
            cmd_in = gr.Textbox(
                label="Command",
                placeholder='Try: "cover this play" or "shot reverse shot of two actors"',
                scale=4,
            )
            submit_btn = gr.Button("Run", variant="primary", scale=1)
        viewer_chk = gr.Checkbox(value=True, label="Open MuJoCo 3D viewer window")
        roles_out = gr.Markdown(label="Director's plan")
        with gr.Row():
            parsed_out = gr.Code(label="Parsed JSON (debug)", language="json")
            status_out = gr.Textbox(label="Status", lines=4)
        outputs = [roles_out, parsed_out, status_out]
        submit_btn.click(submit, inputs=[cmd_in, viewer_chk], outputs=outputs)
        cmd_in.submit(submit, inputs=[cmd_in, viewer_chk], outputs=outputs)
    ui.launch()


def main() -> None:
    settings = yaml.safe_load(SETTINGS_PATH.read_text())
    gui = "--no-gui" not in sys.argv

    one_shot = argv_command()
    if one_shot:
        execute(one_shot, settings, gui)
        return
    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        if text:
            execute(text, settings, gui)
        return

    if "--repl" in sys.argv:
        print("REPL mode. Type a command (empty line or Ctrl+D to exit).")
        while True:
            try:
                text = input("Command: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not text:
                return
            try:
                execute(text, settings, gui)
            except Exception as exc:
                print(f"Error: {exc}", file=sys.stderr)
        return

    launch_ui(settings)


if __name__ == "__main__":
    main()
