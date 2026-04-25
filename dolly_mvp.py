"""Terminal/web command -> OpenAI LLM -> orbit primitive -> crazyflow sim with AMSwarm safety."""

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

OPENAI_MODEL = "gpt-4o-mini"
N_DRONES = 3
DEFAULT_CENTER_XY = (0.0, 0.0)
DEFAULT_DURATION_S = 15.0
WAYPOINT_DT_S = 0.25

REPO_ROOT = Path(__file__).parent
SETTINGS_PATH = REPO_ROOT / "swarm_gpt" / "data" / "settings.yaml"

SYSTEM_PROMPT = (
    "You are a swarm-control parser. The user describes a drone maneuver. "
    "Output ONE line of valid JSON matching exactly this schema:\n"
    '{"primitive": "orbit", "params": {"radius": <float>, "height": <float>, "speed": <float>}}\n'
    "Constraints: radius in [0.3, 2.0] m, height in [0.5, 1.8] m, speed in [0.1, 1.5] m/s. "
    "Output JSON only. No prose, no code fences, no commentary."
)


def parse_command(text: str) -> dict:
    client = OpenAI()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=256,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return json.loads(resp.choices[0].message.content)


def orbit(
    center_xy: tuple[float, float],
    radius: float,
    height: float,
    speed: float,
    n_drones: int,
    duration: float,
    dt: float = WAYPOINT_DT_S,
) -> dict[str, np.ndarray]:
    omega = speed / radius
    n_samples = max(2, int(np.ceil(duration / dt)) + 1)
    times = np.linspace(0.0, duration, n_samples)
    pos = np.zeros((n_drones, n_samples, 3), dtype=np.float64)
    vel = np.zeros((n_drones, n_samples, 3), dtype=np.float64)
    cx, cy = center_xy
    for i in range(n_drones):
        phase = 2.0 * np.pi * i / n_drones
        angle = omega * times + phase
        pos[i, :, 0] = cx + radius * np.cos(angle)
        pos[i, :, 1] = cy + radius * np.sin(angle)
        pos[i, :, 2] = height
        vel[i, :, 0] = -radius * omega * np.sin(angle)
        vel[i, :, 1] = radius * omega * np.cos(angle)
    t = np.tile(times, (n_drones, 1))
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


def argv_command() -> str:
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    return " ".join(positional).strip()


def execute(text: str, settings: dict, gui: bool) -> None:
    print(f"Parsing: {text!r}")
    cmd = parse_command(text)
    print("Parsed:", json.dumps(cmd))
    if cmd.get("primitive") != "orbit":
        print(f"Only 'orbit' is supported in this MVP. Got {cmd.get('primitive')!r}.",
              file=sys.stderr)
        return
    p = cmd["params"]
    waypoints = orbit(
        center_xy=DEFAULT_CENTER_XY,
        radius=float(p["radius"]),
        height=float(p["height"]),
        speed=float(p["speed"]),
        n_drones=N_DRONES,
        duration=DEFAULT_DURATION_S,
    )
    run_sim(waypoints, settings, gui=gui)


def launch_ui(settings: dict) -> None:
    def submit(text: str, show_viewer: bool) -> tuple[str, str]:
        text = text.strip()
        if not text:
            return "", "Type a command first."
        try:
            cmd = parse_command(text)
        except Exception as exc:
            return "", f"LLM/parse error: {exc}"
        parsed = json.dumps(cmd, indent=2)
        if cmd.get("primitive") != "orbit":
            return parsed, f"Only 'orbit' is supported. Got {cmd.get('primitive')!r}."
        try:
            p = cmd["params"]
            waypoints = orbit(
                center_xy=DEFAULT_CENTER_XY,
                radius=float(p["radius"]),
                height=float(p["height"]),
                speed=float(p["speed"]),
                n_drones=N_DRONES,
                duration=DEFAULT_DURATION_S,
            )
            run_sim(waypoints, settings, gui=show_viewer)
        except Exception as exc:
            return parsed, f"Sim error: {exc}"
        return parsed, f"Done. {N_DRONES} drones, {DEFAULT_DURATION_S:.0f}s of orbit."

    with gr.Blocks(title="dolly MVP") as ui:
        gr.Markdown("# dolly MVP\nType a drone command. The LLM parses it and the swarm runs it in sim.")
        with gr.Row():
            cmd_in = gr.Textbox(
                label="Command",
                placeholder="orbit a 1m circle at 0.5 m/s, height 1m",
                scale=4,
            )
            submit_btn = gr.Button("Run", variant="primary", scale=1)
        viewer_chk = gr.Checkbox(value=True, label="Open MuJoCo 3D viewer window")
        with gr.Row():
            parsed_out = gr.Code(label="Parsed JSON", language="json")
            status_out = gr.Textbox(label="Status", lines=4)
        submit_btn.click(submit, inputs=[cmd_in, viewer_chk], outputs=[parsed_out, status_out])
        cmd_in.submit(submit, inputs=[cmd_in, viewer_chk], outputs=[parsed_out, status_out])
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
