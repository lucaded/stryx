"""Terminal/web command -> OpenAI LLM -> per-drone primitives -> crazyflow sim with AMSwarm."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Prefer EGL for offscreen rendering; falls back to OSMesa (software) if EGL libs missing.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import gradio as gr
import imageio.v3 as iio
import jax
import mujoco
import numpy as np
import yaml
from axswarm import SolverData, SolverSettings, solve
from crazyflow.control import Control
from crazyflow.sim import Physics, Sim
from openai import OpenAI

OPENAI_MODEL = "gpt-4o"
LLM_TIMEOUT_S = 20.0

N_DRONES = 3
DEFAULT_DURATION_S = 15.0
WAYPOINT_DT_S = 0.25

RADIUS_RANGE = (0.3, 2.0)
HEIGHT_RANGE = (0.5, 1.8)
SPEED_RANGE = (0.1, 1.5)
OFFSET_RANGE = (-1.5, 1.5)
CENTER_RANGE = (-2.0, 2.0)
DURATION_RANGE = (5.0, 30.0)
FREEFORM_VOLUME_X = (-2.5, 2.5)
FREEFORM_VOLUME_Y = (-2.5, 2.5)
FREEFORM_VOLUME_Z = (0.5, 1.8)
FREEFORM_MIN_WAYPOINTS = 2
FREEFORM_MAX_WAYPOINTS = 12
FREEFORM_OUT_OF_VOLUME_FRACTION_LIMIT = 0.30

SUBJECT_PERIOD_S = 12.0
SUBJECT_AMPLITUDE_X = 1.2
SUBJECT_AMPLITUDE_Y = 0.6
SUBJECT_BOX_HALF = np.array([0.25, 0.25, 0.45])
SUBJECT_BOX_RGBA = np.array([0.85, 0.25, 0.35, 1.0])
POV_FORWARD_OFFSET = 0.12

REPO_ROOT = Path(__file__).parent
SETTINGS_PATH = REPO_ROOT / "swarm_gpt" / "data" / "settings.yaml"

SYSTEM_PROMPT = """You are a swarm-cinematography director controlling 3 drones (indices 0, 1, 2) in simulation. Output ONE JSON object only. No prose, no code fences.

The "subject" is a moving box on the ground. By default it wanders (figure_eight). Set static only when the command clearly implies a stationary subject ("hover around this spot", "static circle"). Top-level field:
"subject": {"motion": "figure_eight" | "random_walk" | "static"}

CRITICAL: When subject motion is NOT static, orbit and follow primitives both TRACK the subject — the orbit "center" parameter is then an OFFSET from the moving subject (so center [0, 0] means orbit centered on the box wherever it goes). For dialogue scenes with two stationary actors at fixed spots (shot_reverse_shot, master_with_coverage), explicitly set subject motion to "static" and use absolute world-frame centers like [-0.7, 0] and [0.7, 0].

Pick the MODE that best matches the command. Use this decision rule:

1. If the command names a SPECIFIC NON-ORBITAL SHAPE, PATH, or CREATIVE MOTION ("figure 8", "figure-eight", "heart", "spiral", "weave between X and Y", "trace a letter", "fly through a gate", "swoosh", "S-curve", "spell"), choose "freeform" and emit waypoints. Do NOT shoehorn into orbit.
2. Else, if the command implies coordinated multi-role COVERAGE of a subject ("cover this", "give me coverage", "film cinematically", "capture from all angles", "shot reverse shot", "master with coverage"), choose "decompose" with a named strategy.
3. Else, if the command names a known SHOT TYPE cleanly ("orbit", "circle", "follow", "hero shot"), choose "primitive".
4. When ambiguous between freeform and primitive, prefer freeform if the command suggests a creative or shape-specific motion; prefer primitive only when it's a plain orbit/circle/follow with no shape qualifier.

Don't default to orbit. Orbit is for circular motion only. If the command says "figure 8", "heart", "weave", or any non-circular path, use freeform.

(A) "primitive" — one shot type, all drones share params, phase offsets handled by runtime:
{
  "mode": "primitive",
  "subject": {"motion": "static"},
  "strategy": "single_orbit" | "follow_train",
  "rationale": "<one sentence>",
  "assignments": [
    {"drone": 0, "role": "<role>", "primitive": "orbit"|"follow", "params": {...}},
    {"drone": 1, ...},
    {"drone": 2, ...}
  ]
}

(B) "decompose" — coordinated multi-role coverage (each drone gets distinct params/role):
{
  "mode": "decompose",
  "subject": {"motion": "static"|"figure_eight"},
  "strategy": "<one of the strategies below>",
  "rationale": "<one sentence justifying the strategy choice>",
  "assignments": [
    {"drone": 0, "role": "<role>", "primitive": "orbit"|"follow", "params": {...}},
    {"drone": 1, ...},
    {"drone": 2, ...}
  ]
}

Decompose strategies:
- full_coverage: wide_establish + close_follow + side_angle (3 distinct framings of same subject).
- triangulation: ~120° angular separation around subject; vary heights for 3D capture.
- lead_chase: same radius and height; phase offsets so drones lead/middle/trail (orbit only).
- hero_with_context: drone 0 hero low close; drones 1-2 wide and high, behind.
- crowd_reaction: drone 0 on subject; drones 1-2 turned outward, capturing audience.
- shot_reverse_shot: dialogue coverage. Drone 0 orbits left actor (center ≈ [-0.7, 0]); drone 1 orbits right actor (center ≈ [0.7, 0]); drone 2 wide (center origin, large radius).
- master_with_coverage: drone 0 wide master; drones 1-2 medium close-ups from different sides.
- over_the_shoulder: drone 0 OTS (close, height ≈ 1.5); drone 1 reverse on subject (close, low); drone 2 wide context.
- high_low: drone 0 overhead (height ≈ 1.8, slow); drone 1 eye-level (height ≈ 1.0); drone 2 low (height ≈ 0.5).
- bullet_time: same radius and height, coordinated phase sweep — Matrix-style synced arc.
- reveal_arc: drones start near center (small radius) — visually a slow expanding orbit.
- goal_to_goal: drones at distinct centers along x-axis ([-1.5, 0], [0, 0], [1.5, 0]), small radii, all facing center.

(C) "freeform" — only when no primitive fits ("figure 8 around the actors", "heart shape", "spell W", "fly through gate"):
{
  "mode": "freeform",
  "subject": {"motion": "static"},
  "strategy": "<short_descriptor>",
  "rationale": "<one sentence justifying why no primitive fits>",
  "duration": <seconds, 5-30>,
  "assignments": [
    {
      "drone": 0, "role": "<role>", "primitive": "freeform",
      "waypoints": [
        {"t": 0.0, "x": <m>, "y": <m>, "z": <m>},
        ...
      ]
    },
    {"drone": 1, ...},
    {"drone": 2, ...}
  ]
}

Primitive parameter formats:
- orbit: {"radius": <m>, "height": <m>, "speed": <m/s>, "center": [<x>, <y>] (optional, default [0,0])}
- follow: {"offset": [<dx>, <dy>], "height": <m>}
  IMPORTANT: "offset" MUST be a 2-element list [dx, dy] in meters (world frame). Never emit 1-element, empty, or scalar offsets. dx and dy must be distinct per drone for multi-drone coverage (e.g. drone 0 [-1.0, 0], drone 1 [0, 0.8], drone 2 [0.6, -0.6]).

Hard parameter bounds:
- orbit: radius [0.3, 2.0] m, height [0.5, 1.8] m, speed [0.1, 1.5] m/s, center each in [-2.0, 2.0] m
- follow: |offset_x|, |offset_y| ≤ 1.5 m; height [0.5, 1.8] m
- freeform: emit 6-10 waypoints per drone (denser = smoother). t strictly increasing, start 0.0, end at duration. x,y in [-2.5, 2.5] m, z in [0.5, 1.8] m. Vary trajectories per drone — three identical paths defeat the point. NEVER emit fewer than 6 waypoints unless the path is genuinely a single straight segment.
- duration in [5, 30] s

All modes: exactly 3 assignments (drones 0, 1, 2). Always include a non-empty rationale."""


# ---------- subject ----------

def subject_position(t: float | np.ndarray, mode: str) -> np.ndarray:
    """Return subject (x, y, z) at time(s) t. Modes: static, figure_eight, random_walk."""
    t = np.asarray(t, dtype=np.float64)
    z = np.zeros_like(t)
    if mode == "figure_eight":
        omega = 2.0 * np.pi / SUBJECT_PERIOD_S
        x = SUBJECT_AMPLITUDE_X * np.sin(omega * t)
        y = SUBJECT_AMPLITUDE_Y * np.sin(2.0 * omega * t)
    elif mode == "random_walk":
        w1, w2, w3 = 2.0 * np.pi / 7.0, 2.0 * np.pi / 11.0, 2.0 * np.pi / 13.0
        x = 0.7 * np.sin(w1 * t) + 0.4 * np.sin(w2 * t + 1.3) + 0.3 * np.cos(w3 * t + 0.7)
        y = 0.5 * np.cos(w1 * t + 2.1) + 0.6 * np.sin(w3 * t) + 0.3 * np.cos(w2 * t + 0.5)
    else:
        x = z.copy()
        y = z.copy()
    return np.stack([x, y, z], axis=-1)


def draw_subject(sim: Sim, t: float, motion: str) -> None:
    """Draw the subject as a moving box on the ground."""
    if sim.viewer is None or sim.viewer.viewer is None:
        return
    viewer = sim.viewer.viewer
    pos = subject_position(t, motion)
    box_pos = np.array([pos[0], pos[1], SUBJECT_BOX_HALF[2]], dtype=np.float64)
    viewer.add_marker(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=SUBJECT_BOX_HALF,
        pos=box_pos,
        mat=np.eye(3).flatten(),
        rgba=SUBJECT_BOX_RGBA,
    )


# ---------- LLM ----------

def parse_command(text: str) -> tuple[dict | None, str | None]:
    client = OpenAI()
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=1800,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            timeout=LLM_TIMEOUT_S,
        )
    except Exception as exc:
        return None, f"LLM call failed: {exc}"
    try:
        return json.loads(resp.choices[0].message.content), None
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc}"


# ---------- normalization & validation ----------

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


DEFAULT_FOLLOW_OFFSETS = [(-0.9, 0.0), (-0.4, 0.7), (-0.4, -0.7)]
DEFAULT_FOLLOW_HEIGHTS = (1.5, 1.0, 0.7)


def clamp_follow_params(p: dict, drone_idx: int = 0) -> dict:
    dx_def, dy_def = DEFAULT_FOLLOW_OFFSETS[drone_idx % len(DEFAULT_FOLLOW_OFFSETS)]
    h_def = DEFAULT_FOLLOW_HEIGHTS[drone_idx % len(DEFAULT_FOLLOW_HEIGHTS)]
    raw_offset = p.get("offset")
    if isinstance(raw_offset, (list, tuple)):
        offset = list(raw_offset)
    elif isinstance(raw_offset, (int, float)):
        offset = [float(raw_offset), 0.0]
    else:
        offset = []
    return {
        "offset": [
            _clip(offset[0] if len(offset) > 0 else dx_def, *OFFSET_RANGE, dx_def),
            _clip(offset[1] if len(offset) > 1 else dy_def, *OFFSET_RANGE, dy_def),
        ],
        "height": _clip(p.get("height", h_def), *HEIGHT_RANGE, h_def),
    }


def follow_fallback(reason: str) -> dict:
    return {
        "mode": "primitive",
        "subject": {"motion": "figure_eight"},
        "strategy": "follow_train",
        "rationale": f"Fallback: {reason}",
        "duration": DEFAULT_DURATION_S,
        "assignments": [
            {"drone": i, "role": "follow", "primitive": "follow",
             "params": clamp_follow_params({}, i)}
            for i in range(N_DRONES)
        ],
    }


def normalize(raw: dict) -> dict:
    if not isinstance(raw.get("assignments"), list) or not raw["assignments"]:
        if raw.get("primitive") == "orbit" and isinstance(raw.get("params"), dict):
            p = clamp_orbit_params(raw["params"])
            return {
                "mode": "primitive",
                "subject": {"motion": "static"},
                "strategy": "single_orbit",
                "rationale": "Legacy single-primitive command; replicated across drones.",
                "duration": DEFAULT_DURATION_S,
                "assignments": [
                    {"drone": i, "role": "orbit", "primitive": "orbit", "params": p}
                    for i in range(N_DRONES)
                ],
            }
        raise ValueError(f"Unrecognized command shape: {json.dumps(raw)[:200]}")

    subject = raw.get("subject")
    motion = (subject or {}).get("motion", "figure_eight")
    if motion not in {"static", "figure_eight"}:
        motion = "figure_eight"

    duration = _clip(raw.get("duration", DEFAULT_DURATION_S), *DURATION_RANGE, DEFAULT_DURATION_S)
    out = {
        "mode": str(raw.get("mode", "primitive")),
        "subject": {"motion": motion},
        "strategy": str(raw.get("strategy", "single_orbit")),
        "rationale": str(raw.get("rationale", "") or ""),
        "duration": duration,
        "assignments": [],
    }
    for a in raw["assignments"]:
        prim = str(a.get("primitive", "orbit"))
        entry: dict = {
            "drone": int(a["drone"]),
            "role": str(a.get("role", prim)),
            "primitive": prim,
        }
        if prim == "orbit":
            entry["params"] = clamp_orbit_params(a.get("params", {}))
        elif prim == "follow":
            entry["params"] = clamp_follow_params(a.get("params", {}), int(a["drone"]))
        elif prim == "freeform":
            entry["waypoints"] = a.get("waypoints", []) or []
        else:
            entry["primitive"] = "orbit"
            entry["params"] = clamp_orbit_params(a.get("params", {}))
        out["assignments"].append(entry)
    out["assignments"].sort(key=lambda x: x["drone"])
    return out


def validate_freeform_or_fallback(parsed: dict) -> tuple[dict, str | None]:
    """If parsed is freeform and fails the validation rules, return follow fallback."""
    if parsed.get("mode") != "freeform":
        return parsed, None
    issues: list[str] = []
    total_wp = 0
    out_of_vol = 0
    for a in parsed["assignments"]:
        wps = a.get("waypoints", [])
        n = len(wps)
        if not (FREEFORM_MIN_WAYPOINTS <= n <= FREEFORM_MAX_WAYPOINTS):
            issues.append(f"drone {a['drone']}: {n} waypoints (need {FREEFORM_MIN_WAYPOINTS}-{FREEFORM_MAX_WAYPOINTS})")
        total_wp += n
        for w in wps:
            try:
                x, y, z = float(w["x"]), float(w["y"]), float(w["z"])
            except (TypeError, KeyError, ValueError):
                out_of_vol += 1
                continue
            if not (FREEFORM_VOLUME_X[0] <= x <= FREEFORM_VOLUME_X[1] and
                    FREEFORM_VOLUME_Y[0] <= y <= FREEFORM_VOLUME_Y[1] and
                    FREEFORM_VOLUME_Z[0] <= z <= FREEFORM_VOLUME_Z[1]):
                out_of_vol += 1
    frac = (out_of_vol / total_wp) if total_wp else 1.0
    if frac > FREEFORM_OUT_OF_VOLUME_FRACTION_LIMIT:
        issues.append(f"{out_of_vol}/{total_wp} waypoints out of volume ({frac:.0%} > {FREEFORM_OUT_OF_VOLUME_FRACTION_LIMIT:.0%})")
    if issues:
        return follow_fallback("freeform validation: " + "; ".join(issues)), "; ".join(issues)
    return parsed, None


# ---------- waypoint generation ----------

def _fill_orbit(pos_arr, vel_arr, times, drone_i, n_drones, params, subject_motion):
    radius = params["radius"]
    height = params["height"]
    speed = params["speed"]
    cx, cy = params["center"]
    if subject_motion != "static":
        subj = subject_position(times, subject_motion)
        cxs = subj[:, 0] + cx
        cys = subj[:, 1] + cy
        if len(times) > 1:
            subj_vx = np.gradient(cxs, times)
            subj_vy = np.gradient(cys, times)
        else:
            subj_vx = np.zeros_like(times)
            subj_vy = np.zeros_like(times)
    else:
        cxs = np.full_like(times, cx)
        cys = np.full_like(times, cy)
        subj_vx = np.zeros_like(times)
        subj_vy = np.zeros_like(times)
    omega = speed / max(radius, 1e-3)
    phase = 2.0 * np.pi * drone_i / n_drones
    angle = omega * times + phase
    pos_arr[:, 0] = cxs + radius * np.cos(angle)
    pos_arr[:, 1] = cys + radius * np.sin(angle)
    pos_arr[:, 2] = height
    vel_arr[:, 0] = subj_vx - radius * omega * np.sin(angle)
    vel_arr[:, 1] = subj_vy + radius * omega * np.cos(angle)
    vel_arr[:, 2] = 0.0


def _fill_follow(pos_arr, vel_arr, times, params, subject_motion):
    dx, dy = params["offset"]
    height = params["height"]
    subj = subject_position(times, subject_motion)
    pos_arr[:, 0] = subj[:, 0] + dx
    pos_arr[:, 1] = subj[:, 1] + dy
    pos_arr[:, 2] = height
    if len(times) > 1:
        dt_arr = np.diff(times)[:, None]
        vel_arr[:-1] = np.diff(pos_arr, axis=0) / dt_arr
        vel_arr[-1] = vel_arr[-2]


def _fill_freeform(pos_arr, vel_arr, times, waypoints):
    cleaned: list[tuple[float, float, float, float]] = []
    last_t = -np.inf
    for w in waypoints:
        try:
            tt, xx, yy, zz = float(w["t"]), float(w["x"]), float(w["y"]), float(w["z"])
        except (TypeError, KeyError, ValueError):
            continue
        if any(np.isnan(v) for v in (tt, xx, yy, zz)) or tt <= last_t:
            continue
        cleaned.append((tt, xx, yy, zz))
        last_t = tt
    if len(cleaned) < 2:
        raise ValueError(f"freeform: only {len(cleaned)} valid waypoint(s)")
    arr = np.array(cleaned, dtype=np.float64)
    pos_arr[:, 0] = np.clip(np.interp(times, arr[:, 0], arr[:, 1]), *FREEFORM_VOLUME_X)
    pos_arr[:, 1] = np.clip(np.interp(times, arr[:, 0], arr[:, 2]), *FREEFORM_VOLUME_Y)
    pos_arr[:, 2] = np.clip(np.interp(times, arr[:, 0], arr[:, 3]), *FREEFORM_VOLUME_Z)
    if len(times) > 1:
        dt_arr = np.diff(times)[:, None]
        vel_arr[:-1] = np.diff(pos_arr, axis=0) / dt_arr
        vel_arr[-1] = vel_arr[-2]


def build_waypoints(parsed: dict) -> dict[str, np.ndarray]:
    duration = float(parsed.get("duration", DEFAULT_DURATION_S))
    motion = parsed.get("subject", {}).get("motion", "static")
    n = len(parsed["assignments"])
    n_samples = max(2, int(np.ceil(duration / WAYPOINT_DT_S)) + 1)
    times = np.linspace(0.0, duration, n_samples)
    pos = np.zeros((n, n_samples, 3), dtype=np.float64)
    vel = np.zeros((n, n_samples, 3), dtype=np.float64)
    by_idx = {int(a["drone"]): a for a in parsed["assignments"]}
    for i in range(n):
        a = by_idx[i]
        prim = a["primitive"]
        if prim == "orbit":
            _fill_orbit(pos[i], vel[i], times, i, n, a["params"], motion)
        elif prim == "follow":
            _fill_follow(pos[i], vel[i], times, a["params"], motion)
        elif prim == "freeform":
            _fill_freeform(pos[i], vel[i], times, a.get("waypoints", []))
        else:
            raise ValueError(f"Unknown primitive: {prim}")
    t = np.tile(times, (n, 1))
    return {"time": t, "pos": pos, "vel": vel, "acc": np.zeros_like(pos)}


# ---------- simulation ----------

POV_HEIGHT = 240
POV_WIDTH = 360
POV_FPS = 4
STATUS_HEARTBEAT_HZ = 2


def _aim_camera_at_subject(cam, drone_pos, subject_pos):
    drone = np.asarray(drone_pos, dtype=np.float64)
    subj = np.asarray(subject_pos, dtype=np.float64)
    to_subject = subj - drone
    norm = float(np.linalg.norm(to_subject))
    cam_world = drone + (to_subject / norm) * POV_FORWARD_OFFSET if norm > 1e-6 else drone
    v = cam_world - subj
    distance = float(np.linalg.norm(v))
    cam.lookat[0] = float(subj[0])
    cam.lookat[1] = float(subj[1])
    cam.lookat[2] = float(subj[2])
    cam.distance = max(distance, 0.1)
    cam.azimuth = float(np.degrees(np.arctan2(v[1], v[0])))
    cam.elevation = float(np.degrees(np.arctan2(v[2], np.linalg.norm(v[:2]) + 1e-6)))


def run_sim(
    waypoints: dict,
    settings: dict,
    gui: bool = True,
    subject_motion: str = "static",
    pov_drone_idx: int | None = None,
):
    """Run the sim. Generator yielding (t, pov_img_or_None) at POV_FPS.

    pov_drone_idx None -> no POV rendering (img is always None).
    Iterate fully to drive the sim to completion.
    """
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

    pov_active = pov_drone_idx is not None and 0 <= pov_drone_idx < sim.n_drones
    pov_viewer = None
    if pov_active and gui:
        try:
            import mujoco.viewer as _mjv
            pov_viewer = _mjv.launch_passive(
                sim.mj_model, sim.mj_data,
                show_left_ui=False, show_right_ui=False,
            )
        except Exception as exc:
            print(f"[POV viewer] failed to open second window: {exc}", file=sys.stderr)
            pov_viewer = None

    fps = 60
    print(f"Running {n_steps} sim steps ({waypoints['time'][0, -1]:.1f}s) "
          f"with {sim.n_drones} drones; GUI={gui}; POV={pov_drone_idx}")
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
            draw_subject(sim, t, subject_motion)
            sim.render()
            if pov_viewer is not None:
                try:
                    drone_pos = np.asarray(sim.data.states.pos[0, pov_drone_idx])
                    subj_pos = subject_position(t, subject_motion)
                    _aim_camera_at_subject(pov_viewer.cam, drone_pos, subj_pos)
                    pov_viewer.sync()
                except Exception as exc:
                    print(f"[POV viewer] {exc}", file=sys.stderr)
        heartbeat_every = max(1, sim.control_freq // STATUS_HEARTBEAT_HZ)
        if step % heartbeat_every == 0:
            yield ("heartbeat", t, None)
    if pov_viewer is not None:
        try:
            pov_viewer.close()
        except Exception:
            pass
    sim.close()
    print("Done.")
    yield ("done", float(waypoints["time"][0, -1]), None)


# ---------- presentation ----------

def render_roles_markdown(parsed: dict) -> str:
    motion = parsed.get("subject", {}).get("motion", "static")
    head = [
        f"**Mode:** `{parsed['mode']}` &nbsp;&nbsp; "
        f"**Strategy:** `{parsed['strategy']}` &nbsp;&nbsp; "
        f"**Subject:** `{motion}` &nbsp;&nbsp; "
        f"**Duration:** {parsed.get('duration', DEFAULT_DURATION_S):.1f}s",
        "",
        f"_{parsed.get('rationale', '') or '(no rationale)'}_",
        "",
    ]
    if parsed["mode"] == "freeform":
        rows = ["| Drone | Role | # waypoints | bbox x | bbox y | bbox z |",
                "|---|---|---|---|---|---|"]
        for a in parsed["assignments"]:
            wps = a.get("waypoints", [])
            if wps:
                xs = [float(w.get("x", 0)) for w in wps if isinstance(w, dict)]
                ys = [float(w.get("y", 0)) for w in wps if isinstance(w, dict)]
                zs = [float(w.get("z", 0)) for w in wps if isinstance(w, dict)]
                bx = f"[{min(xs):.2f}, {max(xs):.2f}]" if xs else "—"
                by = f"[{min(ys):.2f}, {max(ys):.2f}]" if ys else "—"
                bz = f"[{min(zs):.2f}, {max(zs):.2f}]" if zs else "—"
            else:
                bx = by = bz = "—"
            rows.append(f"| {a['drone']} | `{a['role']}` | {len(wps)} | {bx} | {by} | {bz} |")
        return "\n".join(head + rows)

    rows = ["| Drone | Role | Primitive | Params |", "|---|---|---|---|"]
    for a in parsed["assignments"]:
        prim = a["primitive"]
        p = a.get("params", {})
        if prim == "orbit":
            params_md = (f"r={p['radius']:.2f}m, h={p['height']:.2f}m, "
                         f"v={p['speed']:.2f}m/s, c=[{p['center'][0]:.2f}, {p['center'][1]:.2f}]")
        elif prim == "follow":
            params_md = (f"offset=[{p['offset'][0]:.2f}, {p['offset'][1]:.2f}], "
                         f"h={p['height']:.2f}m")
        else:
            params_md = "—"
        rows.append(f"| {a['drone']} | `{a['role']}` | `{prim}` | {params_md} |")
    return "\n".join(head + rows)


def print_summary(parsed: dict) -> None:
    motion = parsed.get("subject", {}).get("motion", "static")
    print(f"Mode: {parsed['mode']} | Strategy: {parsed['strategy']} | Subject: {motion} "
          f"| Duration: {parsed.get('duration', DEFAULT_DURATION_S):.1f}s")
    if parsed.get("rationale"):
        print(f"Rationale: {parsed['rationale']}")
    for a in parsed["assignments"]:
        prim = a["primitive"]
        if prim == "freeform":
            print(f"  drone {a['drone']}  role={a['role']:<18}  freeform: {len(a.get('waypoints', []))} waypoints")
        elif prim == "orbit":
            p = a["params"]
            print(f"  drone {a['drone']}  role={a['role']:<18}  orbit  "
                  f"r={p['radius']:.2f}m  h={p['height']:.2f}m  v={p['speed']:.2f}m/s  "
                  f"c=[{p['center'][0]:.2f},{p['center'][1]:.2f}]")
        elif prim == "follow":
            p = a["params"]
            print(f"  drone {a['drone']}  role={a['role']:<18}  follow "
                  f"offset=[{p['offset'][0]:.2f},{p['offset'][1]:.2f}]  h={p['height']:.2f}m")


# ---------- entrypoints ----------

def plan(text: str) -> dict:
    """Take a natural-language command, return a normalized plan (with fallbacks applied)."""
    raw, err = parse_command(text)
    if raw is None:
        return follow_fallback(err or "LLM call failed")
    try:
        normalized = normalize(raw)
    except Exception as exc:
        return follow_fallback(f"normalize: {exc}")
    final, _ = validate_freeform_or_fallback(normalized)
    return final


def execute(text: str, settings: dict, gui: bool) -> None:
    print(f"Parsing: {text!r}")
    parsed = plan(text)
    print_summary(parsed)
    waypoints = build_waypoints(parsed)
    motion = parsed.get("subject", {}).get("motion", "figure_eight")
    for _evt in run_sim(waypoints, settings, gui=gui, subject_motion=motion):
        pass


def encode_pov_mp4(frames: list[np.ndarray], fps: int = POV_FPS) -> str | None:
    if not frames:
        return None
    path = f"/tmp/dolly_pov_{os.getpid()}.mp4"
    try:
        iio.imwrite(path, np.stack(frames, axis=0), fps=fps, codec="libx264")
        return path
    except Exception as exc:
        print(f"[POV encode] {exc}", file=sys.stderr)
        return None


def launch_ui(settings: dict) -> None:
    def submit(text: str, show_viewer: bool, pov_drone: str):
        text = text.strip()
        if not text:
            yield "", "", "Type a command first.", None
            return
        try:
            parsed = plan(text)
        except Exception as exc:
            yield "", "", f"Plan error: {exc}", None
            return
        roles_md = render_roles_markdown(parsed)
        parsed_json = json.dumps(parsed, indent=2)
        yield roles_md, parsed_json, "Planning complete; starting sim...", None

        if pov_drone in (None, "", "off"):
            pov_idx = None
        else:
            try:
                pov_idx = int(pov_drone)
            except (TypeError, ValueError):
                pov_idx = None
        effective_gui = show_viewer

        try:
            waypoints = build_waypoints(parsed)
            motion = parsed.get("subject", {}).get("motion", "figure_eight")
            for evt in run_sim(
                waypoints, settings,
                gui=effective_gui, subject_motion=motion, pov_drone_idx=pov_idx,
            ):
                kind, t, _payload = evt
                if kind == "heartbeat":
                    yield roles_md, parsed_json, f"t={t:.1f}s", None
                elif kind == "done":
                    yield roles_md, parsed_json, "Done.", None
        except Exception as exc:
            yield roles_md, parsed_json, f"Sim error: {exc}", None
            return

    with gr.Blocks(title="dolly MVP") as ui:
        gr.Markdown("# dolly MVP — central-planner swarm cinematography")
        with gr.Row():
            cmd_in = gr.Textbox(
                label="Command",
                placeholder='Try: "cover this play" or "circle around them slowly"',
                scale=4,
            )
            submit_btn = gr.Button("Run", variant="primary", scale=1)
        with gr.Row():
            viewer_chk = gr.Checkbox(value=True, label="Open MuJoCo 3D viewer window")
            pov_radio = gr.Radio(
                choices=[("Scene (default)", "off"), ("Drone 0 POV", "0"),
                         ("Drone 1 POV", "1"), ("Drone 2 POV", "2")],
                value="off",
                label="MuJoCo viewer camera (camera always points at the subject)",
            )
        roles_out = gr.Markdown(label="Director's plan")
        with gr.Row():
            parsed_out = gr.Code(label="Parsed JSON (debug)", language="json")
            status_out = gr.Textbox(label="Status", lines=4)
        pov_out = gr.Video(visible=False)
        outputs = [roles_out, parsed_out, status_out, pov_out]
        submit_btn.click(submit, inputs=[cmd_in, viewer_chk, pov_radio], outputs=outputs)
        cmd_in.submit(submit, inputs=[cmd_in, viewer_chk, pov_radio], outputs=outputs)
    ui.launch()


def argv_command() -> str:
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    return " ".join(positional).strip()


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
