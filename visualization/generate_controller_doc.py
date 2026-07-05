"""Generate docs/controller_summary.pdf — a concise reference for the
force-control stack (state machine, controllers, FT pipeline, kinematics).

Run once:
    python visualization/generate_controller_doc.py
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

W, H = A4
MARGIN = 2.0 * cm

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
OUT_PATH = os.path.join(OUT_DIR, "controller_summary.pdf")


# ── styles ─────────────────────────────────────────────────────────────────

def _styles():
    base = getSampleStyleSheet()
    mono = "Courier"
    sans = "Helvetica"

    s = {}
    s["title"] = ParagraphStyle("title", fontName=sans+"-Bold",
                                fontSize=18, spaceAfter=6, alignment=TA_CENTER)
    s["subtitle"] = ParagraphStyle("subtitle", fontName=sans,
                                   fontSize=11, spaceAfter=14, alignment=TA_CENTER,
                                   textColor=colors.HexColor("#555555"))
    s["h1"] = ParagraphStyle("h1", fontName=sans+"-Bold", fontSize=13,
                              spaceBefore=14, spaceAfter=4,
                              textColor=colors.HexColor("#1a1a2e"))
    s["h2"] = ParagraphStyle("h2", fontName=sans+"-Bold", fontSize=10,
                              spaceBefore=8, spaceAfter=2,
                              textColor=colors.HexColor("#16213e"))
    s["body"] = ParagraphStyle("body", fontName=sans, fontSize=9,
                                leading=14, spaceAfter=4)
    s["code"] = ParagraphStyle("code", fontName=mono, fontSize=8,
                                leading=12, spaceAfter=2,
                                textColor=colors.HexColor("#2c3e50"),
                                backColor=colors.HexColor("#f4f4f4"),
                                leftIndent=12, rightIndent=12,
                                borderPad=4)
    s["bullet"] = ParagraphStyle("bullet", fontName=sans, fontSize=9,
                                  leading=13, leftIndent=14,
                                  bulletIndent=4, spaceAfter=2)
    return s


def P(text, style):
    return Paragraph(text, style)

def B(text, style):
    return Paragraph(f"• {text}", style)

def HR():
    return HRFlowable(width="100%", thickness=0.5,
                      color=colors.HexColor("#cccccc"), spaceAfter=6)

def table(rows, col_widths, header=True):
    t = Table(rows, colWidths=col_widths)
    style = [
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
         [colors.HexColor("#f9f9f9"), colors.white]),
        ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#dddddd")),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",(0, 0), (-1, -1), 6),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0,0), (-1,-1),  3),
    ]
    if header:
        style += [
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ]
    t.setStyle(TableStyle(style))
    return t


# ── content builders ────────────────────────────────────────────────────────

def build(s):
    story = []

    # ── Title ────────────────────────────────────────────────────────────
    story += [
        Spacer(1, 1.2*cm),
        P("Force-Control Stack — Controller Summary", s["title"]),
        P("Franka Panda · MuJoCo sim + real hardware", s["subtitle"]),
        HR(),
        Spacer(1, 0.3*cm),
    ]

    # ── 1. Overview ──────────────────────────────────────────────────────
    story += [
        P("1. Overview", s["h1"]),
        P("""The stack runs a 1 kHz control loop on the Franka Panda 7-DoF arm.
A state machine sequences the robot through distinct phases; each phase selects
either <b>joint-PD</b> or <b>Cartesian impedance</b> as the active controller.
An admittance outer loop on the world-z force regulates contact during the
force-hold and slide phases.""", s["body"]),
        Spacer(1, 0.2*cm),
    ]

    # ── 2. State Machine ─────────────────────────────────────────────────
    story += [
        P("2. State Machine  <font size='8' color='#888888'>(core/state_machine.py · JointMoveStateMachine)</font>", s["h1"]),
        P("""User interaction (Enter key presses) gates transitions between phases.
Each phase's <i>update()</i> call returns a <b>ControlCommand</b> with
<code>mode ∈ {joint, cartesian, failed}</code> that the backend dispatches to
the appropriate controller.""", s["body"]),
        Spacer(1, 0.15*cm),
    ]

    phase_rows = [
        ["Phase", "Controller", "Entry condition", "Exit condition"],
        ["HOLD", "Joint PD\n(q_hold)", "Boot / IK not yet solved", "User presses Enter"],
        ["WAIT_USER_CONFIRM", "Joint PD\n(q_hold)", "IK solved", "User presses Enter"],
        ["MOVE_JOINT", "Joint PD\n(min-jerk)", "Enter pressed", "t_elapsed ≥ move_time"],
        ["TARE", "Joint PD\n(q_goal)", "Move complete", "tare_duration elapsed"],
        ["WAIT_APPROACH_CONFIRM", "Joint PD\n(q_goal)", "Tare complete", "User presses Enter"],
        ["APPROACH", "Cartesian\n(descent)", "Enter pressed", "Fz ≥ threshold\n(5 consecutive)"],
        ["FORCE_HOLD", "Cartesian\n(admittance)", "Contact detected", "User presses Enter\n(slide enabled)"],
        ["WAIT_SLIDE_CONFIRM", "Cartesian\n(admittance)", "FORCE_HOLD entered", "User presses Enter"],
        ["SLIDE", "Cartesian\n(admittance +\nxy trajectory)", "Enter pressed", "distance covered"],
        ["FAILED", "—", "IK fail / missing pose", "Loop terminates"],
    ]
    cw = [3.2*cm, 2.8*cm, 4.5*cm, 4.5*cm]
    story.append(KeepTogether([table(phase_rows, cw)]))
    story.append(Spacer(1, 0.3*cm))

    # ── 3. Joint PD Controller ───────────────────────────────────────────
    story += [
        P("3. Joint PD Controller  <font size='8' color='#888888'>(core/controller.py · JointPDController)</font>", s["h1"]),
        P("Used during HOLD, MOVE_JOINT, TARE, and WAIT phases. Stateless — no integrator.", s["body"]),
        P("τ = Kp · (q_des − q) + Kd · (dq_des − dq)", s["code"]),
        Spacer(1, 0.1*cm),
    ]
    pd_rows = [
        ["Parameter", "Sim value", "Units", "Notes"],
        ["Kp  (joints 1–4)", "200, 200, 200, 200", "Nm/rad", "Position gain"],
        ["Kp  (joints 5–7)", "80, 80, 40", "Nm/rad", "Distal joints, softer"],
        ["Kd  (joints 1–4)", "20, 20, 20, 20", "Nm·s/rad", "Velocity gain"],
        ["Kd  (joints 5–7)", "8, 8, 4", "Nm·s/rad", ""],
    ]
    story.append(table(pd_rows, [4.5*cm, 4*cm, 2.5*cm, 4*cm]))
    story.append(Spacer(1, 0.15*cm))
    story += [
        P("Min-jerk trajectory (MOVE_JOINT)", s["h2"]),
        P("Position setpoints follow a 5<sup>th</sup>-order polynomial that produces "
          "zero velocity and acceleration at both endpoints:", s["body"]),
        P("s(t) = 10τ³ − 15τ⁴ + 6τ⁵,   τ = t / T", s["code"]),
        P("ṡ(t) = (30τ² − 60τ³ + 30τ⁴) / T", s["code"]),
        Spacer(1, 0.2*cm),
    ]

    # ── 4. Cartesian Impedance Controller ────────────────────────────────
    story += [
        P("4. Cartesian Impedance Controller  <font size='8' color='#888888'>(core/controller.py · CartesianImpedanceController)</font>", s["h1"]),
        P("""Active during APPROACH, FORCE_HOLD, WAIT_SLIDE_CONFIRM, and SLIDE.
No gravity/coriolis term is added here — the backend handles those separately.""", s["body"]),
        P("F = Kp_pos · (x_des − x)  +  Kd_pos · (dx_des − v)", s["code"]),
        P("T = Kp_ori · e_ori         +  Kd_ori · (w_des − w)", s["code"]),
        P("τ = Jᵀ · [F, T]", s["code"]),
        Spacer(1, 0.1*cm),
    ]
    ci_rows = [
        ["DOF", "Kp (K)", "Kd (D)", "Units"],
        ["x, y, z  (translation)", "500, 500, 500", "50, 50, 50", "N/m ; N·s/m"],
        ["rx, ry, rz  (rotation)", "30, 30, 30", "5, 5, 5", "Nm/rad ; Nm·s/rad"],
    ]
    story.append(table(ci_rows, [5*cm, 3.5*cm, 3.5*cm, 3*cm]))
    story += [
        Spacer(1, 0.15*cm),
        P("Orientation error  (world frame):", s["h2"]),
        P("Re = R_des · Rᵀ,   e_ori = 0.5 · vee(Re − Reᵀ)", s["code"]),
        Spacer(1, 0.15*cm),
        P("Gravity & payload compensation (added by backend, not controller):", s["h2"]),
        P("""<b>Sim:</b>  τ_grav = G_zero(q)  (arm-only model) + ΔG_payload(q)  (full − zero model delta).<br/>
<b>Hardware:</b>  libfranka handles arm gravity internally; backend adds ΔG_payload(q) only.""", s["body"]),
        P("τ_total = τ_grav + τ_task  +  null-space term", s["code"]),
        P("Torque rate is clamped to  max_torque_rate = 1.0 Nm per 1 ms tick.", s["body"]),
        Spacer(1, 0.2*cm),
    ]

    # ── 5. Force Admittance (contact phases) ─────────────────────────────
    story += [
        P("5. Contact Force Regulation  <font size='8' color='#888888'>(core/state_machine.py · _update_force_z)</font>", s["h1"]),
        P("""An admittance outer loop converts Fz error to a z-reference velocity,
which is fed as <i>dx_des[2]</i> and <i>x_des[2] = z_ref</i> to the Cartesian
impedance controller.  Separate gain sets are used for FORCE_HOLD vs SLIDE.""", s["body"]),
        P("e  = Fz − F_desired", s["code"]),
        P("ż  = Kp_f · e + Kd_f · Δe/dt      (clamped to ±max_z_speed)", s["code"]),
        P("z_ref  +=  ż · dt             (clamped to [z_contact − max_depth, z_contact + retract])", s["code"]),
        Spacer(1, 0.1*cm),
    ]
    fz_rows = [
        ["Parameter", "FORCE_HOLD (sim)", "SLIDE (sim)", "Units"],
        ["F_desired", "3.0", "3.0", "N"],
        ["Kp_f", "0.0005", "0.05", "m/s per N"],
        ["Kd_f", "0.0", "0.005", "m per N"],
        ["max_z_speed", "0.0015", "0.05", "m/s"],
        ["max_depth", "0.05", "0.05", "m  (below z_contact)"],
        ["retract_allowance", "0.01", "0.01", "m  (above z_contact)"],
    ]
    story.append(table(fz_rows, [4*cm, 3.5*cm, 3.5*cm, 4*cm]))
    story.append(Spacer(1, 0.2*cm))

    # ── 6. Approach Phase ────────────────────────────────────────────────
    story += [
        P("6. Approach Phase  <font size='8' color='#888888'>(_update_approach)</font>", s["h1"]),
        P("Open-loop descent along world −Z at constant speed until contact is confirmed.", s["body"]),
    ]
    ap_rows = [
        ["Parameter", "Sim value", "Units"],
        ["approach_speed", "0.05", "m/s"],
        ["contact_fz_threshold", "2.0", "N  (Fz must exceed this)"],
        ["contact_confirm_samples", "5", "consecutive ticks above threshold"],
    ]
    story.append(table(ap_rows, [5*cm, 3.5*cm, 6.5*cm]))
    story.append(Spacer(1, 0.2*cm))

    # ── 7. Slide Phase ───────────────────────────────────────────────────
    story += [
        P("7. Slide Phase  <font size='8' color='#888888'>(_update_slide)</font>", s["h1"]),
        P("""The slide runs simultaneous z force-hold (SLIDE gains) and a constant-speed
xy trajectory with a min-jerk velocity ramp.  A tangential velocity-error
integral corrects for friction-induced lag.""", s["body"]),
        P("s_ref(t)  = clamp(∫ v_ref dt,  0, distance)", s["code"]),
        P("v_ref(t)  = speed · minJerk(t − t0, ramp_time)  [ramps from 0 to speed]", s["code"]),
        P("v_cmd     = v_ref + ki_vel · ∫(v_ref − v_actual) dt   (clamped)", s["code"]),
        P("x_des     = p_start + u_slide · s_ref", s["code"]),
        Spacer(1, 0.1*cm),
    ]
    sl_rows = [
        ["Parameter", "Sim value", "Units"],
        ["direction_xy", "[0, 1]", "world-frame XY (auto-normalised)"],
        ["speed", "0.01", "m/s  (after ramp)"],
        ["ramp_time", "1.0", "s  (min-jerk ramp)"],
        ["distance", "0.1", "m  (then hold in place)"],
        ["ki_vel", "20.0", "integral gain on v-error"],
        ["vel_int_limit", "0.003", "m/s  (anti-windup)"],
    ]
    story.append(table(sl_rows, [5*cm, 3.5*cm, 6.5*cm]))
    story.append(Spacer(1, 0.2*cm))

    # ── 8. FT Processor ─────────────────────────────────────────────────
    story += [
        P("8. FT Sensor Pipeline  <font size='8' color='#888888'>(core/ft_processor.py · FTProcessor)</font>", s["h1"]),
        P("Three-stage pipeline applied each tick:", s["body"]),
        P("1.  Sign flip   : w_signed = sign · w_raw          (sign = −1 in sim)", s["code"]),
        P("2.  Bias remove : w_unbiased = w_signed − bias      (bias set by tare())", s["code"]),
        P("3.  Low-pass    : w_out = α · w_unbiased + (1−α) · w_prev   (α = 1.0 in sim = no filter)", s["code"]),
        P("""<b>Tare procedure:</b> after MOVE_JOINT completes the robot holds the goal pose for
<i>tare_duration</i> seconds (2 s in sim). All valid FT samples are averaged
and saved as the bias, zeroing out the tool's gravity wrench at that orientation.
All subsequent readings are pure contact force.""", s["body"]),
        Spacer(1, 0.2*cm),
    ]

    # ── 9. Kinematics ────────────────────────────────────────────────────
    story += [
        P("9. Forward Kinematics  <font size='8' color='#888888'>(core/kinematics.py · SiteKinematics)</font>", s["h1"]),
        P("""Uses a private scratch MjData driven with measured joint positions.
Reads the world-frame pose and spatial Jacobian of the controlled site
(<i>stick_tip</i>) via the minimal MuJoCo pipeline (no dynamics, no collision).""", s["body"]),
        P("mj_kinematics(model, data)   →  site_xpos, site_xmat", s["code"]),
        P("mj_comPos(model, data)        →  cdof terms needed by mj_jacSite", s["code"]),
        P("mj_jacSite(...)               →  Jp (3×7), Jr (3×7)", s["code"]),
        P("v = Jp · dq,   w = Jr · dq,   J = stack([Jp, Jr])", s["code"]),
        Spacer(1, 0.2*cm),
    ]

    # ── 10. IK Solver ────────────────────────────────────────────────────
    story += [
        P("10. IK Solver  <font size='8' color='#888888'>(core/ik.py · solve_ik_dls)</font>", s["h1"]),
        P("""Damped Least Squares (DLS / Levenberg–Marquardt) IK run once at startup
on a scratch MjData — the live simulation state is never corrupted.""", s["body"]),
        P("e = [x_des − x_tip ; e_ori]         (6-vector pose error)", s["code"]),
        P("Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ e          (DLS step)", s["code"]),
        P("q ← q + Δq,   iterate until ‖e‖ < tol", s["code"]),
        P("""Target orientation: <i>R_ref</i> (site at home keyframe) rotated by
<i>target_euler_xyz = [30°, 0°, 0°]</i> expressed as intrinsic XYZ Euler angles.""", s["body"]),
        Spacer(1, 0.2*cm),
    ]

    # ── 11. Data Logging ─────────────────────────────────────────────────
    story += [
        P("11. Trajectory Logging  <font size='8' color='#888888'>(core/data_logger.py · TrajectoryLogger)</font>", s["h1"]),
        P("""RT-safe: a fixed NumPy buffer (<i>capacity = 1 500 000 rows</i> ≈ 25 min at 1 kHz)
is preallocated at startup. Each tick writes one row by index — no allocation
or disk I/O in the hot path. The CSV is flushed once after the loop exits
(or on Ctrl+C via try/finally).""", s["body"]),
        Spacer(1, 0.1*cm),
    ]
    log_rows = [
        ["Column", "Description", "Unit"],
        ["t", "Simulation / wall time since loop start", "s"],
        ["x, y, z", "Tip world-frame position (stick_tip site)", "m"],
        ["vx, vy, vz", "Tip world-frame linear velocity  (Jp · dq)", "m/s"],
        ["Fz", "Processed contact force, world z  (wrench[2])", "N"],
        ["phase", "State machine phase name (e.g. FORCE_HOLD)", "string"],
    ]
    story.append(table(log_rows, [3*cm, 8*cm, 4*cm]))
    story.append(Spacer(1, 0.2*cm))

    # ── 12. File Map ─────────────────────────────────────────────────────
    story += [
        P("12. Key File Map", s["h1"]),
    ]
    fm_rows = [
        ["File", "Role"],
        ["core/state_machine.py", "JointMoveStateMachine — all phase transitions & force loop"],
        ["core/controller.py", "JointPDController, CartesianImpedanceController"],
        ["core/ft_processor.py", "FTProcessor — sign / bias / low-pass pipeline"],
        ["core/kinematics.py", "SiteKinematics — FK + Jacobian via scratch MjData"],
        ["core/data_logger.py", "TrajectoryLogger — RT-safe per-tick CSV writer"],
        ["core/ik.py", "solve_ik_dls — DLS IK solver, min_jerk, euler helpers"],
        ["backends/mujoco_backend.py", "MuJoCo sim: viewer loop, FT read, log flush on exit"],
        ["backends/franka_backend.py", "Real robot: 1 kHz RT loop, ROS/SHM FT, safety limits"],
        ["sensors/ft_mujoco.py", "FTMuJoCo — reads named MuJoCo force/torque sensors"],
        ["configs/sim.yaml", "All tunable parameters for the simulation run"],
        ["visualization/plot_log.py", "Post-run analysis: 5 matplotlib figures from CSV log"],
        ["visualization/live_monitor.py", "Watches data/logs/ and auto-plots when new CSV appears"],
    ]
    story.append(table(fm_rows, [6*cm, 10*cm]))

    return story


# ── main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    s = _styles()
    doc = SimpleDocTemplate(
        OUT_PATH,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=MARGIN,
        title="Force-Control Stack — Controller Summary",
        author="force_control",
    )
    doc.build(build(s))
    print(f"written → {OUT_PATH}")


if __name__ == "__main__":
    main()
