"""Report assets: annotated video, phase stills, trajectory chart, HTML page."""
import html
import subprocess
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .pose import LM, POSE_CONNECTIONS, open_video, scale_frame, apply_rotation
from .metrics import hitting_arm, joint_angle

SKELETON_COLOR = (80, 220, 80)
JOINT_COLOR = (60, 160, 255)
PHASE_LABELS = {"toss_peak": "TOSS PEAK", "trophy": "TROPHY POSITION",
                "contact": "CONTACT", "follow_through": "FOLLOW-THROUGH"}


def _draw_skeleton(frame, pts):
    for a, b in POSE_CONNECTIONS:
        pa, pb = pts[a], pts[b]
        if pa[2] > 0.3 and pb[2] > 0.3 and not np.isnan(pa[0]) and not np.isnan(pb[0]):
            cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     SKELETON_COLOR, 2, cv2.LINE_AA)
    for p in pts:
        if p[2] > 0.3 and not np.isnan(p[0]):
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, JOINT_COLOR, -1, cv2.LINE_AA)


def _banner(frame, text):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (30, 30, 30), -1)
    cv2.putText(frame, text, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)


def _angle_callout(frame, pts, joint, angle_deg, label):
    if np.isnan(angle_deg) or np.isnan(pts[joint][0]):
        return
    x, y = int(pts[joint][0]), int(pts[joint][1])
    cv2.circle(frame, (x, y), 10, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{label}: {angle_deg:.0f} deg", (x + 14, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)


def _reencode_h264(src, dst):
    """Re-encode to H.264 so the report video plays in a browser."""
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg, "-y", "-i", str(src), "-c:v", "libx264", "-preset", "fast",
         "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
         "-movflags", "+faststart", "-an", str(dst)],
        check=True, capture_output=True)


def render_assets(video_path, kp, phases, handedness, out_dir, fps):
    """Write overlay video + phase stills; return their file names."""
    out_dir = Path(out_dir)
    cap, _, rotation = open_video(video_path)
    raw_path = out_dir / "overlay_raw.mp4"
    writer = None
    shoulder, elbow, wrist = hitting_arm(handedness)
    frame_to_phase = {f: name for name, f in phases.items()}
    stills = {}
    hold = max(2, int(0.15 * fps))  # frames to keep each phase banner visible

    i = 0
    while i < len(kp):
        ok, frame = cap.read()
        if not ok:
            break
        frame = apply_rotation(frame, rotation)
        frame = scale_frame(frame)
        if writer is None:
            writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (frame.shape[1], frame.shape[0]))
        pts = kp[i]
        _draw_skeleton(frame, pts)
        for name, f in phases.items():
            if abs(i - f) <= hold:
                _banner(frame, PHASE_LABELS[name])
        if i in frame_to_phase:
            name = frame_to_phase[i]
            still = frame.copy()
            if name == "contact":
                ang = joint_angle(pts[shoulder][:2], pts[elbow][:2], pts[wrist][:2])
                _angle_callout(still, pts, elbow, ang, "elbow")
            elif name == "trophy":
                for hip, knee, ankle in ((LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE),
                                         (LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE)):
                    ang = joint_angle(pts[hip][:2], pts[knee][:2], pts[ankle][:2])
                    _angle_callout(still, pts, knee, ang, "knee")
            cv2.imwrite(str(out_dir / f"{name}.jpg"), still,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            stills[name] = f"{name}.jpg"
        writer.write(frame)
        i += 1
    cap.release()
    if writer:
        writer.release()

    # OpenCV writes mp4v, which browsers cannot decode in a <video> tag, so
    # the H.264 re-encode is what makes the clip playable at all. Falling back
    # silently would ship a video element that loads and then plays nothing --
    # report the failure so the caller can surface it.
    video_name, h264 = "overlay.mp4", True
    try:
        _reencode_h264(raw_path, out_dir / video_name)
        raw_path.unlink()
    except Exception:
        video_name, h264 = "overlay_raw.mp4", False
    return {"video": video_name, "stills": stills, "h264": h264}


def render_chart(series, phases, fps, out_dir):
    t = np.asarray(series["t"])
    fig, ax1 = plt.subplots(figsize=(9, 3.6), dpi=110)
    ax1.plot(t, series["hit_wrist"], color="#1D9E75", label="hitting wrist height")
    ax1.plot(t, series["toss_wrist"], color="#378ADD", label="tossing wrist height")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("height above ground\n(× nose-to-ankle length)")
    ax2 = ax1.twinx()
    ax2.plot(t, series["knee"], color="#D85A30", alpha=0.7, label="knee angle")
    ax2.set_ylabel("knee interior angle (°)", color="#D85A30")
    for name, f in phases.items():
        if name == "follow_through":
            continue
        ax1.axvline(f / fps, color="#888780", linestyle=":", linewidth=1)
        ax1.text(f / fps, ax1.get_ylim()[1], PHASE_LABELS[name].lower(),
                 rotation=90, va="top", ha="right", fontsize=7, color="#5F5E5A")
    ax1.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "chart.png")
    plt.close(fig)
    return "chart.png"


STATUS_STYLE = {
    "good": ("#EAF3DE", "#3B6D11", "looks good"),
    "info": ("#F1EFE8", "#444441", "for reference"),
    "warn": ("#FAEEDA", "#854F0B", "worth working on"),
}

CSS = """
html { background: #f4f4f0; }
body { font-family: system-ui, sans-serif; max-width: 880px; margin: 24px auto;
       padding: 0 16px; color: #222; }
h1 { font-size: 1.5rem; } h2 { font-size: 1.1rem; margin-top: 28px; }
.meta { color: #666; font-size: 0.9rem; }
.warn-box { background: #FAEEDA; border: 1px solid #EF9F27; border-radius: 8px;
            padding: 10px 14px; margin: 14px 0; font-size: 0.92rem; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
         gap: 12px; margin: 14px 0; }
.card { border-radius: 10px; padding: 14px; }
.card .value { font-size: 1.6rem; font-weight: 700; }
.card .label { font-weight: 600; margin-bottom: 4px; }
.card .tag { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; }
.card .note, .card .guide { font-size: 0.85rem; margin-top: 6px; }
.card .guide { opacity: 0.75; }
.card .src { font-size: 0.75rem; margin-top: 6px; opacity: 0.7; }
.card .src a { color: inherit; }
.stills { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 10px; }
.stills figure { margin: 0; } .stills img { width: 100%; border-radius: 8px; }
.stills figcaption { font-size: 0.85rem; color: #555; text-align: center; }
video, .chart img { width: 100%; border-radius: 10px; }
.footnote { color: #777; font-size: 0.82rem; margin: 26px 0; border-top: 1px solid #ddd;
            padding-top: 10px; }
"""


def write_report(out_dir, metrics, phases, handedness, warnings, fps, assets, series):
    out_dir = Path(out_dir)
    # run_analysis renders the chart so both frontends get it; render here only
    # when write_report is called directly (tests, the CLI path).
    chart = assets.get("chart") or render_chart(series, phases, fps, out_dir)

    cards = ""
    for m in metrics:
        bg, fg, tag = STATUS_STYLE[m["status"]]
        src = m.get("source")
        src_html = (f'<div class="src"><a href="{html.escape(src["url"])}" '
                    f'target="_blank" rel="noopener">source: '
                    f'{html.escape(src["cite"])}</a></div>') if src else ""
        cards += f"""
        <div class="card" style="background:{bg};color:{fg};">
          <div class="label">{html.escape(m['label'])}</div>
          <div class="value">{html.escape(m['display'])}</div>
          <div class="tag">{tag}</div>
          <div class="note">{html.escape(m['note'])}</div>
          <div class="guide">{html.escape(m['guideline'])}</div>
          {src_html}
        </div>"""

    stills_html = ""
    for name in ("toss_peak", "trophy", "contact"):
        if name in assets["stills"]:
            secs = phases[name] / fps
            stills_html += f"""
            <figure><img src="{assets['stills'][name]}" alt="{name}">
            <figcaption>{PHASE_LABELS[name].title()} — {secs:.2f}s</figcaption></figure>"""

    warn_html = ""
    if warnings:
        items = "".join(f"<li>{html.escape(w)}</li>" for w in warnings)
        warn_html = f'<div class="warn-box"><ul style="margin:0;padding-left:18px;">{items}</ul></div>'

    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Serve analysis report</title><style>{CSS}</style></head>
<body>
<h1>Serve analysis report</h1>
<p class="meta">Detected a {handedness}-handed serve &middot; footage {fps:.0f}fps
&middot; contact at {phases['contact'] / fps:.2f}s</p>
{warn_html}
<h2>Metrics</h2>
<div class="cards">{cards}</div>
<h2>Key positions</h2>
<div class="stills">{stills_html}</div>
<h2>Annotated video</h2>
<video src="{assets['video']}" controls muted playsinline></video>
<h2>Motion timeline</h2>
<div class="chart"><img src="{chart}" alt="wrist heights and knee angle over time"></div>
<p class="footnote">How to read this: reference ranges are the mean &plusmn;1 SD
reported for <b>elite servers</b> (Olympic and skilled-player studies) &mdash;
a benchmark, not a target, and falling outside one is not automatically a
fault. Those studies used 3D motion capture; this tool measures 2D angles
projected onto a single camera plane, which under-reads any movement swinging
out of that plane, so treat the numbers as approximations. Contact height uses
an uncalibrated heuristic with no published equivalent. This is an analysis
prototype, not certified coaching advice.</p>
</body></html>"""
    report_path = out_dir / "report.html"
    report_path.write_text(page, encoding="utf-8")
    return report_path
