"""Measure peak memory of a full analysis run.

Deployment targets have hard RAM caps (Streamlit Community Cloud: 1GB;
Render free: 512MB), and MediaPipe/OpenCV allocate in C extensions where
tracemalloc is blind — so sample RSS from a background thread instead.

Uses a realistic phone clip: 1080x1920 portrait, 60fps, 4s.
"""
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

FPS, SECONDS = 60, 4
W, H = 1080, 1920  # portrait, as a phone actually records


def make_clip(path):
    """Textured moving content -- blank frames would under-read decode cost."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 60, (H, W, 3), dtype=np.uint8)
    for i in range(FPS * SECONDS):
        frame = np.full((H, W, 3), 90, np.uint8) + noise
        cv2.rectangle(frame, (400, 600 - i * 2), (700, 1500), (200, 180, 160), -1)
        cv2.circle(frame, (550, 400 + i), 40, (240, 240, 60), -1)
        writer.write(frame)
    writer.release()


class PeakSampler(threading.Thread):
    """Sample RSS of this process plus children.

    The H.264 re-encode runs ffmpeg as a subprocess: its memory lives outside
    our RSS but still counts against a container's cap, so include children.
    """
    daemon = True

    def __init__(self):
        super().__init__()
        self.proc = psutil.Process()
        self.peak = 0
        self.stop_flag = threading.Event()

    def _total(self):
        total = self.proc.memory_info().rss
        for child in self.proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
        return total

    def run(self):
        while not self.stop_flag.is_set():
            try:
                self.peak = max(self.peak, self._total())
            except psutil.Error:
                break
            time.sleep(0.02)


def mb(n):
    return n / (1024 * 1024)


def _run_full_pipeline(clip, out_dir):
    """Force every stage to run, including rendering.

    MediaPipe finds no player in a synthetic clip, so the pipeline would bail
    before the render stages -- which are the memory-heavy ones (frame-by-frame
    overlay write, ffmpeg re-encode, matplotlib). Run real pose extraction to
    measure its true cost, then swap in synthetic keypoints so the render path
    executes against the real 1080x1920 video.
    """
    from analyzer.pose import extract_keypoints
    from analyzer.phases import smooth_keypoints, detect_phases
    from analyzer.metrics import compute_metrics
    from analyzer.report import render_assets, write_report
    import test_synthetic as ts

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        extract_keypoints(clip)  # real cost of the pose stage on a real clip
        note = "pose stage tracked a player"
    except ValueError:
        note = "pose stage found no player (expected on synthetic frames)"

    # Synthetic keypoints, scaled from the 640x360 test canvas to this clip.
    kp = ts.synthetic_serve()
    kp[:, :, 0] *= W / 640.0
    kp[:, :, 1] *= H / 360.0
    kp = smooth_keypoints(kp, ts.FPS)
    phases, hand, _ = detect_phases(kp, ts.FPS)
    metrics, series = compute_metrics(kp, phases, hand, ts.FPS)
    assets = render_assets(clip, kp, phases, hand, out_dir, ts.FPS)
    write_report(out_dir, metrics, phases, hand, [], ts.FPS, assets, series)
    return f"full pipeline incl. render ({note})"


def main():
    proc = psutil.Process()
    base = proc.memory_info().rss
    print(f"baseline (bare interpreter):        {mb(base):7.1f} MB")

    from analyzer.pipeline import analyze  # imported late to attribute lib cost
    after_import = proc.memory_info().rss
    print(f"after importing analyzer stack:     {mb(after_import):7.1f} MB")

    tmp = Path(tempfile.mkdtemp())
    clip = tmp / "clip.mp4"
    make_clip(clip)
    print(f"test clip: {W}x{H} @ {FPS}fps, {SECONDS}s "
          f"({clip.stat().st_size / 1e6:.1f} MB on disk)")

    sampler = PeakSampler()
    sampler.start()
    t0 = time.time()
    outcome = _run_full_pipeline(clip, tmp / "out")
    elapsed = time.time() - t0
    sampler.stop_flag.set()
    sampler.join(timeout=1)

    peak = max(sampler.peak, proc.memory_info().rss)
    print(f"\nanalysis {outcome} in {elapsed:.1f}s")
    print(f"PEAK RSS during analysis:           {mb(peak):7.1f} MB")
    print(f"  ({mb(peak - base):.1f} MB above baseline)")

    print("\nfit against free-tier caps:")
    for name, cap in (("Streamlit Community Cloud", 1024),
                      ("Render free web service", 512),
                      ("HF Spaces CPU Basic", 16 * 1024)):
        head = cap - mb(peak)
        verdict = "FITS" if head > 150 else ("TIGHT" if head > 0 else "EXCEEDS")
        print(f"  {name:28s} {cap:5d} MB -> {verdict} ({head:+.0f} MB headroom)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
