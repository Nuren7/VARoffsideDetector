import math
import time


def report_frame_progress(
    stage: str,
    current_frame: int,
    total_frames: int,
    started_at: float,
    force: bool = False,
) -> None:
    """Print visible, throttled progress updates while processing video frames."""
    interval = max(1, math.ceil(total_frames / 20)) if total_frames > 0 else 100
    if not force and current_frame != 1 and current_frame % interval != 0:
        return
    if force and current_frame > 0 and current_frame % interval == 0:
        return

    elapsed = time.perf_counter() - started_at
    if total_frames > 0 and current_frame > 0:
        percent = min(100, round(current_frame / total_frames * 100))
        estimated_total = elapsed / current_frame * total_frames
        remaining = max(0, estimated_total - elapsed)
        progress = (
            f"frame {current_frame}/{total_frames} ({percent}%)"
            f" | elapsed {elapsed:.1f}s | ETA {remaining:.1f}s"
        )
    elif total_frames > 0:
        progress = f"frame 0/{total_frames} (0%) | elapsed {elapsed:.1f}s"
    else:
        progress = f"frame {current_frame} | elapsed {elapsed:.1f}s"

    print(f"[processing] {stage}: {progress}", flush=True)
