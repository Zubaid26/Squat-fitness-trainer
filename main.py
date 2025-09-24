import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2, tempfile, shutil, time, glob
from pathlib import Path
import os
import torch

st.set_page_config(
    page_title="Video Pose Estimation + Squat Counter", layout="centered"
)
st.title("🏋️‍♂️ Squat Pose • Angle & Rep Counter (YOLO Pose)")

with st.expander("⚙️ Settings", expanded=True):
    model_name = st.selectbox(
        "Model",
        ["yolo11n-pose.pt", "yolo11s-pose.pt", "yolov8n-pose.pt", "yolov8s-pose.pt"],
        index=0,
        help="*n* = fastest; *s* = better accuracy.",
    )
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("IoU threshold (NMS)", 0.1, 0.95, 0.7, 0.05)
    vid_stride = st.number_input("Video frame stride (1 = every frame)", 1, 10, 1, 1)
    use_gpu = st.checkbox("Use GPU if available (CUDA)", value=True)

with st.expander("🧮 Rep logic (tune if needed)", expanded=False):
    up_threshold = st.slider(
        "Angle: stand-up threshold (°)",
        140,
        180,
        165,
        1,
        help="Above this knee angle counts as 'up' (standing).",
    )
    down_threshold = st.slider(
        "Angle: squat-bottom threshold (°)",
        60,
        140,
        100,
        1,
        help="Below this knee angle counts as 'down' (bottom).",
    )
    require_hip_below_knee = st.checkbox(
        "Require hip lower than knee at bottom",
        value=True,
        help="Stricter depth criterion.",
    )
    kp_min_conf = st.slider("Min keypoint confidence to use", 0.1, 0.9, 0.35, 0.05)

uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
if uploaded is not None:
    st.video(uploaded)

# ---------- Helpers ----------

# COCO-17 keypoint indices used by Ultralytics (YOLO pose):
# 0 nose, 1 eyeL, 2 eyeR, 3 earL, 4 earR, 5 shoulderL, 6 shoulderR,
# 7 elbowL, 8 elbowR, 9 wristL, 10 wristR, 11 hipL, 12 hipR,
# 13 kneeL, 14 kneeR, 15 ankleL, 16 ankleR
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16


def angle_deg(a, b, c):
    """
    Returns the angle ABC (at point b) in degrees.
    a, b, c: (x, y)
    """
    a, b, c = (
        np.array(a, dtype=float),
        np.array(b, dtype=float),
        np.array(c, dtype=float),
    )
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return np.nan
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def pick_side_and_metrics(kpts_xy, kpts_conf, kp_min=0.35):
    """
    Compute knee angles on both sides if confident.
    Returns: side ('L' or 'R' or None), angle, hip_below_knee(bool), dict with both angles
    """

    def valid(idx_list):
        return all(kpts_conf[i] >= kp_min for i in idx_list)

    out = {"L": np.nan, "R": np.nan}
    hip_below_L = hip_below_R = False

    # Left side
    if valid([L_HIP, L_KNEE, L_ANK]):
        out["L"] = angle_deg(kpts_xy[L_HIP], kpts_xy[L_KNEE], kpts_xy[L_ANK])
        hip_below_L = kpts_xy[L_HIP][1] > kpts_xy[L_KNEE][1]  # y increases downward

    # Right side
    if valid([R_HIP, R_KNEE, R_ANK]):
        out["R"] = angle_deg(kpts_xy[R_HIP], kpts_xy[R_KNEE], kpts_xy[R_ANK])
        hip_below_R = kpts_xy[R_HIP][1] > kpts_xy[R_KNEE][1]

    # Choose side with smaller (deeper) angle when both valid; else whichever is valid
    side = None
    if np.isfinite(out["L"]) and np.isfinite(out["R"]):
        side = "L" if out["L"] <= out["R"] else "R"
    elif np.isfinite(out["L"]):
        side = "L"
    elif np.isfinite(out["R"]):
        side = "R"

    chosen_angle = out[side] if side else np.nan
    hip_below = hip_below_L if side == "L" else (hip_below_R if side == "R" else False)
    return side, chosen_angle, hip_below, out


def ema(prev, new, alpha=0.2):
    if np.isnan(new):
        return prev
    return new if prev is None else (alpha * new + (1 - alpha) * prev)


def put_text(img, text, org, scale=0.7, thickness=2, color=(255, 255, 255), bg=True):
    if bg:
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x, y = org
        cv2.rectangle(img, (x - 6, y - h - 6), (x + w + 6, y + 6), (0, 0, 0), -1)
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
    )


# ---------- Main Pipeline ----------


def process_video_with_reps(
    in_path,
    out_path,
    model,
    device,
    conf,
    iou,
    vid_stride,
    up_thr,
    down_thr,
    need_hip_below,
    kp_min_conf,
):
    """
    Reads video, runs YOLO pose (streaming), annotates frames with skeleton + angles,
    counts valid reps, writes result to out_path (mp4).
    """
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # MP4 writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(out_path), fourcc, fps / max(1, vid_stride), (width, height)
    )

    # Rep state
    reps = 0
    skipped = 0
    stage = "up"  # 'up' or 'down'
    bottom_reached_valid = False
    angle_ema = None

    # Run model in stream mode (yields per-frame results)
    results = model.predict(
        source=str(in_path),
        stream=True,
        conf=conf,
        iou=iou,
        vid_stride=vid_stride,
        device=device,
        verbose=False,
    )

    frame_idx = 0
    for r in results:
        # r.plot() gives an annotated BGR image with skeletons
        frame = r.plot()

        # Pick the person with highest mean keypoint conf (if multiple)
        chosen = None
        best_score = -1.0
        if r.keypoints is not None and len(r.keypoints) > 0:
            for i, k in enumerate(r.keypoints):
                confs = (
                    k.conf[0].cpu().numpy()
                    if hasattr(k, "conf") and k.conf is not None
                    else None
                )
                xys = k.xy[0].cpu().numpy() if hasattr(k, "xy") else None
                if xys is None:
                    continue
                score = float(np.nanmean(confs)) if confs is not None else 0.0
                if score > best_score:
                    best_score = score
                    chosen = (
                        xys,
                        (
                            confs
                            if confs is not None
                            else np.ones((xys.shape[0],), dtype=float)
                        ),
                    )

        # Compute angles if we have a person
        angle_now = np.nan
        hip_below = False
        side = None
        both_angles = {}

        if chosen is not None:
            kpts_xy, kpts_conf = chosen
            side, angle_now, hip_below, both_angles = pick_side_and_metrics(
                kpts_xy, kpts_conf, kp_min=kp_min_conf
            )
            angle_ema = ema(angle_ema, angle_now, alpha=0.2)

        # Rep logic (simple FSM)
        rep_valid_now = False
        if np.isfinite(angle_ema):
            # At bottom
            if angle_ema <= down_thr:
                if stage == "up":
                    stage = "down"
                # mark whether depth is valid at bottom
                if not bottom_reached_valid:
                    depth_ok = hip_below if need_hip_below else True
                    if depth_ok:
                        bottom_reached_valid = True

            # Back to top
            if angle_ema >= up_thr:
                if stage == "down":
                    # completed a cycle: count only if bottom was valid
                    if bottom_reached_valid:
                        reps += 1
                        rep_valid_now = True
                    else:
                        skipped += 1
                    # reset for next rep
                    bottom_reached_valid = False
                    stage = "up"

        # ----- Overlay HUD -----
        # Box in top-left
        put_text(
            frame,
            f"Count: {reps}  (skipped: {skipped})",
            (20, 40),
            scale=0.9,
            thickness=2,
        )
        put_text(
            frame,
            f"Stage: {stage.upper()}",
            (20, 80),
            scale=0.9,
            thickness=2,
            color=(0, 255, 0) if stage == "up" else (0, 200, 255),
        )
        if np.isfinite(angle_ema):
            put_text(
                frame,
                f"Knee angle ({side or '-'}): {angle_ema:5.1f}°",
                (20, 120),
                scale=0.9,
                thickness=2,
            )
        else:
            put_text(frame, "Knee angle: --", (20, 120), scale=0.9, thickness=2)

        if stage == "down":
            color = (0, 255, 0) if bottom_reached_valid else (0, 0, 255)
            msg = "Depth OK" if bottom_reached_valid else "Depth NOT OK"
            put_text(frame, msg, (20, 160), scale=0.9, thickness=2, color=color)

        if rep_valid_now:
            put_text(
                frame,
                "✔ Valid rep",
                (20, 200),
                scale=0.9,
                thickness=2,
                color=(0, 255, 0),
            )
        # (Optional) show both angles
        if both_angles:
            txt = f"L:{both_angles.get('L', np.nan):.1f}°  R:{both_angles.get('R', np.nan):.1f}°"
            put_text(
                frame, txt, (20, 240), scale=0.8, thickness=2, color=(200, 200, 255)
            )

        writer.write(frame)
        frame_idx += 1

    writer.release()
    cap.release()
    return reps, skipped


# ---------- UI flow ----------


def find_saved_video(path: Path):
    if path.exists():
        return path
    # fallback scan
    vids = [p for p in path.parent.glob("*.mp4")]
    return vids[0] if vids else None


if uploaded is not None and st.button("▶️ Run Pose + Counter", type="primary"):
    # temp workspace
    workdir = Path(tempfile.mkdtemp(prefix="pose_app_"))
    in_path = workdir / uploaded.name
    with open(in_path, "wb") as f:
        f.write(uploaded.read())

    st.info("Loading model…")
    start_time = time.time()
    try:
        model = YOLO(model_name)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        shutil.rmtree(workdir, ignore_errors=True)
        st.stop()
    st.success("Model ready.")

    out_path = workdir / f"pose_squat_{in_path.stem}.mp4"
    progress = st.empty()
    progress.write("Processing video…")

    try:
        device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        reps, skipped = process_video_with_reps(
            in_path=in_path,
            out_path=out_path,
            model=model,
            device=device,
            conf=conf,
            iou=iou,
            vid_stride=vid_stride,
            up_thr=up_threshold,
            down_thr=down_threshold,
            need_hip_below=require_hip_below_knee,
            kp_min_conf=kp_min_conf,
        )
        elapsed = time.time() - start_time
        st.success(f"Done in {elapsed:.1f}s ✅  |  Reps: {reps}  (skipped: {skipped})")
        st.video(str(out_path))
        with open(out_path, "rb") as f:
            st.download_button(
                "⬇️ Download processed video",
                f,
                file_name=out_path.name,
                mime="video/mp4",
            )
        st.caption(f"Saved to: {out_path}")
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        progress.empty()

st.markdown("---")
st.caption(
    "Tip: If reps aren’t detected, lower the down-threshold (deeper squat) or reduce keypoint min conf."
)
