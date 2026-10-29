"""
test_video_local.py
─────────────────────────────────────────────────────────────────────────────
Run your trained YOLOv8 model on a LOCAL video file.
Saves:
  • Annotated output video  (big visible boxes + labels)
  • Excel report            (Start Time | End Time | Duration | Action)

SETUP (run once in your venv):
    pip install ultralytics opencv-python openpyxl

USAGE — just edit the 3 lines in the CONFIG section below, then run:
    python test_video_local.py

OR use command-line flags:
    python test_video_local.py \
        --model  "C:/Users/bavad/Downloads/weights.pt" \
        --video  "C:/Users/bavad/Downloads/clip_21_to_22.mp4" \
        --output "C:/Users/bavad/Downloads/annotated_clip.mp4"
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

import cv2

try:
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
except ImportError:
    print("ERROR: openpyxl not installed.  Run: pip install openpyxl")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed.  Run: pip install ultralytics")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  ★  EDIT THESE 3 LINES — then just press F5 / Run
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_PATH  = r"C:\Users\bavad\Downloads\weights.pt"
VIDEO_PATH  = r"C:\Users\bavad\Downloads\samplevideo2.mp4"
OUTPUT_DIR  = r"C:\Users\bavad\Downloads\crane_results-21to22"
# ═══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
#  ANNOTATION SETTINGS  (tweak these to make boxes/text bigger or smaller)
# ─────────────────────────────────────────────────────────────────────────────
BOX_THICKNESS   = 7        # bounding box line thickness (px)
FONT_SCALE      = 1.9     # label text size (1.0 = normal, 2.0 = double)
FONT_THICKNESS  = 4        # label text stroke thickness
LABEL_PADDING   = 12       # padding inside label background box (px)
CONF_THRESHOLD  = 0.45     # minimum confidence to show a detection
DEBOUNCE_FRAMES = 5        # frames a new action must persist before logging

# Overlay bar at top of frame (action status)
OVERLAY_FONT_SCALE    = 1.6
OVERLAY_FONT_THICK    = 3
OVERLAY_BAR_HEIGHT    = 60   # px height of the top status bar

# Class → colour map (BGR).  Add your own classes here if needed.
CLASS_COLORS = {
    "bucket_empty_grounding":  (0,   200, 255),   # orange
    "bucket_empty_lifted":     (0,   255, 200),   # cyan-green
    "bucket_loaded_grounding": (0,   100, 255),   # deep orange
    "bucket_loaded_lifted":    (50,  255,  50),   # green
    "idle":                    (180, 180, 180),   # grey
    "shuttering_grounding":    (255, 100,  50),   # blue-ish
    "shuttering_lifted":       (255,  50, 200),   # purple
}
DEFAULT_COLOR = (0, 255, 255)   # yellow — used for any class not in the map above


# ─────────────────────────────────────────────────────────────────────────────
#  EXCEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
HEADERS = ["#", "Video File", "Start Time", "End Time", "Duration (s)", "Action"]

def init_excel(path: str):
    if os.path.isfile(path):
        try:
            wb = load_workbook(path)
            ws = wb.active
            print(f"📄 Appending to existing report: {path}")
            return wb, ws
        except Exception as e:
            print(f"⚠️  Could not open existing report ({e}). Creating fresh one.")

    wb = Workbook()
    ws = wb.active
    ws.title = "Crane Action Log"

    # Header row styling
    header_fill = PatternFill("solid", fgColor="1F4E79")
    header_font = Font(bold=True, color="FFFFFF", size=12)
    widths = [6, 32, 22, 22, 16, 30]

    for col, (h, w) in enumerate(zip(HEADERS, widths), start=1):
        cell = ws.cell(row=1, column=col, value=h)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
        ws.column_dimensions[get_column_letter(col)].width = w

    ws.row_dimensions[1].height = 22
    wb.save(path)
    print(f"📄 Created new report: {path}")
    return wb, ws


def log_segment(wb, ws, xlsx_path, video_name, row_idx, start_dt, end_dt, action):
    duration_s = round((end_dt - start_dt).total_seconds(), 1)

    # Alternate row shading
    fill_color = "D9E8F5" if row_idx % 2 == 0 else "FFFFFF"
    row_fill   = PatternFill("solid", fgColor=fill_color)

    values = [
        row_idx,
        video_name,
        start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        duration_s,
        action,
    ]
    new_row = ws.max_row + 1
    for col, val in enumerate(values, start=1):
        cell = ws.cell(row=new_row, column=col, value=val)
        cell.fill = row_fill
        cell.alignment = Alignment(horizontal="center" if col != 2 else "left",
                                   vertical="center")

    try:
        wb.save(xlsx_path)
    except PermissionError:
        print(f"⚠️  Excel file is open — close it in Excel and it will save on next update.")

    return duration_s


# ─────────────────────────────────────────────────────────────────────────────
#  DRAW ANNOTATIONS (big, visible)
# ─────────────────────────────────────────────────────────────────────────────
def draw_detections(frame, result, class_names):
    """Draw bounding boxes + labels with large, clearly visible text."""
    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf   = float(box.conf.item())
        label  = class_names.get(cls_id, f"cls{cls_id}")
        color  = CLASS_COLORS.get(label, DEFAULT_COLOR)

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        # Label text
        text  = f"{label}  {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)

        # Label background pill
        pad = LABEL_PADDING
        lx1 = x1
        ly1 = max(0, y1 - th - 2 * pad)
        lx2 = x1 + tw + 2 * pad
        ly2 = y1

        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, -1)

        # Dark contrasting text
        text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
        cv2.putText(frame, text,
                    (lx1 + pad, ly2 - pad),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE, text_color, FONT_THICKNESS,
                    cv2.LINE_AA)

    return frame


def draw_overlay(frame, action, elapsed_s, frame_no, total_frames, fps):
    """Draw top status bar with current action, progress, FPS."""
    h, w = frame.shape[:2]

    # Dark semi-transparent bar
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, OVERLAY_BAR_HEIGHT), (20, 20, 20), -1)
    cv2.addWeighted(bar, 0.75, frame, 0.25, 0, frame)

    # Action text (left)
    action_color = CLASS_COLORS.get(action, DEFAULT_COLOR)
    cv2.putText(frame, f"Action: {action}   {elapsed_s:.1f}s",
                (16, OVERLAY_BAR_HEIGHT - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                OVERLAY_FONT_SCALE, action_color, OVERLAY_FONT_THICK,
                cv2.LINE_AA)

    # Progress + FPS (right)
    progress = f"Frame {frame_no}/{total_frames}   FPS:{fps:.1f}"
    (pw, _), _ = cv2.getTextSize(progress, cv2.FONT_HERSHEY_SIMPLEX,
                                  0.9, 2)
    cv2.putText(frame, progress,
                (w - pw - 16, OVERLAY_BAR_HEIGHT - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (200, 200, 200), 2, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run(model_path, video_path, output_dir, conf=CONF_THRESHOLD,
        debounce=DEBOUNCE_FRAMES, show=True):

    # ── Validate paths ──────────────────────────────────────────────────────
    for p, label in [(model_path, "Model"), (video_path, "Video")]:
        if not os.path.isfile(p):
            print(f"❌ {label} file not found: {p}")
            sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    video_name   = os.path.splitext(os.path.basename(video_path))[0]
    out_video    = os.path.join(output_dir, f"{video_name}_annotated.mp4")
    out_xlsx     = os.path.join(output_dir, "crane_report.xlsx")

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"\n🔧 Loading model: {model_path}")
    model       = YOLO(model_path)
    class_names = model.names
    print(f"   Classes: {list(class_names.values())}")

    # ── Open video ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        sys.exit(1)

    src_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_total = total_frm / src_fps

    print(f"\n📹 Video : {video_path}")
    print(f"   Size  : {src_w}x{src_h}  |  FPS: {src_fps:.2f}  |  "
          f"Frames: {total_frm}  |  Duration: {duration_total/60:.1f} min")

    # ── Video writer ────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, src_fps, (src_w, src_h))
    if not writer.isOpened():
        print(f"❌ Cannot create output video: {out_video}")
        sys.exit(1)

    # ── Excel ───────────────────────────────────────────────────────────────
    wb, ws = init_excel(out_xlsx)
    row_idx = ws.max_row   # track row number for alternating shading

    # ── State tracking ──────────────────────────────────────────────────────
    # Use the video's start timestamp as base (or just datetime.now())
    base_time         = datetime.now().replace(microsecond=0)
    current_action    = None
    current_start     = None
    candidate_action  = None
    candidate_count   = 0
    NO_DET            = "no_detection"

    frame_no   = 0
    fps_timer  = time.time()
    fps_buf    = 0
    live_fps   = 0.0

    print(f"\n▶️  Processing — output will be saved to:\n   {out_video}\n   {out_xlsx}\n")
    print("   (Press 'q' in the preview window to stop early)\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_no += 1

            # ── Inference ───────────────────────────────────────────────────
            results = model.predict(frame, conf=conf, verbose=False)
            result  = results[0]

            # ── Best detection this frame ────────────────────────────────────
            detected = NO_DET
            if len(result.boxes):
                confs   = result.boxes.conf.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                detected = class_names[cls_ids[confs.argmax()]]

            # Timestamp for this frame
            frame_offset = timedelta(seconds=(frame_no / src_fps))
            now          = base_time + frame_offset

            # ── Debounce state machine ───────────────────────────────────────
            if current_action is None:
                current_action = detected
                current_start  = now
                candidate_action = None
                candidate_count  = 0

            elif detected == current_action:
                candidate_action = None
                candidate_count  = 0

            else:
                if detected == candidate_action:
                    candidate_count += 1
                else:
                    candidate_action = detected
                    candidate_count  = 1

                if candidate_count >= debounce:
                    row_idx += 1
                    dur = log_segment(wb, ws, out_xlsx, video_name,
                                      row_idx, current_start, now, current_action)
                    print(f"  📝 {current_action:<30s}  "
                          f"{current_start.strftime('%H:%M:%S')} → "
                          f"{now.strftime('%H:%M:%S')}  ({dur}s)")
                    current_action   = candidate_action
                    current_start    = now
                    candidate_action = None
                    candidate_count  = 0

            # ── Draw big annotations ─────────────────────────────────────────
            annotated = draw_detections(frame.copy(), result, class_names)
            elapsed   = (now - current_start).total_seconds()
            annotated = draw_overlay(annotated, current_action,
                                     elapsed, frame_no, total_frm, live_fps)

            # ── Write frame ──────────────────────────────────────────────────
            writer.write(annotated)

            # ── Live FPS counter ─────────────────────────────────────────────
            fps_buf += 1
            if time.time() - fps_timer >= 1.0:
                live_fps  = fps_buf / (time.time() - fps_timer)
                fps_buf   = 0
                fps_timer = time.time()
                pct = frame_no / total_frm * 100 if total_frm else 0
                print(f"  ⏳ {pct:5.1f}%  frame {frame_no}/{total_frm}"
                      f"  inf-FPS: {live_fps:.1f}", end="\r")

            # ── Optional live preview ────────────────────────────────────────
            if show:
                # Show at half size so it fits the screen
                preview = cv2.resize(annotated, (src_w // 2, src_h // 2))
                cv2.imshow("Crane Annotator  (q = quit early)", preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n\n🛑 Stopped early by user.")
                    break

    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted.")

    finally:
        # Log the final open segment
        if current_action and current_start:
            final_time = base_time + timedelta(seconds=(frame_no / src_fps))
            row_idx += 1
            dur = log_segment(wb, ws, out_xlsx, video_name,
                              row_idx, current_start, final_time, current_action)
            print(f"\n  📝 Final: {current_action:<30s}  "
                  f"{current_start.strftime('%H:%M:%S')} → "
                  f"{final_time.strftime('%H:%M:%S')}  ({dur}s)")

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        print(f"\n✅ Annotated video : {out_video}")
        print(f"✅ Excel report     : {out_xlsx}")
        print("Done.\n")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test YOLO model on local video.")
    parser.add_argument("--model",  default=MODEL_PATH,  help="Path to best.pt weights")
    parser.add_argument("--video",  default=VIDEO_PATH,  help="Path to input video")
    parser.add_argument("--output", default=OUTPUT_DIR,  help="Output folder")
    parser.add_argument("--conf",   type=float, default=CONF_THRESHOLD)
    parser.add_argument("--debounce", type=int, default=DEBOUNCE_FRAMES)
    parser.add_argument("--no-display", action="store_true",
                        help="Disable live preview window (faster)")
    args = parser.parse_args()

    run(
        model_path = args.model,
        video_path = args.video,
        output_dir = args.output,
        conf       = args.conf,
        debounce   = args.debounce,
        show       = not args.no_display,
    )
