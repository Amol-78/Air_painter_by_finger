import cv2
import numpy as np
import time
from collections import deque
from datetime import datetime

# ---------- MediaPipe (hand tracking) ----------
try:
    import mediapipe as mp
except ImportError:
    raise SystemExit("Please install MediaPipe: pip install mediapipe")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---------- Settings ----------
BRUSH_SIZE_INIT = 8
ERASER_SIZE_INIT = 40
UNDO_LIMIT = 30
SMOOTHING = 0.4  # 0..1 (higher = smoother but laggier)
PALETTE = [
    (255, 255, 255),  # 1: White
    (0, 0, 0),        # 2: Black
    (0, 0, 255),      # 3: Red (BGR)
    (0, 255, 0),      # 4: Green
    (255, 0, 0),      # 5: Blue
    (255, 255, 0),    # 6: Cyan
    (255, 0, 255),    # 7: Magenta
    (0, 255, 255),    # 8: Yellow
]

# ---------- Helpers ----------
def lerp(a, b, t):
    return a + (b - a) * t

def put_hud(frame, color, brush, eraser, drawing, erasing, rainbow, neon):
    h, w = frame.shape[:2]
    y = 24
    lines = [
        "Air Painter — Controls:",
        "[1..8] Color  |  +/- Brush  |  [E] Eraser toggle  |  [D] Draw toggle",
        "[R] Rainbow   |  [G] Neon Glow  |  [U] Undo  |  [S] Save  |  [Q] Quit",
        f"Mode: {'ERASER' if erasing else 'DRAW'} | Brush: {brush}px | Eraser: {eraser}px | "
        f"Rainbow: {'On' if rainbow else 'Off'} | Neon: {'On' if neon else 'Off'}",
    ]
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        y += 22

    # palette preview
    px, py, size, gap = 10, y + 8, 24, 6
    for i, c in enumerate(PALETTE, start=1):
        x1 = px + (i - 1) * (size + gap)
        cv2.rectangle(frame, (x1, py), (x1 + size, py + size), c, -1)
        cv2.rectangle(frame, (x1, py), (x1 + size, py + size), (50, 50, 50), 2)
        cv2.putText(frame, str(i), (x1 + 6, py + size + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)

    # current color indicator
    cv2.rectangle(frame, (w - 120, 10), (w - 20, 40), color, -1)
    cv2.rectangle(frame, (w - 120, 10), (w - 20, 40), (40, 40, 40), 2)
    cv2.putText(frame, "Color", (w - 118, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)

def fingertip_px(landmarks, image_shape):
    """Return (x,y) for index fingertip (id=8) in pixel coords."""
    h, w = image_shape[:2]
    tip = landmarks[8]
    return int(tip.x * w), int(tip.y * h)

def fingers_up(landmarks):
    """
    Very simple heuristic:
    - Index finger up if tip (8) y is above PIP (6)
    - Middle finger up if tip (12) above PIP (10)
    """
    up = {'index': False, 'middle': False}
    up['index'] = landmarks[8].y < landmarks[6].y
    up['middle'] = landmarks[12].y < landmarks[10].y
    return up

def cycle_rainbow(t):
    """Return BGR color cycling over time t (seconds)."""
    # Use HSV-like wheel
    period = 3.0
    phase = (t % period) / period  # 0..1
    # Convert to BGR roughly
    r = int(255 * max(0, min(1, 1 - abs(phase * 6 - 3))))
    g = int(255 * max(0, min(1, 1 - abs((phase * 6 - 2) % 6 - 3))))
    b = int(255 * max(0, min(1, 1 - abs((phase * 6 - 4) % 6 - 3))))
    return (b, g, r)

# ---------- Main ----------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera")

    ret, frame = cap.read()
    if not ret:
        raise SystemExit("Failed to read from camera")

    h, w = frame.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)       # strokes layer
    glow_layer = np.zeros_like(canvas)                  # for neon glow
    undo_stack = deque(maxlen=UNDO_LIMIT)

    color = (0, 255, 0)
    brush_size = BRUSH_SIZE_INIT
    eraser_size = ERASER_SIZE_INIT
    drawing_enabled = True
    eraser_enabled = False
    rainbow = False
    neon = False

    prev_point = None
    smoothing_point = None
    stroke_active = False

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    print("""
Air Painter ready.

Controls:
  1..8 = pick color     R = toggle rainbow     G = toggle neon glow
  + / - = brush size     E = toggle eraser      D = toggle drawing
  U = undo               S = save image         Q = quit
""")

    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---- Hand detection ----
        res = hands.process(rgb)
        index_up = False
        point = None

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            up = fingers_up(lm.landmark)
            index_up = up['index']
            point = fingertip_px(lm.landmark, frame.shape)

            # optional: draw landmarks small
            mp_drawing.draw_landmarks(
                frame, lm, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(120, 120, 120), thickness=1)
            )

        # ---- Determine drawing state ----
        # Draw when index finger is up AND drawing is enabled
        can_draw_now = drawing_enabled and index_up

        # Smooth point
        if point is not None:
            if smoothing_point is None:
                smoothing_point = point
            else:
                sx = int(lerp(smoothing_point[0], point[0], SMOOTHING))
                sy = int(lerp(smoothing_point[1], point[1], SMOOTHING))
                smoothing_point = (sx, sy)

        # Start stroke: transition from no-prev to having a point while drawing
        if can_draw_now and prev_point is None and smoothing_point is not None:
            # Save undo snapshot
            undo_stack.append((canvas.copy(), glow_layer.copy()))
            stroke_active = True

        # Draw line segment
        if can_draw_now and smoothing_point is not None and prev_point is not None:
            p1 = prev_point
            p2 = smoothing_point
            if eraser_enabled:
                # Erase by drawing black on canvas & glow
                cv2.line(canvas, p1, p2, (0, 0, 0), eraser_size, cv2.LINE_AA)
                cv2.line(glow_layer, p1, p2, (0, 0, 0), eraser_size + 8, cv2.LINE_AA)
            else:
                # Color select (rainbow vs fixed)
                draw_color = cycle_rainbow(time.time() - t0) if rainbow else color

                # Base stroke
                cv2.line(canvas, p1, p2, draw_color, brush_size, cv2.LINE_AA)

                # Neon glow: draw thick on glow layer, then blur later
                if neon:
                    cv2.line(glow_layer, p1, p2, draw_color, max(brush_size * 3, brush_size + 10), cv2.LINE_AA)

        # Update prev
        prev_point = smoothing_point if can_draw_now else None
        if not can_draw_now:
            stroke_active = False

        # ---- Compose final output ----
        out = frame.copy()

        # Apply glow (blurred)
        if neon:
            blurred = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=8, sigmaY=8)
            out = cv2.addWeighted(out, 1.0, blurred, 0.6, 0)

        # Overlay strokes
        out = cv2.addWeighted(out, 1.0, canvas, 1.0, 0)

        # Crosshair/indicator at fingertip
        if smoothing_point is not None:
            clr = (0, 255, 255) if rainbow else ((0, 0, 0) if eraser_enabled else color)
            size = eraser_size if eraser_enabled else brush_size
            cv2.circle(out, smoothing_point, max(6, size // 2), clr, 2, cv2.LINE_AA)

        # HUD
        hud_color = cycle_rainbow(time.time() - t0) if rainbow else color
        put_hud(out, hud_color, brush_size, eraser_size, drawing_enabled, eraser_enabled, rainbow, neon)

        cv2.imshow("Air Painter — Index finger to draw", out)

        # ---- Keys ----
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('+'), ord('=')):
            brush_size = min(100, brush_size + 1)
        elif key in (ord('-'), ord('_')):
            brush_size = max(1, brush_size - 1)
        elif key == ord('e'):
            eraser_enabled = not eraser_enabled
        elif key == ord('d'):
            drawing_enabled = not drawing_enabled
        elif key == ord('r'):
            rainbow = not rainbow
        elif key == ord('g'):
            neon = not neon
        elif key == ord('u'):
            if undo_stack:
                canvas, glow_layer = undo_stack.pop()
        elif key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            # save just the art (canvas + glow) on white background
            white_bg = np.full_like(canvas, 255)
            art = white_bg.copy()
            if neon:
                blurred = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=8, sigmaY=8)
                art = cv2.addWeighted(art, 1.0, blurred, 0.6, 0)
            art = cv2.addWeighted(art, 1.0, canvas, 1.0, 0)
            cv2.imwrite(f"air_paint_{ts}.png", art)
            print(f"[✔] Saved artwork: air_paint_{ts}.png")

        # color keys 1..8
        if key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8')):
            idx = int(chr(key)) - 1
            color = PALETTE[idx]
            rainbow = False  # picking a color disables rainbow

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
