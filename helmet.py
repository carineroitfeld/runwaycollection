from ultralytics import YOLO
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict, deque
from pythonosc import udp_client

from machine_logic import classify_machine
from human_logic import interpret_human


# -------------------------
# Fingerprint helpers
# -------------------------
EMA_ALPHA = 0.08        # lower = smoother over time (was 0.15)
MIN_SHOW = 0.02         # hide bars smaller than 2%
BAR_W, BAR_H = 220, 20  # bar width/height in px
PAD = 10                # padding from window edges


def color_for(i):
    # simple distinct palette (BGR for OpenCV)
    palette = [
        (50, 170, 220), (80, 220, 100), (240, 200, 70),
        (210, 100, 230), (255, 130, 90), (120, 200, 255),
        (90, 160, 255), (150, 230, 160), (200, 120, 255)
    ]
    return palette[i % len(palette)]


def aggregate_probs(detections, num_classes):
    """
    detections: list of (cls_id:int, conf:float) for current frame
    Combines multiple detections per class using 1 - Π(1 - conf),
    then normalizes across classes to produce a probability vector.
    """
    scores = np.zeros(num_classes, dtype=np.float32)
    accum = defaultdict(lambda: 1.0)
    for cls_id, conf in detections:
        if 0 <= cls_id < num_classes:
            accum[cls_id] *= (1.0 - float(conf))
    for cls_id, prod in accum.items():
        scores[cls_id] = 1.0 - prod

    s = scores.sum()
    return (scores / s) if s > 0 else scores


class FingerprintSmoother:
    def __init__(self, num_classes):
        self.ema = np.zeros(num_classes, dtype=np.float32)

    def update(self, probs):
        # exponential moving average + renormalize
        self.ema = (1 - EMA_ALPHA) * self.ema + EMA_ALPHA * probs
        s = self.ema.sum()
        if s > 0:
            self.ema = self.ema / s
        return self.ema


def draw_fingerprint(frame, probs, class_names, top_k=5, anchor="tr"):
    """
    Draw compact bar chart of top-k classes on the frame.
    anchor: 'tl' (top-left), 'tr', 'bl', 'br'
    """
    h, w = frame.shape[:2]
    label_col_w = 180  # space to print "name: XX%"
    x0 = PAD - 20 if 'l' in anchor else w - (BAR_W + label_col_w) - PAD
    # estimate vertical block height
    rows = min(top_k, len(class_names))
    block_h = rows * (BAR_H + 6)
    y0 = PAD + 60 if 't' in anchor else h - block_h - PAD

    # pick top-k by probability
    idx = np.argsort(-probs)[:top_k]
    y = y0

    # title
    cv2.putText(frame, "Style fingerprint", (x0, max(y0 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    for i in idx:
        p = float(probs[i])
        if p < MIN_SHOW:
            continue
        label = class_names[i]
        cw = int(BAR_W * p)

        # background bar
        cv2.rectangle(frame, (x0, y), (x0 + BAR_W, y + BAR_H), (35, 35, 35), -1)
        # value bar
        cv2.rectangle(frame, (x0, y), (x0 + cw, y + BAR_H), color_for(i), -1)
        # border
        cv2.rectangle(frame, (x0, y), (x0 + BAR_W, y + BAR_H), (0, 0, 0), 1)

        # text: "Label: 67%"
        txt = f"{label}: {int(round(p * 100))}%"
        cv2.putText(frame, txt, (x0 + BAR_W + 10, y + BAR_H - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        y += BAR_H + 6


# -------------------------
# Model + class names
# -------------------------
model = YOLO("best3.pt")  # or "best.pt" if that's your model

# Keep this order identical to your model's class order
class_names = ['Bohemian', 'Chic', 'Denim', 'Elegant',
               'Formal', 'Rocker', 'Luxury', 'Sportswear']
num_classes = len(class_names)

# smoother state (once)
fp = FingerprintSmoother(num_classes=num_classes)


# -------------------------
# Label smoothing state
# -------------------------
STABLE_WINDOW = 10  # number of frames to look back for majority vote

machine_label_history = deque(maxlen=STABLE_WINDOW)
human_label_history = deque(maxlen=STABLE_WINDOW)


def most_common_label(history: deque) -> str | None:
    """Return the label that appears most often in the deque, or None if empty."""
    if not history:
        return None
    counts = {}
    for lbl in history:
        counts[lbl] = counts.get(lbl, 0) + 1
    return max(counts, key=counts.get)


# -------------------------
# OSC clients to TouchDesigner (separate ports)
# -------------------------
OSC_IP = "127.0.0.1"

STYLE_PORT = 8000       # raw detected class (Denim)
MACHINE_PORT = 8001     # PURE/HYBRID/TRIAD
HUMAN_PORT = 8002       # archetype

osc_style = udp_client.SimpleUDPClient(OSC_IP, STYLE_PORT)
osc_machine = udp_client.SimpleUDPClient(OSC_IP, MACHINE_PORT)
osc_human = udp_client.SimpleUDPClient(OSC_IP, HUMAN_PORT)


# -------------------------
# Video loop
# -------------------------
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Resize frame to 640x640 for consistent detection size
    resized_frame = cv2.resize(frame, (640, 640))

    # Run YOLO detection (generator over results for this frame)
    results = model(resized_frame, stream=True)

    # Collect detections for fingerprint
    frame_dets = []

    # Process detection results: draw boxes/labels and collect (cls, conf)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # draw rectangle and label
            color = (0, 255, 255)
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), color, 2)

            # guard against class index issues
            label_name = class_names[cls_id] if 0 <= cls_id < num_classes else f"cls_{cls_id}"
            label = f"{label_name} {conf:.2f}"
            cv2.putText(resized_frame, label, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            # collect for fingerprint
            frame_dets.append((cls_id, conf))

    # ---- compute and draw the style fingerprint (for your OpenCV view) ----
    probs = aggregate_probs(frame_dets, num_classes=num_classes)   # per-frame normalized mix
    smoothed = fp.update(probs)                                    # EMA smoothing across time
    draw_fingerprint(resized_frame, smoothed, class_names, top_k=5, anchor="tr")

    # ---- Machine + Human logic on the same fingerprint ----
    machine_view = None
    human_view = None

    try:
        machine_view = classify_machine(smoothed.tolist(), class_names)
        machine_label_history.append(machine_view.segment_label)
    except Exception as e:
        print(f"[machine_logic error] {e}")

    try:
        human_view = interpret_human(smoothed.tolist())
        human_label_history.append(human_view.label)
    except Exception as e:
        print(f"[human_logic error] {e}")

    # Stable labels from last N frames
    stable_machine_label = most_common_label(machine_label_history)
    stable_human_label = most_common_label(human_label_history)

    # Print + OSC for machine layer
    if stable_machine_label and machine_view is not None:
        print(
            f"Machine: {stable_machine_label} | "
            f"conf={machine_view.confidence:.2f} | "
            f"H={machine_view.uncertainty:.2f}"
        )
        osc_machine.send_message("/machine_label", stable_machine_label)

    # Print + OSC for human layer
    if stable_human_label and human_view is not None:
        print(
            f"Human:   {stable_human_label} | "
            f"fit={human_view.fit:.2f} | "
            f"alpha={human_view.alpha:.2f} | "
            f"ambiguous={human_view.ambiguous}"
        )
        osc_human.send_message("/archetype", stable_human_label)

    # ---- OSC: send ONLY the main detected class name for TD visuals (port 8000) ----
    if frame_dets:
        # pick highest-confidence detection in this frame
        best_cls_id, best_conf = max(frame_dets, key=lambda t: t[1])
        if 0 <= best_cls_id < num_classes:
            top_label = class_names[best_cls_id]   # e.g. "Denim"
            osc_style.send_message("/style", top_label)

    # Show the frame
    cv2.imshow("Helmet Detection - YOLOv11n (640x640)", resized_frame)

    # Press 'q' to exit loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()