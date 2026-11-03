"""Gradio demo: ECG beat classification (research prototype, not for clinical use)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import gradio as gr

CLASSES = ["N (normal)", "S (supraventricular)", "V (ventricular)", "F (fusion)", "Q (unknown)"]
MODEL = tf.keras.models.load_model("ecg_cnn.keras")
DISCLAIMER = ("Research prototype trained on the public MIT-BIH beat dataset. "
              "Not a medical device. Not for clinical use. Input: one heartbeat of 187 samples "
              "(125 Hz, values scaled to 0-1), pasted as comma-separated numbers or uploaded as a CSV.")


def parse(text, file):
    if file is not None:
        text = open(file, encoding="utf-8").read().splitlines()[0]
    vals = np.array([float(v) for v in text.replace(";", ",").split(",") if v.strip()], dtype="float32")
    if len(vals) < 187:
        vals = np.pad(vals, (0, 187 - len(vals)))
    return vals[:187]


def predict(text, file):
    try:
        x = parse(text or "", file)
    except Exception:
        return {}, None
    p = MODEL.predict(x[None, :, None], verbose=0)[0]
    fig, ax = plt.subplots(figsize=(6, 2.4))
    ax.plot(x)
    ax.set_title("Input beat")
    ax.set_xlabel("sample (125 Hz)")
    fig.tight_layout()
    return {c: float(v) for c, v in zip(CLASSES, p)}, fig


demo = gr.Interface(
    fn=predict,
    inputs=[gr.Textbox(lines=3, label="187 comma-separated values"), gr.File(type="filepath", label="or a CSV (first row)")],
    outputs=[gr.Label(label="Model output"), gr.Plot(label="Input beat")],
    title="ECG heartbeat classification (1D CNN)",
    description=DISCLAIMER,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
