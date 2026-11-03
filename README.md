# ECG heartbeat classification (1D CNN)

Five-class single-beat classifier (N normal, S supraventricular, V ventricular, F fusion, Q unknown) on preprocessed MIT-BIH arrhythmia beats, with a small 1D convolutional network and a web demo.

> **Research prototype. Not a medical device. Not for clinical use.** Trained and tested only on a public dataset; no patient data is included in this repository.

**Demo:** a Gradio app is in `demo/` (see Run it below). There is no hosted version yet.

## Results (dataset test file, 21,892 beats)

| Metric | Value |
|---|---|
| Accuracy | 95.6% |
| Macro F1 | 0.821 |
| Macro AUC (one-vs-rest) | 0.980 |

| Class | Recall | Precision | AUC | Test beats |
|---|---|---|---|---|
| N | 96.6% | 98.5% | 0.983 | 18,118 |
| S | 58.5% | 76.8% | 0.935 | 556 |
| V | 94.5% | 84.5% | 0.995 | 1,448 |
| F | 82.7% | 52.3% | 0.990 | 162 |
| Q | 99.4% | 87.9% | 0.999 | 1,608 |

Overall accuracy is dominated by the normal class (83% of test beats). Supraventricular beats are the weak point: 35% of them are labelled normal. Full numbers in `results/metrics_summary.json`.

## Method

- Data: Kaggle `shayanfazeli/heartbeat`, MIT-BIH arrhythmia beats segmented to 187 samples (125 Hz). The dataset's own train and test files are used. 10% of train is held out for validation and early stopping.
- Model: three Conv1D blocks (32, 64, 128 filters; kernel 7, 5, 3) with batch normalisation, max pooling, global average pooling, dropout 0.3 and a softmax output. Adam 1e-3, softened class weights, early stopping on validation loss.

## Limitations (read before quoting the numbers)

- The dataset's train and test files are not split by patient, so beats from the same recordings can appear in both. Patient-independent protocols (for example the AAMI/de Chazal inter-patient split) give lower scores, so these numbers are optimistic.
- Single-beat classification of pre-segmented beats: no rhythm context and no beat detection from raw ECG.
- Strong class imbalance; read the per-class recall, not just accuracy. The validation loss during training was unstable between epochs, and early stopping restored the best epoch.
- One source database with a small number of subjects and no external validation.

## Run it

Train (Google Colab, free T4 GPU): open `ecg_arrhythmia_training.ipynb` and run all cells. It saves `ecg_cnn.keras` and `metrics.json`.

Demo locally:

```bash
cd demo
pip install -r requirements.txt
# put ecg_cnn.keras in this folder
python app.py
```

Paste 187 comma-separated values (one beat) or upload a CSV whose first row is one beat.

## Author

Abdullah Ghani, data and AI engineering student at ECE Paris.
