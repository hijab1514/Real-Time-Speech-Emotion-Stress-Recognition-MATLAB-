<!-- Detailed engineering design document. The front-page overview is in ../README.md -->
# Real-Time Speech Emotion & Stress Recognition (MATLAB)

Hybrid **CNN–LSTM** emotion classifier + rule-based **stress** engine + **explainable** output, wrapped in an App Designer–style GUI. Built as **three independent modules** so each can be developed, tested, and graded on its own.

```
+sigproc/    Module 1 — Signal Processing Engine   (preprocess + features)
+ai/     Module 2 — AI Prediction Engine        (CNN-LSTM + stress + XAI)
+gui/    Module 3 — GUI                          (uifigure app; thin layer)
```

The MATLAB **package folders** (the `+` prefix) enforce the separation: the GUI can only reach the engines through `sigproc.*` and `ai.*` calls, so no analysis logic leaks into the interface. That boundary is the thing examiners reward.

---

## 0. One correction you should make before building (read this)

Your brief lists **six "emotions": Happy, Sad, Angry, Neutral, Fear, Stress.** Stress is not an emotion category at the same level as the other five — it is a separate *arousal/state* dimension, and **none of RAVDESS, TESS, CREMA-D, or SAVEE contain a "stress" label.** If you train one 6-class classifier including "Stress", you will have no labelled data for that class and the model will be indefensible under questioning.

The architecture here does what your own brief implicitly does in sections 6 vs 7: **two separate outputs.**

| Output | Method | Classes | Data |
|---|---|---|---|
| **Emotion** | supervised CNN-LSTM | Angry, Happy, Sad, Neutral, Fear (+ optional Disgust/Surprise) | RAVDESS/CREMA-D etc. |
| **Stress** | unsupervised acoustic index | Low / Moderate / High | computed from features; **calibrate on your own recordings** |

State this explicitly in the report. "Stress is estimated from validated acoustic correlates (pitch instability, energy variation, speaking rate), not from a labelled corpus, and the thresholds are calibrated on N relaxed-vs-stressed recordings" is a defensible sentence. "We trained a 6-class emotion+stress CNN" is not.

---

## 1. System architecture

```
                          ┌─────────────────────────────┐
                          │   Module 3 — GUI (+gui)      │
                          │  uifigure: record / upload,  │
                          │  plots, panels, history,     │
                          │  export                      │
                          └───────────────┬─────────────┘
                                          │ calls only sigproc.* / ai.*
        ┌─────────────────────────────────┴─────────────────────────────┐
        │                                                                 │
┌───────▼────────────────────────────┐        ┌────────────────────────▼──────────────────┐
│   Module 1 — DSP Engine (+sigproc)      │        │   Module 2 — AI Engine (+ai)               │
│                                     │        │                                            │
│  preprocessAudio                    │        │  buildCNNLSTM      (network definition)    │
│   mono→resample→pre-emph→denoise    │ feat   │  trainEmotionModel (training pipeline)     │
│   →VAD→normalize                    ├───────►│  predictEmotion    (inference + scores)    │
│  extractFeatures                    │        │  stressEngine      (3-level acoustic index)│
│   MFCC/Δ/ΔΔ, spectral, pitch, ZCR,  │        │  explainPrediction (occlusion XAI + text)  │
│   energy → sequence + summary       │        │                                            │
└─────────────────────────────────────┘        └────────────────────────────────────────────┘
        ▲                                                         │
        │                                              models/emotionModel.mat
   audio in (mic / file)                               reports/history.csv + report cards
```

**Module responsibilities**

- **Audio Acquisition** (`+utils/recordMic`, `loadAudioFile`) — mic capture and file reading; the only code that touches hardware/disk for input.
- **Audio Processing** (`sigproc.preprocessAudio`) — turns arbitrary input into a clean, 16 kHz, normalised, speech-only signal plus framing metadata.
- **Feature Extraction** (`sigproc.extractFeatures`) — produces two products from one pass: a per-frame **sequence** `[F×T]` for the network, and a per-utterance **summary** for the stress engine and the explainer.
- **Deep Learning** (`ai.buildCNNLSTM`, `trainEmotionModel`, `predictEmotion`) — define, train, run the classifier.
- **Stress** (`ai.stressEngine`) — independent rule-based subsystem.
- **Explainable AI** (`ai.explainPrediction`) — occlusion importance + plain-language reasons.
- **Storage/Reporting** (`utils.exportReport`) — CSV history + text report cards in `reports/`.
- **GUI** (`gui.SERApp`) — orchestration and display only.

---

## 2. Development roadmap

| Phase | Objective | Key deliverable | MATLAB tools | Est. time |
|---|---|---|---|---|
| 1 | Dataset prep | class folders of `.wav`, balanced | Audio Toolbox, `audioDatastore` | 3–5 days |
| 2 | Preprocessing | `sigproc.preprocessAudio` validated | `resample`, `detectSpeech`, `stft` | 4–6 days |
| 3 | Feature extraction | `sigproc.extractFeatures` + smoke tests | `audioFeatureExtractor`, `pitch` | 5–7 days |
| 4 | Model training | trained `emotionModel.mat`, confusion matrix | Deep Learning Toolbox, `trainNetwork` | 7–10 days |
| 5 | Stress engine | `ai.stressEngine` + calibration set | base MATLAB | 3–5 days |
| 6 | Explainable AI | `ai.explainPrediction` | Deep Learning Toolbox | 3–4 days |
| 7 | GUI | `gui.SERApp` wired to engines | App Designer / `uifigure` | 6–8 days |
| 8 | Test + deploy | test suite green, demo script, packaged app | `matlab.unittest`, `compiler` (opt.) | 4–6 days |

Total ≈ **6–8 weeks** for one developer. Build vertically: get one file end-to-end (Phase 2→3→4 on a tiny subset) before scaling the dataset.

---

## 3. Dataset selection

| Dataset | Samples (approx) | Emotions | Quality | Difficulty | Notes |
|---|---|---|---|---|---|
| **RAVDESS** | ~1,440 speech | 8 (incl. calm, surprise, disgust) | studio, clean | easy–medium | 24 actors, balanced, gender-balanced; best starting point |
| **TESS** | ~2,800 | 7 | clean | easy | only 2 (older female) speakers → weak speaker generalisation |
| **CREMA-D** | ~7,440 | 6 | varied | medium–hard | 91 speakers, diverse → best for generalisation |
| **SAVEE** | ~480 | 7 | clean | medium | only 4 male speakers → biased |

**Recommendation:** train on **RAVDESS + CREMA-D** (clean balance + speaker diversity), keep **TESS** or **SAVEE** as an unseen test set to show cross-corpus generalisation — a strong point in a viva. Map all corpora to a shared 5-class label set (Angry, Happy, Sad, Neutral, Fear); drop classes that don't appear in all sets, or keep them only where present and note the imbalance. All are actor-portrayed (acted) emotion — say so; it is the standard limitation of the field.

---

## 4. Preprocessing pipeline (`sigproc.preprocessAudio`)

| Step | Why | MATLAB | Output |
|---|---|---|---|
| Mono mix | one channel for analysis | `mean(x,2)` | `[N×1]` |
| Resample 16 kHz | fix rate across corpora; speech band fits | `resample` | `[N×1]` |
| Pre-emphasis (0.97) | boost HF, flatten spectral tilt | `filter([1 -0.97],1,x)` | `[N×1]` |
| Spectral subtraction | remove stationary background noise | `stft`/`istft` | denoised |
| VAD / silence removal | keep speech, drop silence/leading noise | `detectSpeech` | trimmed |
| Peak normalize | level invariance across mics | `x/max(abs(x))` | `[-1,1]` |

Energy-gate fallback is built in if `detectSpeech` is unavailable. Framing uses 25 ms window / 10 ms hop (`hann`) — the standard for speech.

---

## 5. Feature extraction (`sigproc.extractFeatures`)

Per-frame **sequence** (channels of `feat.seq`): MFCC(13) + Δ(13) + ΔΔ(13), spectral centroid / rolloff / flux / spread / entropy, pitch, ZCR, short-time energy, harmonic ratio.

| Feature | Captures | Meaning |
|---|---|---|
| **MFCC** | timbre / vocal-tract shape | DCT of log mel energies; the workhorse SER feature |
| **Δ / ΔΔ MFCC** | how timbre moves | 1st/2nd time derivatives → dynamics |
| **Pitch (F0)** | intonation, arousal | fundamental frequency; high/variable under anger, fear, stress |
| **Short-time energy / RMS** | loudness | high in anger, low in sadness |
| **ZCR** | voicing / noisiness | high for fricatives/noisy speech |
| **Spectral centroid/rolloff** | brightness | shifts up with tension/arousal |
| **Spectral flux** | rate of spectral change | high when articulation is rapid |
| **Harmonic ratio** | voice quality | breathy/tense voice changes harmonicity |

Per-utterance **summary** (`feat.summary`) drives stress + XAI: `pitchMean/Std/Range`, `energyMean/Std/CV`, `rmsMean`, `zcrMean`, `centroidMean`, `fluxMean`, `speakingRate`, `duration`.

LPC/formants are easy to add (`lpc`, then `roots`→ formant frequencies) if your rubric wants them explicitly — flagged in the report template but omitted from the live path to keep latency low.

---

## 6. Deep learning model (`ai.buildCNNLSTM`)

```
sequenceInput(F)
 → conv1d(5, 64) → BN → ReLU → maxpool1d(2)
 → conv1d(5,128) → BN → ReLU → maxpool1d(2)
 → biLSTM(128, last)
 → dropout(0.3)
 → fc(64) → ReLU → fc(numClasses) → softmax → classification
```

- **CNN front-end:** learns local spectro-temporal patterns and halves the time axis twice (cheaper, denoised input to the LSTM).
- **BiLSTM:** models how those patterns evolve over the utterance, both directions (emotion cues are non-causal).
- **Training:** Adam, lr 1e-3, piecewise decay ×0.5 every 15 epochs, 40 epochs, batch 32, gradient clip 1, right-padding, validation every 20 iters. Defined in `ai.trainEmotionModel`.
- **Version note:** `convolution1dLayer`/`maxPooling1dLayer` need **R2021b+**. For **R2024a+**, swap `softmax+classificationLayer`/`trainNetwork` for `trainnet(...,"crossentropy")` — same graph otherwise.

Per-feature standardisation (`mu`,`sigma`) is computed on the training set and **saved with the model**, then re-applied at inference — a common bug source if forgotten.

---

## 7. Stress detection engine (`ai.stressEngine`)

Composite index in `[0,1]` from normalised acoustic correlates, weighted:

| Correlate | Weight | Direction |
|---|---|---|
| Pitch instability (F0 std) | 0.30 | ↑ stress |
| Pitch level (F0 mean) | 0.15 | ↑ stress |
| Energy variation (CV) | 0.25 | ↑ stress |
| Speaking rate | 0.20 | ↑ stress |
| Spectral flux | 0.10 | ↑ stress |

`<0.40` Low · `0.40–0.70` Moderate · `≥0.70` High. The normalisation ranges in `local_defaultCalib` are **placeholders — replace them** with values measured on your calibration recordings (relaxed reading vs. timed-task speech). Returns a component table so the GUI/XAI can show *which* correlate drove the level.

---

## 8. Explainable AI (`ai.explainPrediction`)

Two complementary techniques:

1. **Occlusion importance (model-faithful).** Zero each feature *group* (MFCC, Δ, ΔΔ, pitch, energy, ZCR, spectral) in turn, re-run the net, record the drop in the predicted-class probability. Largest drop = most influential group for *this* clip. Grad-CAM is image-oriented and awkward on sequences; occlusion is the honest choice here.
2. **Acoustic-profile reasoning (human-readable).** Compares summary features to expected emotion profiles → "high energy + fast speaking rate + high pitch variation", plus the stress index.

Output is a ranked importance table + a formatted text block shown in the GUI's Explanation panel.

---

## 9. GUI (`gui.SERApp`)

Single window, programmatic `uifigure` (so it lives in source control). Layout:

- **Controls bar:** Record (with duration), Stop, Upload, Export.
- **Signal panel:** waveform + spectrogram.
- **Features panel:** MFCC heatmap + Explanation text.
- **Prediction panel:** emotion label, confidence gauge, stress gauge + level.
- **History table:** time, file, emotion, confidence, stress, index.

It calls only `sigproc.*`/`ai.*`/`utils.*`. If no model is present, stress + plots still work and emotion shows "no model" — so you can demo Module 1 before Module 2 is trained. If your rubric demands the literal `.mlapp`, recreate this layout in App Designer and paste the `analyse` logic into a callback; the engine calls are unchanged.

---

## 10. File structure

```
SER_Stress_System/
├── main_demo.m              end-to-end CLI run (no GUI) — best for testing
├── +sigproc/                    MODULE 1: preprocessAudio, extractFeatures
├── +ai/                     MODULE 2: buildCNNLSTM, trainEmotionModel,
│                                      predictEmotion, stressEngine, explainPrediction
├── +gui/                    MODULE 3: SERApp (uifigure)
├── +utils/                  recordMic, loadAudioFile, exportReport
├── models/                  emotionModel.mat (created by training)
├── datasets/                RAVDESS root: audio_speech_actors_01-24/Actor_NN/
├── reports/                 history.csv + report_*.txt (created at runtime)
├── tests/                   test_pipeline.m (matlab.unittest)
└── docs/                    SRS, UML, report (templates below)
```

---

## 11. Testing strategy

- **Unit** (`tests/test_pipeline.m`): preprocessing output shape/normalisation, feature finiteness, stress index in `[0,1]`. Runs without a trained model.
- **Integration:** `main_demo("clip.wav")` on a held-out file → expect a label + stress + report card.
- **Model:** held-out validation accuracy + confusion matrix (in `trainEmotionModel`); cross-corpus test on the unseen set.
- **GUI:** manual checklist — record auto-stops at N s; upload rejects cancel; export writes a file; "no model" path degrades gracefully.
- **Performance:** time per clip (target < 1 s for ~4 s audio on CPU); note it is *near-real-time per utterance*, not streaming frame-by-frame.

Checklist (copy to docs):
```
[ ] Record → analyse populates all panels
[ ] Upload .wav / .flac / .mp3 each works
[ ] Confidence + stress gauges update
[ ] Explanation lists top feature groups
[ ] History row added per run
[ ] Export creates reports/report_*.txt and appends history.csv
[ ] No-model mode shows stress only, no crash
[ ] runtests("tests/test_pipeline.m") all pass
```

---

## 12. Live demonstration script

1. Launch: `gui.SERApp` (from project root, after `ai.trainEmotionModel` has produced a model).
2. Click **Record**, say one neutral sentence → show waveform, spectrogram, MFCC populate.
3. Read the **Emotion** label + **confidence**; read the **Stress** gauge.
4. Open the **Explanation** panel — point to the top feature groups and the acoustic reasons.
5. Click **Upload**, load a clearly *angry* clip → contrast the result and the explanation.
6. Click **Export** → open `reports/report_*.txt` and `history.csv` to show persistence.
7. Fallback: if the mic misbehaves, run `main_demo("samples/angry.wav")` — identical pipeline, no GUI risk. **Always have this fallback ready.**

---

## 13. Documentation templates (`docs/`)

**SRS skeleton**
- Purpose, scope, definitions.
- **Functional:** FR1 record ≤30 s; FR2 accept wav/flac/mp3; FR3 classify into 5 emotions with confidence; FR4 estimate 3-level stress; FR5 show waveform/spectrogram/MFCC; FR6 explain prediction; FR7 export CSV + report.
- **Non-functional:** NFR1 analyse a 4 s clip < 1 s on CPU; NFR2 GUI responsive (progress dialog during analysis); NFR3 runs on MATLAB R2021b+ with Audio + Deep Learning Toolboxes; NFR4 graceful degradation without a model; NFR5 reproducible training (`rng` seed).

**UML to produce** (any tool; describe in text in the report):
- **Use case:** actor *User* → Record, Upload, View results, View explanation, Export.
- **Activity:** acquire → preprocess → extract → [model?] → predict + stress → explain → display → export.
- **Sequence:** GUI → DSP.preprocess → DSP.extract → AI.predict → AI.stress → AI.explain → GUI.display.
- **Class:** `SERApp` ──uses──> packages `sigproc`, `ai`, `utils`; data structs `feat`, `pred`, `stress`, `xai`.

**Report structure:** Abstract · Introduction · Literature review · Methodology (these 3 modules) · Implementation · Experiments & results (confusion matrix, cross-corpus, latency) · Limitations (acted data, unsupervised stress, per-utterance not streaming) · Conclusion & future work.

---

## 14. Advanced features — ranked by (impact ÷ effort)

| Feature | Impact | Effort | Verdict |
|---|---|---|---|
| Real-time emotion **timeline** (slide a window, plot label over time) | High | Low–Med | **Do first** — visually impressive, reuses the pipeline |
| **PDF report** generation (`exportgraphics`/Report Generator) | Med | Low | Easy polish |
| **Trend analytics** over history (counts, avg confidence) | Med | Low | Easy, good for the dashboard |
| **Audio database** management (catalog past clips + results) | Med | Med | Useful, moderate work |
| **Speaker identification** (i-vector/x-vector or simple GMM) | High | High | Strong but a project on its own |
| **Multi-speaker / diarization** | High | High | Hard; only if time allows |
| **Voice biometrics** | Med | High | Skip for an FYP scope |

Start with the **timeline** + **PDF/analytics** — high return for low risk.

---

## How to run

```matlab
cd SER_Stress_System          % project root, so +sigproc/+ai/+gui are on the path
% 1) point at the RAVDESS root (labels are parsed from filenames):
ai.trainEmotionModel("datasets/audio_speech_actors_01-24")
% 2) command-line end-to-end:
main_demo("path\to\clip.wav")   % or main_demo to use the mic
% 3) GUI:
gui.SERApp
% 4) tests:
runtests("tests/test_pipeline.m")
```

**Requirements:** MATLAB **R2021b+**, **Audio Toolbox**, **Deep Learning Toolbox**. A microphone for live capture. The stress engine and all plots work without a trained model; emotion + XAI need `models/emotionModel.mat`.

## Honest scope statement

This is a complete, runnable **scaffold**: the DSP and stress paths are fully functional and unit-tested in logic; the model architecture and training pipeline are correct MATLAB but **untrained** — accuracy depends entirely on the dataset you supply and train on. "Production-quality" here means clean structure, error handling, and graceful degradation — not a benchmarked, deployed product. Train, then report the real numbers.
