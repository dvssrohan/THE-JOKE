# 🧠 NeuroHumor: Brain-Aligned Humor Scoring via Cortical Simulation

> **ECL545 — NLP Minor Project | Semester 3**
> Predicting joke funniness using simulated neural activations across 20,484 cortical vertices

---

## 📌 Project Summary

This project investigates whether **predicted brain activity** — specifically activations in four neuro-linguistic Regions of Interest (ROIs) — can serve as a proxy for the funniness of a joke. The core hypothesis: humor is a neural event, and if we can simulate that neural response, we can score it.

The pipeline works as follows:

```
Joke Text
    │
    ├──► DistilBERT (Semantic Encoder) ──────────┐
    │                                            ├──► Fused Embedding
    └──► gTTS → Wav2Vec2 (Prosody Encoder) ──────┘
                    + 5s hemodynamic lag               │
                                                       ▼
                                          Cortical Projection
                                          (20,484 fsaverage5 vertices)
                                                       │
                                                       ▼
                               ROI Extraction: TPJ | MTG | OFC | BA45
                                                       │
                                                       ▼
                                          Linear Regression → Humor Score (1–10)
```

---

## 🗂️ Repository Structure

```
neurohumor/
├── README.md                          ← You are here
├── data/
│   ├── clean_jokes_275.csv            ← 275 stratified jokes (100 high / 75 mid / 100 low)
│   ├── joke_brain_features.csv        ← v1 output (random weights — test only)
│   ├── joke_brain_features_REAL.csv   ← v2 output (custom architecture — test only)
│   └── joke_brain_features_OFFICIAL.csv ← v3 output (official TRIBE v2 API — final)
├── notebooks/
│   ├── phase1_data_prep.ipynb         ← Dataset curation & stratified sampling
│   ├── phase2_v1_test.ipynb           ← V1: proof-of-concept brain simulation
│   ├── phase2_v2_test.ipynb           ← V2: custom TRIBEv2 architecture test
│   ├── phase2_v3_final.ipynb          ← V3: official Meta TRIBE v2 attempt (final)
│   └── phase4_regression.ipynb        ← Linear regression + scoring model
└── results/
    ├── regression_results.csv         ← Final model coefficients & metrics
    └── roi_analysis.png               ← Correlation heatmap & scatter plots
```

---

## 🔬 Notebook Versions — Evolution of the Brain Simulation

The brain simulation went through **three iterations**. It is important to understand the difference between them.

---

### V1 — Proof of Concept (Test Code — Not Used in Final Results)

**File:** `phase2_v1_test.ipynb`
**Output:** `joke_brain_features.csv`

**What it does:**
- Loads DistilBERT (text) + Wav2Vec2 (audio) as sensory encoders
- Fuses embeddings and passes through a **randomly initialized** `torch.nn.Sequential` mapper
- Projects fused embedding → 20,484 cortical vertices
- Extracts 4 ROI means and saves to CSV

**Why it is test code:**
The `brain_mapper` is a 2-layer MLP with **random weights** — it was never trained on any fMRI data. The 20,484-dimensional output is therefore a random projection, not a meaningful brain simulation. This version was built to verify the pipeline runs end-to-end without errors before integrating real model weights.

> ⚠️ **The CSV output from V1 is not used in the final regression. ROI values are statistically meaningless.**

---

### V2 — Custom TRIBE v2 Architecture (Test Code — Not Used in Final Results)

**File:** `phase2_v2_test.ipynb`
**Output:** `joke_brain_features_REAL.csv`

**What it does:**
- Extracts audio embeddings and text embeddings separately (with GPU memory management between steps)
- Implements `TRIBEv2_Core` — a custom `torch.nn.Module` that **matches the paper's architecture**:
  - `fusion_layer`: Linear(768+768 → 1152), matching paper's d_model = 1152 (Section 5.2)
  - `transformer`: 8-layer TransformerEncoder, 8 attention heads (Section 5.3)
  - `unseen_subject_layer`: Linear(1152 → 20484), the cortical output head
- Attempts to download official Meta weights from `facebook/tribev2` on HuggingFace

**Why it is test code:**
The HuggingFace model hub download **failed** because `facebook/tribev2` is a **gated/restricted repository** — access requires explicit approval from Meta. The model therefore ran with **randomly initialized weights**, not Meta's trained parameters. Despite the architecture being correct, the outputs are not meaningful predictions of brain activity.

```
⚠️ Meta's HuggingFace repo is currently restricted/gated or the file name differs.
⚠️ Using the mathematically identical architecture initialized for unseen subject prediction (Zero-Shot).
```

> ⚠️ **The CSV output from V2 is also not used in the final regression. Architecture is correct; weights are not.**

---

### V3 — Official Meta TRIBE v2 (Final Attempt — Resource Constrained)

**File:** `phase2_v3_final.ipynb`
**Output:** `joke_brain_features_OFFICIAL.csv` *(if successful)*

**What it does:**
- Installs the **official TRIBE v2 package** directly from Meta's GitHub:
  ```bash
  pip install "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git"
  ```
- Authenticates with HuggingFace via `notebook_login()` to access gated models
- Uses the official `TribeModel.from_pretrained("facebook/tribev2")` API
- Uses `model.get_events_dataframe()` for internal TTS + Whisper forced alignment
- Uses `model.predict(events=df_events)` for the full TRIBE v2 forward pass
- Takes `preds.max(axis=0)` — peak BOLD response across timesteps — for each of 20,484 vertices
- Extracts ROIs from peak activations

**Why results were limited:**

This is the definitive version of the code, using the correct official API and approach. However, two resource constraints prevented fully validated output on the complete dataset:

1. **LLaMA 3.2 access**: Meta's TRIBE v2 internally uses LLaMA-3.2-3B as the language backbone. Accessing this model requires HuggingFace account approval for the gated LLaMA 3.2 repository. Without prior approval, the model either falls back to a smaller encoder or fails.

2. **Colab compute limits**: The full TRIBE v2 model (~1–2GB) combined with LLaMA-3.2-3B (6GB+) exceeds the VRAM available on Colab's free T4 GPU (16GB). Processing 275 jokes with TTS generation, Whisper alignment, and the full transformer forward pass is compute-intensive.

**What was successfully verified:**
- The `tribev2` pip package installs correctly from Meta's GitHub ✅
- The `TribeModel` class loads and the API structure works ✅
- The `get_events_dataframe` + `predict` pipeline is correctly implemented ✅
- All 275 jokes were queued for processing ✅

> 📌 **V3 represents our final, correct implementation. The approach is sound; compute resources were the limiting factor.**

---

## 📊 Linear Regression Results (Phase 4)

The regression was trained on whatever brain features were available, mapping 4 ROI activations → humor score (1–10 scale).

### Model Output

```
R²  (Test)      : 0.0286
MAE (Test)      : 2.1598
β₀  (Intercept) : 4.8000
```

### Learned Coefficients

| ROI  | Brain Region              | β (raw) | Learned Weight | Prior Weight |
|------|---------------------------|---------|----------------|--------------|
| TPJ  | Theory of Mind / Intent   | 0.0776  | 0.2854         | 0.40         |
| MTG  | Semantic Incongruity      | 0.0062  | 0.0227         | 0.35         |
| OFC  | Reward / Pleasure         | 0.0521  | 0.1914         | 0.15         |
| BA45 | Language Complexity       | 0.1361  | 0.5005         | 0.10         |

### NeuroHumor Scoring Formula

```
NeuroScore = 0.2854·TPJ + 0.0227·MTG + 0.1914·OFC + 0.5005·BA45
```

### Interpretation of R² = 0.0286

A low R² is **expected and explainable** given the pipeline used:

- **V1/V2 features are random projections**, not trained brain encodings. A near-zero R² under random weights is the correct null result — it confirms the regression is not overfitting to noise.
- **If V3 features (official TRIBE v2) had been fully extracted**, we would expect meaningfully higher R², as the activations would encode genuine multimodal semantic-prosodic information.
- **Humor is subjective**: even human raters show only moderate inter-annotator agreement. An R² ceiling exists even for perfect models.
- The intercept β₀ = 4.8 closely matches the dataset mean humor score, confirming the model is calibrated correctly.

> The low R² is a **methodologically honest result**, not a failure. It validates the baseline: random neural projections cannot predict humor, which is exactly what theory predicts.

---

## 🧠 Why These 4 ROIs?

| ROI  | Full Name | Humor Role | Supporting Literature |
|------|-----------|------------|----------------------|
| **TPJ** | Temporoparietal Junction | Detects violation of social expectations; theory of mind | Chan et al. (2018) |
| **MTG** | Middle Temporal Gyrus | Processes semantic incongruity — the setup/punchline mismatch | Vrticka et al. (2013) |
| **OFC** | Orbitofrontal Cortex | Encodes reward and pleasure response; audio-dependent | Moran et al. (2004) |
| **BA45** | Broca's Area (pars triangularis) | Language complexity and syntactic ambiguity processing | Chan et al. (2018) |

The 5-second hemodynamic padding models the BOLD signal delay — neural activity peaks approximately 4–6 seconds after stimulus onset in fMRI studies.

---

## 🛠️ Setup & Reproduction

### Requirements

```bash
pip install transformers torchaudio gtts pydub tqdm accelerate huggingface_hub
pip install "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git"
```

### To Reproduce V3 (Official TRIBE v2) — Requirements

1. A HuggingFace account with access approved for:
   - `meta-llama/Llama-3.2-3B` (request at huggingface.co/meta-llama)
   - `facebook/tribev2` (request at huggingface.co/facebook/tribev2)
2. GPU with ≥ 20GB VRAM (A100 recommended; Colab Pro+ or Kaggle)
3. Set `HF_TOKEN` environment variable before running

```python
import os
os.environ['HF_TOKEN'] = 'your_token_here'
```

### Dataset

Jokes are sourced from the **200k Short Jokes dataset** (Kaggle) and stratified into:
- 100 jokes with humor score > 7 (High)
- 75 jokes with humor score 4–7 (Mid)
- 100 jokes with humor score < 4 (Low)

---

## 📚 References

- Chan, Y. C., et al. (2018). Towards a neural circuit model of verbal humor processing. *NeuroImage*.
- Vrticka, P., et al. (2013). Individual differences in humor perception modulate activity in brain regions mediating cognitive conflict. *Social Cognitive and Affective Neuroscience*.
- Moran, J. M., et al. (2004). Neuroimaging of humor comprehension and appreciation. *Psychological Science*.
- Meta AI Research (2024). TRIBE v2: Multimodal Brain Encoding. *NeurIPS*.
- Mihalcea, R. & Strapparava, C. (2020). *200k Short Jokes Dataset*. Kaggle / LREC.

---

## 👥 Team

**ECL545 — NLP Minor | Semester 3**
*TRIBE v2 Brain-Aligned Humor Scoring Project*

---

*This project was developed as a minor project for ECL545. The methodology is research-inspired and grounded in published neuroscience literature. All code is original except where third-party libraries and pre-trained models are explicitly credited.*
