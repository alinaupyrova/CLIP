# CLIP zero-shot reproduction – CIFAR-10 / CIFAR-100

**Course:** Analiza danych obrazowych i multimedialnych – Warsaw University of Technology, Faculty of Electrical Engineering
**Team 9:** Tymur Rozhkovskyi · Alina Upyrova · Radosław Tkaczyk
**Stage:** Consultation 2 – reproduction of a published result

We reproduce the **zero-shot** top-1 accuracy of **CLIP ViT-B/32** reported in Table 11 of
Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision*
([arXiv:2103.00020](https://arxiv.org/abs/2103.00020)), using the official code
([openai/CLIP](https://github.com/openai/CLIP), MIT license).

| Dataset | Model | Top-1 (paper) |
|---|---|---|
| CIFAR-10 | CLIP ViT-B/32 | 91.3% |
| CIFAR-100 | CLIP ViT-B/32 | 65.1% |

---

## TL;DR – what to expect

| Step | What happens | Time |
|---|---|---|
| 1. Install | `pip install -r requirements.txt` | 2–5 min |
| 2. **Download CIFAR** | ~325 MB from the University of Toronto server | **⚠️ ~40 min** (the server is slow, ~70 kB/s) |
| 3. Download CLIP weights | 338 MB, done automatically by `clip.load()` | < 1 min |
| 4. Image embeddings | 20 000 test images through ViT-B/32 | 2–5 min on Apple M3 (MPS), 10–20 min on CPU |
| 5. Prompts, metrics, plots | 18 prompt templates per class, accuracy, F1, bootstrap CI | seconds |

Steps 2–4 happen **only once**. Data is kept in `data/`, weights in `~/.cache/clip`,
image embeddings in `cache/`. Every later run of the notebook takes well under a minute.

---

## 1. Installation

Requires Python 3.11 (3.13 also works) and `git` (on macOS: `xcode-select --install`).

```bash
cd ~/Desktop/konsultacja2_clip        # or wherever you unpacked / cloned the project
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> Always run the commands **inside the project folder** (the one that contains `requirements.txt`).
> Otherwise you get `Could not open requirements file`.

## 2. Downloading the data (do this first – it is the slow part)

The notebook downloads CIFAR-10 and CIFAR-100 automatically via `torchvision`, but the source server
(`www.cs.toronto.edu`) is very slow – about **40 minutes** for both archives. It is faster to download
both files **in parallel in a separate Terminal tab before starting the notebook**:

```bash
mkdir -p ~/Desktop/konsultacja2_clip/data
cd ~/Desktop/konsultacja2_clip/data
rm -f cifar-10-python.tar.gz cifar-100-python.tar.gz     # remove partial downloads, if any
curl -fL -O https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz & curl -fL -O https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz & wait
```

The progress lines of the two downloads overlap in the Terminal – that is normal. When the prompt `%` returns, check the sizes:

```bash
ls -lh
```

| File | Expected size | MD5 (checked automatically by torchvision) |
|---|---|---|
| `cifar-10-python.tar.gz` | ~163 MB | `c58f30108f718f92721af3b95e74349a` |
| `cifar-100-python.tar.gz` | ~161 MB | `eb9058c3a382ffc7106e4002c42a8d85` |

If a file is only a few hundred bytes, the server returned an error page: delete it and run the `curl` command again.

**Notes**
* The server does **not** support resuming (`curl -C -` fails with `Cannot resume`), so an interrupted download must start from zero.
* Do not run the notebook while `curl` is still downloading – both would write to the same file.
* Do not paste commands into the Terminal tab where Jupyter is running. Open a new tab with `Cmd + T`.
* **Teammates:** ask for the two `.tar.gz` files (e.g. via Google Drive) and put them into your own `data/` folder. With the files in place you can skip the 40-minute download.

## 3. Running the notebook

```bash
source .venv/bin/activate
jupyter notebook konsultacja2_clip_reprodukcja.ipynb
```

In Jupyter choose **Run → Restart Kernel and Run All Cells…** No manual steps are needed:

1. **Environment.** Library versions, device and configuration are printed and saved to `results/environment.json`.
   The device is selected automatically: CUDA → Apple MPS → CPU.
2. **Model.** `clip.load("ViT-B/32", jit=False)` loads the official weights in float32. A correct load prints `logit_scale = 100.00`.
3. **Data.** The notebook finds the archives in `data/`, verifies the checksums and skips the download.
4. **Prompts.** For every class, the class name is inserted into the **18 templates** from the authors' file
   [`data/prompts.md`](https://github.com/openai/CLIP/blob/main/data/prompts.md), e.g. `a photo of a {}.` or `a blurry photo of a {}.`.
   The text embeddings are L2-normalised, averaged (prompt ensembling) and normalised again.
   This gives the zero-shot classifier, with no training on CIFAR.
5. **Image embeddings.** This is the expensive cell. The result is cached in `cache/`. Delete `cache/` to recompute.
6. **Metrics.** The notebook computes top-1 accuracy (the paper's metric), macro-F1, top-5, the difference from the paper
   in percentage points and a 95% bootstrap confidence interval.
7. **Comparison with the paper** (table and plot) and **error analysis**: 1 template vs 18 templates, confusion matrix,
   per-class accuracy, ambiguous class names (`mouse`, `ray`, `tank`, …) and examples of wrong predictions.

The harmless `FutureWarning: torch.jit.load is deprecated` may appear. It comes from the CLIP library and does not affect the results.

## 4. Outputs (`results/`)

| File | Content |
|---|---|
| `summary_reprodukcja.csv` | paper vs ours: top-1, difference [p.p.], 95% CI, macro-F1, top-5 |
| `environment.json` | Python / library versions, device, configuration |
| `predictions_CIFAR10.csv`, `predictions_CIFAR100.csv` | raw prediction for every test image (true, predicted, probability, top-5) |
| `per_class_CIFAR10.csv`, `per_class_CIFAR100.csv` | per-class accuracy and the most frequent confusion |
| `kontrola_ensemble.csv` | 1 template vs 18 templates |
| `*.png` | figures used in the presentation |

## 5. Project structure

```
konsultacja2_clip/
├── konsultacja2_clip_reprodukcja.ipynb   # notebook (runs top to bottom)
├── requirements.txt                      # dependencies, CLIP pinned to a commit
├── README.md
├── results/                              # outputs (committed)
├── data/                                 # CIFAR archives (NOT committed, see .gitignore)
└── cache/                                # cached image embeddings (NOT committed)
```

## 6. Deviations from the authors' setup

* The model runs in **float32** on Apple MPS / CPU. The authors used float16 on NVIDIA GPUs.
* Library versions are newer than in 2021 (see `results/environment.json`).
* Everything else follows the authors' notebook *Prompt_Engineering_for_ImageNet.ipynb*: the same checkpoint,
  the same `preprocess`, the same class names and 18 templates, L2 normalisation and cosine similarity.

## 7. Next step (Consultation 3)

Research question: *how does the form of the text prompt affect CLIP zero-shot accuracy?*
We will compare three variants: **A** – class name only (`cat`), **B** – a single template (`a photo of a cat.`),
**C** – an ensemble of templates. Results will include bootstrap confidence intervals.
The cached image embeddings are reused, so only the prompts change.

## Use of AI tools

AI tools (Claude) were used to draft the notebook skeleton and this README. The team reviewed the code,
ran the notebook and is responsible for the results and their interpretation.

## References

* Radford A., Kim J.W., Hallacy C. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML 2021. https://arxiv.org/abs/2103.00020
* Krizhevsky A. (2009). *Learning Multiple Layers of Features from Tiny Images*. Technical Report, University of Toronto.
* Official CLIP repository (MIT): https://github.com/openai/CLIP
