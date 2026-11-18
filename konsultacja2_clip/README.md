# CLIP zero-shot reproduction – CIFAR-10, CIFAR-100, Oxford-IIIT Pet, EuroSAT

**Course:** Analiza danych obrazowych i multimedialnych – Warsaw University of Technology, Faculty of Electrical Engineering
**Team 9:** Tymur Rozhkovskyi · Alina Upyrova · Radosław Tkaczyk
**Stage:** Consultation 2 – reproduction of a published result

We reproduce the **zero-shot** accuracy of **CLIP ViT-B/32** reported in Table 11 of
Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision*
([arXiv:2103.00020](https://arxiv.org/abs/2103.00020)), using the official code
([openai/CLIP](https://github.com/openai/CLIP), MIT license).

After Consultation 1 the instructor asked us to also **verify the result on 2 datasets other than CIFAR**,
so the same procedure is applied to Oxford-IIIT Pet and EuroSAT. Everything is in **one notebook:
`clip_zero_shot_4_zbiory.ipynb`**.

| Dataset | Domain | Classes | Resolution | Role | Paper (Table 11, ViT-B/32) |
|---|---|---|---|---|---|
| CIFAR-10 | general objects | 10 | 32×32 | reproduction | 91.3% (accuracy) |
| CIFAR-100 | general objects | 100 | 32×32 | reproduction | 65.1% (accuracy) |
| Oxford-IIIT Pet | cat & dog breeds | 37 | various, larger | domain shift | 87.0% (mean per-class accuracy) |
| EuroSAT | satellite images | 10 | 64×64 | domain shift | 49.4% (accuracy) |

---

## TL;DR – what to expect

| Step | What happens | Time |
|---|---|---|
| 1. Install | `pip install -r requirements.txt` | 2–5 min |
| 2. **Download CIFAR** | ~325 MB from the University of Toronto server | ** ~40 min** (the server is slow, ~70 kB/s) |
| 3. Download CLIP weights | 338 MB, done automatically by `clip.load()` | < 1 min |
| 4. Download Oxford-IIIT Pet + EuroSAT | ~0.9 GB (Oxford server + Hugging Face) | ~20 min |
| 5. Image embeddings | ~28 700 test images (4 datasets) through ViT-B/32 | 5–10 min on Apple M3 (MPS), 20–40 min on CPU |
| 6. Prompts, metrics, plots | authors' prompt templates, metrics, bootstrap CI | seconds |

Steps 2–5 happen **only once**. Data is kept in `data/`, weights in `~/.cache/clip`,
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

### 2a. CIFAR-10 and CIFAR-100

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
* **Teammates:** ask for the `.tar.gz` files (e.g. via Google Drive) and put them into your own `data/` folder. With the files in place you can skip the download.

### 2b. Oxford-IIIT Pet and EuroSAT

| Dataset | Domain | Classes | Test data | Paper (Table 11) | Download |
|---|---|---|---|---|---|
| Oxford-IIIT Pet | cat & dog breeds, natural photos | 37 | official `test` split, 3 669 images | 87.0% (mean per-class accuracy) | ~800 MB from `www.robots.ox.ac.uk` |
| EuroSAT (RGB) | satellite images 64×64 | 10 | fixed sample of 5 000 images (500 per class, seed 42) | 49.4% (accuracy) | ~90 MB from Hugging Face |

Both datasets download automatically when the notebook runs. If the Oxford server is slow, download Pets
manually in a separate Terminal tab (`Cmd + T`) first:

```bash
mkdir -p ~/Desktop/konsultacja2_clip/data/oxford-iiit-pet
cd ~/Desktop/konsultacja2_clip/data/oxford-iiit-pet
curl -fL -O https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz & curl -fL -O https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz & wait
tar -xzf images.tar.gz && tar -xzf annotations.tar.gz
ls     # must show the folders: images  annotations
```

| File | Expected size | MD5 |
|---|---|---|
| `images.tar.gz` | ~755 MB | `5c4f3ee8e5d25df40f4fd59a7f44e54c` |
| `annotations.tar.gz` | ~18 MB | `95a8c909bbe2e81eed6a22bccdf3f68f` |

After extracting, the `.tar.gz` files can be deleted. EuroSAT is small and fast; `torchvision` saves it to `data/eurosat/`.

**Note on EuroSAT:** `torchvision` provides the whole dataset (27 000 images) without a train/test split.
We use a fixed, stratified random sample (indices saved in `results/eurosat_test_indices.csv`), so the comparison
with the paper is approximate for this dataset.

## 3. Running the notebook

```bash
cd ~/Desktop/konsultacja2_clip
source .venv/bin/activate
jupyter notebook clip_zero_shot_4_zbiory.ipynb
```

Choose **Run → Restart Kernel and Run All Cells…** No manual steps are needed. The notebook does the following:

1. **Environment.** Library versions, device and configuration are printed and saved to `results/environment.json`.
   The device is selected automatically: CUDA → Apple MPS → CPU.
2. **Model.** `clip.load("ViT-B/32", jit=False)` loads the official weights in float32. A correct load prints `logit_scale = 100.00`.
3. **Data.** The notebook finds the datasets in `data/` (CIFAR-10, CIFAR-100, Oxford-IIIT Pet, EuroSAT) and skips the download.
4. **Prompts.** For every class, the class name is inserted into the authors' templates from
   [`data/prompts.md`](https://github.com/openai/CLIP/blob/main/data/prompts.md):
   18 templates for CIFAR (e.g. `a blurry photo of a {}.`), 1 for Pets (`a photo of a {}, a type of pet.`),
   3 for EuroSAT (e.g. `a centered satellite photo of {}.`).
   The text embeddings are L2-normalised, averaged (prompt ensembling) and normalised again.
   This gives the zero-shot classifier, with no training on the target datasets.
5. **Image embeddings.** This is the expensive cell (~28 700 images). The result is cached in `cache/`. Delete `cache/` to recompute.
6. **Metrics.** For each dataset the notebook computes the paper's metric (accuracy, or mean per-class accuracy for Pets),
   the difference from the paper in percentage points, a 95% bootstrap confidence interval, top-1, top-5 and macro-F1.
7. **Comparison with the paper** (one table and plot for all 4 datasets) and **error analysis**: authors' templates vs one
   simple template, confusion matrices (CIFAR-10, EuroSAT), per-class accuracy, ambiguous CIFAR-100 class names
   (`mouse`, `ray`, `tank`, …) and examples of wrong predictions.

The harmless `FutureWarning: torch.jit.load is deprecated` may appear. It comes from the CLIP library and does not affect the results.

## 4. Outputs (`results/`)

| File | Content |
|---|---|
| `summary_4_zbiory.csv` | paper vs ours for all 4 datasets: metric, difference [p.p.], 95% CI, top-1, mean per-class, macro-F1, top-5 |
| `environment.json` | Python / library versions, device, configuration |
| `predictions_<dataset>.csv` | raw prediction for every test image (true, predicted, probability, top-5) |
| `per_class_<dataset>.csv` | per-class accuracy and the most frequent confusion |
| `kontrola_szablony.csv` | authors' templates vs one simple template |
| `eurosat_test_indices.csv` | indices of the EuroSAT test sample |
| `*.png` | figures used in the presentation |

## 5. Project structure

```
konsultacja2_clip/
├── clip_zero_shot_4_zbiory.ipynb         # notebook: all 4 datasets, runs top to bottom
├── requirements.txt                      # dependencies, CLIP pinned to a commit
├── README.md
├── results/                              # outputs (committed)
├── data/                                 # datasets (NOT committed, see .gitignore)
└── cache/                                # cached image embeddings (NOT committed)
```

## 6. Deviations from the authors' setup

* The model runs in **float32** on Apple MPS / CPU. The authors used float16 on NVIDIA GPUs.
* Library versions are newer than in 2021 (see `results/environment.json`).
* EuroSAT: our own fixed test sample (no official split in `torchvision`), so the comparison is approximate.
* Everything else follows the authors' notebook *Prompt_Engineering_for_ImageNet.ipynb*: the same checkpoint,
  the same `preprocess`, the same class names and templates, L2 normalisation and cosine similarity.

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
* Parkhi O.M., Vedaldi A., Zisserman A., Jawahar C.V. (2012). *Cats and Dogs*. CVPR 2012.
* Helber P., Bischke B., Dengel A., Borth D. (2019). *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification*. IEEE JSTARS.
* Official CLIP repository (MIT): https://github.com/openai/CLIP
