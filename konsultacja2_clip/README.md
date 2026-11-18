# Konsultacja 2 – reprodukcja CLIP ViT-B/32 zero-shot (CIFAR-10 / CIFAR-100)

Zespół 9: Tymur Rozhkovskyi · Alina Upyrova · Radosław Tkaczyk

Cel: odtworzenie wyniku z Tabeli 11 artykułu CLIP (Radford i in., 2021):
CIFAR-10 – 91,3%, CIFAR-100 – 65,1% top-1 accuracy (zero-shot, ViT-B/32).

## Instalacja

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uruchomienie

```bash
jupyter notebook konsultacja2_clip_reprodukcja.ipynb
```

W Jupyterze: `Kernel → Restart & Run All`. Nie są potrzebne żadne ręczne kroki:
- dane CIFAR-10/100 pobierają się przez `torchvision` do `data/` (~160 MB),
- wagi ViT-B/32 pobierają się przez `clip.load()` do `~/.cache/clip` (~340 MB),
- embeddingi obrazów są zapisywane w `cache/` – kolejne uruchomienia trwają kilkadziesiąt sekund.

Urządzenie wybierane automatycznie: CUDA → Apple MPS (M1/M2/M3) → CPU.
Najdłuższy krok (embeddingi 20 000 obrazów): ok. 2–5 min na Apple M3 (MPS), 10–20 min na CPU.

## Struktura

| Plik / katalog | Zawartość |
|---|---|
| `konsultacja2_clip_reprodukcja.ipynb` | notatnik wykonywalny od początku |
| `requirements.txt` | zależności (CLIP przypięty do commita) |
| `results/environment.json` | wersje bibliotek, urządzenie, konfiguracja |
| `results/summary_reprodukcja.csv` | porównanie z artykułem: top-1, różnica, 95% CI, macro-F1, top-5 |
| `results/predictions_*.csv` | surowe predykcje dla każdego obrazu testowego |
| `results/per_class_*.csv` | dokładność per klasa i najczęstsze pomyłki |
| `results/kontrola_ensemble.csv` | 1 szablon vs 18 szablonów (kontrola wrażliwości) |
| `results/*.png` | wykresy użyte w prezentacji |

## Odstępstwa od kodu autorów

- Model ładowany w `float32` (na CPU, potem przenoszony na MPS/CPU); autorzy na GPU używali `float16`.
- Poza tym procedura jest zgodna z oficjalnym notebookiem autorów (*Prompt_Engineering_for_ImageNet.ipynb*):
  te same szablony i nazwy klas (`data/prompts.md`), ten sam `preprocess`, normalizacja L2, podobieństwo kosinusowe.

## Użycie AI

Narzędzia AI (Claude) wspierały przygotowanie szkieletu notatnika i opisów. Zespół zweryfikował kod,
uruchomił notatnik i odpowiada za wyniki oraz ich interpretację.
