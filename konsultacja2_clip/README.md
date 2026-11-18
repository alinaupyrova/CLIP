## 8. Podsumowanie i wnioski

| Zbiór | Artykuł | Nasze | Różnica | Ocena |
|---|---|---|---|---|
| CIFAR-100 | 65,1% | 65,08% | −0,02 p.p. | zgodny (artykuł w 95% CI) |
| Oxford-IIIT Pet | 87,0% | 87,14% | +0,14 p.p. | zgodny (artykuł w 95% CI) |
| CIFAR-10 | 91,3% | 89,83% | −1,47 p.p. | bliski (mała, ale istotna różnica) |
| EuroSAT | 49,4% | 46,23% | −3,17 p.p. | wyraźna różnica, porównanie przybliżone |

* **CIFAR-10 / CIFAR-100 (reprodukcja):** na CIFAR-100 wynik artykułu odtworzony praktycznie dokładnie
  (−0,02 p.p., w 95% CI). Na CIFAR-10 różnica wynosi −1,47 p.p.: jest mała, ale większa niż niepewność statystyczna.
  Nie wynika z błędu procedury – ta sama procedura daje zgodność na CIFAR-100 i Oxford-IIIT Pet. Prawdopodobne
  przyczyny to szczegóły środowiska: obliczenia w fp32 zamiast fp16, nowsze wersje bibliotek, powiększanie obrazów 32→224 px.
* **Oxford-IIIT Pet (zmiana domeny):** 87,14% vs 87,0% (mean per-class accuracy, +0,14 p.p., w 95% CI) – pełna
  zgodność na oficjalnym podziale testowym.
* **EuroSAT (zmiana domeny):** 46,23% vs 49,4% (−3,17 p.p.) na całym zbiorze (27 000 obrazów); na próbce
  5 000 obrazów wynik był zbliżony (45,24%), więc różnica nie wynika z doboru obrazów po naszej stronie. Zbiór nie ma
  oficjalnego podziału testowego (autorzy użyli nieznanego podzbioru), a zdjęcia satelitarne są odległe od danych
  treningowych CLIP – wynik jest tu bardzo wrażliwy na szczegóły: sam dobór szablonu promptu zmienia go o ok. 10 p.p.
  (36,32% → 46,23%, sekcja 7.1).
* **Wniosek ogólny:** procedura zero-shot odtwarza wyniki artykułu na zbiorach z oficjalnym podziałem testowym
  (różnice ≤ 1,5 p.p.). Różnica rośnie, gdy dane są odległe od danych treningowych CLIP i gdy nie ma oficjalnego
  podziału testowego. Duży wpływ szablonu promptu na EuroSAT motywuje pytanie badawcze Konsultacji 3.
