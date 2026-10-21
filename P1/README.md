# nn — sieć MLP do regresji i klasyfikacji

Program w Ruście, który trenuje sieć MLP (SGD + backpropagation) według opisu w pliku JSON
i zapisuje wyniki do plików CSV/JSON. Nie trzeba ruszać kodu w Ruście, żeby robić eksperymenty:
piszesz config, uruchamiasz binarkę, czytasz wyniki w Pythonie.

Prosty zestaw eksperymentów: [scripts/experiments.py](scripts/experiments.py).
Wykresy tworzy [scripts/plot_results.py](scripts/plot_results.py), bez dodatkowego systemu uruchamiania.
Dla jednego eksperymentu i ziarna służy [scripts/visualize_experiment.py](scripts/visualize_experiment.py).
Gotowe [figury do przejrzenia](figures/README.md) oraz [opis eksperymentów](EXPERIMENTS.md).
Raport czeka na wybór i ocenę figur.

Spis treści:
1. [Uruchomienie](#1-uruchomienie)
2. [Uruchomienie z Pythona](#2-uruchomienie-z-pythona)
3. [Config — opis pól](#3-config--opis-pól)
4. [Dane wejściowe](#4-dane-wejściowe)
5. [Wyniki — opis plików](#5-wyniki--opis-plików)
6. [Czytanie wyników w Pythonie](#6-czytanie-wyników-w-pythonie)
7. [Dobrze wiedzieć](#7-dobrze-wiedzieć)

---

## 1. Uruchomienie

Wymagany Rust (`rustup`), wszystko inne ściąga cargo.

```bash
cargo build --release --manifest-path source/Cargo.toml
cargo run --release --manifest-path source/Cargo.toml -- configs/eksperyment.json
cargo run --release --manifest-path source/Cargo.toml -- configs/a.json configs/b.json
cargo run --release --manifest-path source/Cargo.toml -- --overwrite configs/eksperyment.json
```

Co się dzieje po uruchomieniu:

1. Wczytywane są wszystkie configi i sprawdzane, czy są poprawne (literówka w nazwie pola albo
   wartości kończy się błędem z listą dozwolonych wartości — nic się wtedy nie liczy).
2. Wczytywane są dane (każdy plik tylko raz, nawet jeśli używa go wiele configów).
3. Każda para *(config, seed)* to osobne zadanie; zadania liczą się równolegle na wszystkich rdzeniach.
4. Po każdym skończonym zadaniu wypisywana jest jedna linia:

```
2 config(s), 3 run(s), 8 thread(s)
[three_gauss_tanh seed=1] test acc 0.9560 F1 0.9553 (0.1s)
[three_gauss_tanh seed=0] test acc 0.9540 F1 0.9534 (0.1s)
[cube_relu seed=0] test MSE 38220.0847 MAE 92.3531 (0.0s)
```

Kod wyjścia: `0` — wszystko OK, `1` — część runów się nie udała (np. sieć się rozjechała przy za
dużym `lr`; reszta i tak się policzyła), `2` — błąd w configu lub danych (nic się nie policzyło).

Jeśli katalog wyników danego eksperymentu już istnieje, program odmawia startu — trzeba dodać
`--overwrite` albo zmienić `name`. Liczbę wątków można ograniczyć: `RAYON_NUM_THREADS=4 cargo run ...`.

---

## 2. Uruchomienie z Pythona

Najwygodniej budować configi jako słowniki, zapisać je do jednego pliku (lista) i odpalić binarkę raz.

```python
import json, subprocess
from pathlib import Path

def run(configs, overwrite=False, path="configs/generated.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(configs, indent=2))
    cmd = ["cargo", "run", "--release", "--quiet", "--manifest-path", "source/Cargo.toml", "--"]
    if overwrite:
        cmd.append("--overwrite")
    subprocess.run(cmd + [path], check=False)   # kod 1 = część runów nieudana, nie przerywamy

base = {
    "seeds": [0, 1, 2, 3, 4],
    "data": {
        "task": "regression",
        "train": "data/projekt1/regression/data.cube.train.1000.csv",
        "test":  "data/projekt1/regression/data.cube.test.1000.csv",
    },
    "net":   {"hidden": [16], "activation": "tanh", "head": "identity_mse"},
    "train": {"epochs": 500, "batch": 32, "lr": 0.05, "log_every": 10},
}

# przykład: liczba warstw L × szerokość B
configs = []
for L in [1, 2, 3]:
    for B in [4, 16, 64]:
        cfg = json.loads(json.dumps(base))          # głęboka kopia
        cfg["name"] = f"cube_L{L}_B{B}"
        cfg["net"]["hidden"] = [B] * L
        configs.append(cfg)

run(configs, overwrite=True)
```

---

## 3. Config — opis pól

Przykład (klasyfikacja 2D, sieć 2 → 8 → 8 → 3):

```json
{
  "name": "three_gauss_tanh",
  "seeds": [0, 1, 2, 3, 4],
  "data": {
    "task": "classification",
    "train": "data/projekt1/classification/data.three_gauss.train.1000.csv",
    "test":  "data/projekt1/classification/data.three_gauss.test.1000.csv",
    "validation": 0.1
  },
  "net":    { "hidden": [8, 8], "activation": "tanh", "head": "softmax_cross_entropy" },
  "train":  { "epochs": 100, "batch": 32, "lr": 0.1, "log_every": 10 },
  "output": { "weights_every": 50 }
}
```

Pola oznaczone ✱ są wymagane, pozostałe mają wartość domyślną.

### Poziom główny

| pole | znaczenie | domyślnie |
|---|---|---|
| `name` ✱ | nazwa eksperymentu = nazwa katalogu z wynikami (bez `/`) | — |
| `seeds` ✱ | lista seedów, np. `[0,1,2,3,4]`; seed ustala losowe wagi początkowe i kolejność próbek w epokach. Każdy seed = osobny trening | — |
| `out_dir` | katalog, w którym powstanie `<out_dir>/<name>/` | `"out"` |
| `dtype` | precyzja obliczeń: `"f64"` lub `"f32"` (f32 ~2× szybciej, wystarcza do MNIST) | `"f64"` |

### `data` — skąd brać dane i jak je przygotować

| pole | znaczenie | domyślnie |
|---|---|---|
| `task` ✱ | `"regression"` lub `"classification"` | — |
| `train` ✱, `test` | ścieżka do pliku **albo** `{"x": "plik_x", "y": "plik_y"}`, gdy wejścia i etykiety są w osobnych plikach (patrz [4](#4-dane-wejściowe)); `test` można pominąć, jeżeli `validation > 0` | — |
| `normalize` | jak skalować wejścia x:<br>`"per_column"` — każda kolumna osobno do średniej 0 i odch. std. 1<br>`"global"` — jedna średnia i odch. std. dla wszystkich kolumn<br>`"min_max"` — wszystkie wartości do przedziału [0, 1]<br>`"none"` — bez zmian | `"per_column"` |
| `validation` | jaki ułamek zbioru treningowego odciąć jako zbiór walidacyjny, np. `0.1` = 10%. `0` = brak walidacji | `0` |
| `split_seed` | seed losowania podziału train/val (ten sam dla wszystkich `seeds`, żeby porównywać na tych samych danych) | `0` |
| `format` | `"csv"` lub `"idx"`; normalnie niepotrzebne — wykrywane z nazwy pliku | auto |
| `label_column` | gdy x i y są w jednym pliku: w której kolumnie jest cel — `"last"` lub `"first"` | `"last"` |
| `first_label` | numer pierwszej klasy w danych (klasy `1,2,3` → `1`; cyfry `0..9` → `0`) | `1` dla CSV, `0` dla IDX |

### `net` — architektura

| pole | znaczenie | domyślnie |
|---|---|---|
| `hidden` ✱ | rozmiary warstw ukrytych; `[8, 8]` = dwie warstwy po 8 neuronów, `[]` = brak warstw ukrytych. Rozmiar wejścia i wyjścia wynika z danych | — |
| `activation` ✱ | funkcja aktywacji wszystkich warstw ukrytych: `"sigmoid"`, `"tanh"`, `"relu"`, `"identity"` | — |
| `head` ✱ | aktywacja warstwy wyjściowej + funkcja kosztu, patrz niżej | — |
| `bias` | czy neurony mają bias | `true` |

| `head` | wyjście | koszt | zadanie |
|---|---|---|---|
| `"identity_mse"` | liniowe | błąd średniokwadratowy (½·MSE) | regresja |
| `"identity_mae"` | liniowe | średni błąd bezwzględny | regresja |
| `"softmax_cross_entropy"` | softmax (prawdopodobieństwa klas) | entropia krzyżowa | klasyfikacja |
| `"softmax_mse"` | softmax | MSE do wektora one-hot | klasyfikacja |

### `train` — uczenie

| pole | znaczenie | domyślnie |
|---|---|---|
| `epochs` ✱ | liczba przejść przez cały zbiór treningowy | — |
| `batch` ✱ | rozmiar mini-batcha (liczba próbek na jeden krok SGD) | — |
| `lr` ✱ | współczynnik uczenia (learning rate) | — |
| `log_every` | co ile epok liczyć i zapisywać metryki do `history.csv` | `1` |

### `output` — dodatkowe pliki

| pole | znaczenie | domyślnie |
|---|---|---|
| `weights_every` | co ile epok zapisywać wszystkie wagi (do wykresów/animacji ewolucji wag). Musi być wielokrotnością `log_every`. `0` = nie zapisuj | `0` |
| `grid_steps` | rozdzielczość siatki do rysowania granicy decyzyjnej (klasyfikacja z 2 wejściami) | `200` |
| `log_deltas` | zapisuj sygnały propagacji wstecznej δ = ∂ℓ/∂z dla każdego neuronu, co `log_every` epok, na jednej stałej próbce uczącej | `false` |

---

## 4. Dane wejściowe

| źródło | jak podać w configu |
|---|---|
| CSV z projektu (`x,y` / `x,y,cls`, nagłówek, cel w ostatniej kolumnie, klasy od 1) | `"train": "data/.../data.simple.train.1000.csv"` |
| MNIST w formacie IDX (Kaggle *hojjatk/mnist-dataset*) — obrazy i etykiety w osobnych plikach | `"train": {"x": "mnist/train-images.idx3-ubyte", "y": "mnist/train-labels.idx1-ubyte"}`<br>`"normalize": "min_max"` |
| MNIST w CSV z Kaggle (`mnist_train.csv`, etykieta w **pierwszej** kolumnie, cyfry 0–9) | `"train": "mnist/mnist_train.csv"`<br>`"label_column": "first", "first_label": 0, "normalize": "min_max"` |
| dowolne CSV z x i y w osobnych plikach | `"train": {"x": "x_train.csv", "y": "y_train.csv"}` |

Pliki `.gz` trzeba najpierw rozpakować. Nagłówek w CSV jest opcjonalny.

---

## 5. Wyniki — opis plików

Dla configu `name = "three_gauss_tanh"` z dwoma seedami powstaje:

```
out/three_gauss_tanh/
├── config.json              config z uzupełnionymi wartościami domyślnymi
├── points_train.csv         punkty treningowe (tylko gdy wejście ma 1 lub 2 wymiary)
├── seed_0/
│   ├── metrics.json         wynik końcowy (po ostatniej epoce)
│   ├── history.csv          metryki w trakcie uczenia
│   ├── grid.csv             granica decyzyjna (klasyfikacja, 2 wejścia)
│   ├── pred.csv             predykcje na teście (regresja, 1 wejście)
│   └── weights_layer0.csv   wagi warstwy 0, 1, 2, … (gdy weights_every > 0)
│       weights_layer1.csv
│       weights_layer2.csv
└── seed_1/
    └── ...
```

### `metrics.json` — wynik końcowy jednego seeda

Klasyfikacja:
```json
{
  "seed": 0,
  "seconds": 0.07,
  "epochs": 100,
  "train": { "cost": 0.0923, "accuracy": 0.9589, "macro_f1": 0.9586, "confusion": [[299, 5, 2], [4, 273, 13], [1, 12, 291]] },
  "val":   { "cost": 0.0936, "accuracy": 0.9500, "macro_f1": 0.9483, "confusion": [[36, 1, 0], [0, 32, 4], [0, 0, 27]] },
  "test":  { "cost": 0.1134, "accuracy": 0.9540, "macro_f1": 0.9534, "confusion": [[333, 4, 7], [10, 277, 13], [5, 7, 344]] }
}
```

Regresja:
```json
{
  "seed": 0, "seconds": 0.02, "epochs": 100,
  "train": { "cost": 0.0049, "mse": 26.15,    "mae": 4.21 },
  "test":  { "cost": 7.177,  "mse": 38220.08, "mae": 92.35 }
}
```

| klucz | znaczenie |
|---|---|
| `seed`, `epochs` | seed tego treningu i liczba epok |
| `seconds` | czas treningu tego seeda |
| `train` / `val` / `test` | metryki na danym zbiorze (`val` tylko gdy `validation > 0`) |
| `cost` | wartość funkcji kosztu z `head`, liczona na **znormalizowanych** danych — do porównywania przebiegu uczenia, nie do raportowania |
| `mse`, `mae` | błąd średniokwadratowy / bezwzględny w **oryginalnych jednostkach** y — to raportujemy |
| `accuracy` | odsetek poprawnie sklasyfikowanych próbek |
| `macro_f1` | F1 policzone osobno dla każdej klasy i uśrednione |
| `confusion` | macierz pomyłek: `confusion[prawdziwa][przewidziana]`, klasy numerowane od 0 |

Jeśli trening się rozjechał (koszt = NaN/∞, zwykle za duże `lr`), plik wygląda tak, a `history.csv` zawiera przebieg do tego momentu:
```json
{ "seed": 3, "seconds": 0.4, "error": "diverged at epoch 37 (train cost NaN)" }
```

### `history.csv` — przebieg uczenia

Wiersz dla epoki 0 (przed uczeniem), co `log_every` epok i dla ostatniej epoki.
Kolumny `<zbiór>_<metryka>` dla zbiorów `train` i `val` (jeśli jest).
Test jest oceniany wyłącznie po zakończeniu uczenia; jego wynik znajduje się w `metrics.json`.
Configi pilotażowe mogą całkowicie pomijać `data.test`, aby nie wczytywać testu podczas strojenia.

Klasyfikacja:
```
epoch,train_cost,train_accuracy,train_macro_f1,val_cost,val_accuracy,val_macro_f1
0,1.3297,0.1522,0.2053,1.3813,0.1600,0.2430
10,0.1050,0.9600,0.9597,0.0910,0.9700,0.9692
...
```

Regresja (`mse`, `mae` w oryginalnych jednostkach):
```
epoch,train_cost,train_mse,train_mae
0,5.1794,27581.48,124.66
50,0.0100,53.23,6.17
100,0.0049,26.15,4.21
```

### `weights_layer<l>.csv` — wagi w czasie

Jeden plik na warstwę (`0` = między wejściem a pierwszą warstwą ukrytą, ostatni = warstwa wyjściowa),
jeden wiersz na każdy zapis (co `weights_every` epok, plus epoka 0 i ostatnia).

```
epoch,w0_0,w0_1,w0_2,w1_0,w1_1,w1_2,...,w8_0,w8_1,w8_2
0,0.0463,-0.3163,0.6727,0.4912,0.0554,0.1045,...,0,0,0
50,...
```

- `w{i}_{j}` — waga połączenia **z wejścia `i`** tej warstwy **do neuronu `j`** tej warstwy.
- Ostatni indeks `i` (równy liczbie wejść warstwy) to **bias** neuronu `j` (np. wyżej: warstwa 8 → 3, więc `w8_j` to biasy; startują od 0).
- Macierz ma więc wymiar `(wejścia + 1) × neurony` (bez `+1`, gdy `bias: false`).

### `deltas_layer<l>.csv` — propagowany sygnał błędu

Gdy `output.log_deltas: true`, powstaje jeden plik na warstwę:

```text
epoch,delta_0,delta_1,...
```

`delta_j` to pochodna kosztu **jednej próbki** względem wejścia neuronu `j` przed aktywacją,
czyli δ = ∂ℓ/∂z. To nie wartość funkcji kosztu ani gradient wagi.
Rust wykorzystuje zwykłe `forward` i `backward` na tym samym pierwszym przykładzie z przygotowanej części uczącej.
Dla danych 1D/2D jest to pierwszy wiersz danych w `points_train.csv`.
Obliczenie diagnostyczne nie aktualizuje wag i nie losuje nowych wartości.
Epoka 0 pokazuje stan początkowy, kolejne wiersze — stan po zapisanych epokach.
Warstwa 0 jest pierwszą warstwą za wejściem; ostatnia to wyjście. Propagacja przebiega w odwrotnej kolejności.

Aby zapisywać sygnały w każdej epoce, ustaw `train.log_every: 1` i `output.log_deltas: true`.
W regresji obliczenia wykorzystują znormalizowany cel. Diagnostyczna partia ma jeden przykład,
więc δ nie zawiera dzielnika 1/B z uśredniania kosztu treningowej partii o B przykładach.

### Pozostałe pliki

| plik | kolumny | znaczenie |
|---|---|---|
| `points_train.csv` | `x1,[x2,]label` | surowe (nieznormalizowane) punkty treningowe; `label` = klasa (od 1) albo wartość y w regresji |
| `pred.csv` | `x,y_true,y_pred` | regresja z 1 wejściem: prawdziwa i przewidziana wartość dla każdego punktu zbioru testowego |
| `grid.csv` | `x1,x2,class` | klasyfikacja z 2 wejściami: siatka `grid_steps × grid_steps` punktów obejmująca dane (+10% marginesu) i klasa (od 1) przewidziana przez sieć — do narysowania obszarów decyzyjnych |
| `config.json` | — | dokładny config, z którym liczono (z wartościami domyślnymi) — do odczytania parametrów przy analizie |

---

## 6. Czytanie wyników w Pythonie

### Tabela: wszystkie eksperymenty × seedy

```python
import json
from pathlib import Path
import pandas as pd

rows = []
for cfg_path in Path("out").glob("*/config.json"):
    cfg = json.loads(cfg_path.read_text())
    for m_path in cfg_path.parent.glob("seed_*/metrics.json"):
        m = json.loads(m_path.read_text())
        if "error" in m or "test" not in m:
            continue
        rows.append({
            "name": cfg["name"],
            "L": len(cfg["net"]["hidden"]),
            "B": cfg["net"]["hidden"][0] if cfg["net"]["hidden"] else 0,
            "activation": cfg["net"]["activation"],
            "lr": cfg["train"]["lr"],
            "seed": m["seed"],
            **{f"test_{k}": v for k, v in m["test"].items() if k != "confusion"},
        })

df = pd.DataFrame(rows)
summary = df.groupby(["name", "L", "B"])["test_mse"].agg(["mean", "std"])   # albo test_accuracy
print(summary)
```

### Krzywa uczenia jednego runu

```python
import matplotlib.pyplot as plt

h = pd.read_csv("out/three_gauss_tanh/seed_0/history.csv")
plt.plot(h.epoch, h.train_accuracy, label="train")
if "val_accuracy" in h:
    plt.plot(h.epoch, h.val_accuracy, label="val")
plt.xlabel("epoka"); plt.ylabel("accuracy"); plt.legend()
```

### Wagi jako tablica `[zapis, wejście, neuron]`

```python
import numpy as np

def load_weights(path):
    header = open(path).readline().strip().split(",")
    rows, cols = (int(v) + 1 for v in header[-1][1:].split("_"))   # z nazwy ostatniej kolumny
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    epochs = data[:, 0].astype(int)
    W = data[:, 1:].reshape(-1, rows, cols)
    return epochs, W

epochs, W = load_weights("out/three_gauss_tanh/seed_0/weights_layer0.csv")
print(W.shape)          # (3, 3, 8): 3 zapisy, 2 wejścia + bias, 8 neuronów
weights, bias = W[:, :-1, :], W[:, -1, :]
```

### Granica decyzyjna

```python
g = pd.read_csv("out/three_gauss_tanh/seed_0/grid.csv")
p = pd.read_csv("out/three_gauss_tanh/points_train.csv")
n = int(len(g) ** 0.5)
plt.pcolormesh(g.x1.values.reshape(n, n), g.x2.values.reshape(n, n),
               g["class"].values.reshape(n, n), alpha=0.3, shading="auto", cmap="tab10", vmin=1, vmax=10)
plt.scatter(p.x1, p.x2, c=p.label, s=5, cmap="tab10", vmin=1, vmax=10)
```

---

## 7. Dobrze wiedzieć

- **Determinizm**: ten sam config + ten sam seed = identyczny wynik, niezależnie od liczby wątków.
- **Normalizacja** liczona jest wyłącznie na części treningowej i potem stosowana do val/test.
  W regresji y też jest normalizowane do uczenia, ale `mse`/`mae` w wynikach są przeliczone z powrotem na oryginalne jednostki.
- **`cost` vs metryki**: `cost` jest w jednostkach znormalizowanych i zależy od wybranego `head` — do porównań między różnymi `head` używaj `mse`/`mae`/`accuracy`/`macro_f1`.
- **Wybór liczby epok**: używaj `val_*` w `history.csv`; test jest oceniany dopiero po treningu. Wybieranie po teście zawyża wynik.
- **Rozmiar plików wag**: dla MNIST (784 × 64) jeden zapis to ~1 MB na seed — ustaw rzadsze `weights_every` albo włącz je tylko w jednym eksperymencie.
- **Ekstrapolacja**: w obu zbiorach regresyjnych test (x od −5 do 7) wychodzi poza zakres treningu (około −2 do 5), więc `test_mse` może być dużo większe niż `train_mse`.
