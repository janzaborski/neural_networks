# 1. Informacje ogólne

Każdy projekt polega na wykonaniu eksperymentu (eksperymentów) mających zbadać działanie zadanej sieci neuronowej i zaprezentowaniu przeprowadzonych eksperymentów i wniosków w formie raportu. Wykonanie samego programu, bez raportu, nie wystarcza do zaliczenia. Jeśli zostały przeprowadzone jakieś eksperymenty lub inne działania, które nie są opisane w raporcie, nie są one brane pod uwagę przy ocenie projektu. 

Każdy raport musi zawierać:

- opis rozwiązywanego problemu, zrozumiały dla osoby, która nie widziała treści zadania,
- opis zastosowanej sieci i jej implementacji,
- opis przeprowadzonych eksperymentów,
- wyniki (prezentowane w czytelny sposób) i wnioski,
- bibliografię z odwołaniami w tekście (bez pozycji "wiszących", czyli takich bez odnośników w tekście)

Struktura raportu: streszczenie, wstęp, część główna, podsumowanie z wnioskami, bibliografia.

## Powtarzalność eksperymentów

Planując eksperymenty i opisując wyniki należy pamiętać, że proces nauki sieci nie jest w 100% powtarzalny. Często rozpoczynamy uczenie od losowej inicjacji wag. W zależności od wyniku tego losowania końcowy efekt może się różnić. Oczywiście, dobrze działająca metoda uczenia powinna dawać zbliżone rezultaty, niezależnie od wag początkowych (tak długo, jak początkowe wagi są wybierane z rozsądnego rozkładu).

Należy zaadresować ten problem, na przykład przez przeprowadzenie uczenia dla wielu różnych układów startowych i sprawdzenie powtarzalności wyników.

# 2. Projekt 1: klasyfikacja i regresja z użyciem perceptronu (20 punktów)

## 2.1. Część pierwsza (15 punktów)

Program pozwalający rozwiązywać zadania klasyfikacji i regresji z użyciem perceptronu wielowarstwowego i wizualizujący jego działanie, w tym uczenie sieci z użyciem algorytmu propagacji wstecznej błędu (SGD lub jego warianty). Wymagana są wizualizacje:

- wyników uczenia sieci dla problemu regresji dla funkcji R → R i problemu klasyfikacji wektorów w R2;
- przebieg błędu po iteracjach;
- ewolucja wartości wag.

### 2.1.1. Zbiory danych

Na początku zajęć udostępniony będzie zestaw zbiorów danych, na których należy przeprowadzić eksperymenty badające skuteczność działania sieci.

### 2.1.2. Uwagi do implementacji

Implementację należy wykonać na poziomie podstawowym, tak aby kod pokazywał zrozumienie zasad działania i uczenia sieci. Nie można korzystać z bibliotek implementujących sieci neuronowe. Można korzystać z biblioteki numpy.

### 2.1.3. Elementy do zaimplementowania

- możliwość zainicjowania (powtarzalnego) procesu uczenia z zadanym ziarnem generatora liczb losowych
- łatwa konfiguracja liczby warstw w sieci i neuronów w warstwie, obecności biasów
- wizualizacja zbioru uczącego i efektów klasyfikacji oraz regresji
- wizualizacja błędu propagowanego w kolejnych iteracjach uczenia
- wizualizacja wartości wag w kolejnych iteracjach uczenia

### 2.1.4. Elementy do zbadania

- Wpływ funkcji aktywacji na skuteczność działania sieci – sprawdzić funkcję sigmoidalną i dwie inne wybrane (funkcje aktywacji wewnątrz sieci). Uwaga: funkcja aktywacji na wyjściu musi być dobrana odpowiednio do rodzaju problemu.
- Wpływ liczby warstw ukrytych w sieci oraz ich liczności. Zbadać różne liczby warstw od 0 do 4, kilka różnych architektur.
- Wpływ miary błędu na wyjściu sieci na skuteczność uczenia. Sprawdzić dwie miary błędu dla klasyfikacji i dwie dla regresji.

Wyniki i wnioski należy koniecznie opisać w raporcie z projektu.

## 2.2. Część druga (5 punktów)

Z użyciem implementacji wytworzonej w części pierwszej, modyfikując jedynie parametry sieci (liczbę warstw, neuronów, funkcję aktywacji, itp.), ale nie rozszerzając implementacji o nowe funkcjonalności, nauczyć sieć rozpoznawać cyfry ze zbioru MNIST. Można modyfikować kod programu w celu jego zoptymalizowania (np. tablice zamiast obiektów)

W procesie uczenia sieci korzystamy JEDYNIE ze zbioru oznaczonego jako uczący, a zbiór testowy wykorzystujemy wyłącznie do oceny jakości już nauczonej sieci.

Sieć zgłoszona do oceny wyników musi umożliwiać powtórzenie procesu uczenia. To oznacza, że trzeba dostarczyć wszystkie niezbędne parametry programu do ponownego uczenia. Jeśli uczenie korzysta z generatora liczb pseudolosowych, należy również dostarczyć ziarno generatora, aby można było powtórzyć proces uczenia dokładnie.

Uwaga. Uczenie na dużym zbiorze zajmuje odpowiednio dużo czasu. W ramach wstępnych eksperymentów można wybrać jakiś podzbiór danych uczących i uczyć tylko na nich, a dopiero po wstępnym rozpoznaniu robić pełne uczenie.
