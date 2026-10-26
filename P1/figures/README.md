# Wyniki

Wartość końcowa = średnia z ostatnich K epok; linia = mediana po ziarnach, pas = kwartyle; ✗n = rozbieżne ziarna (lista w failures.csv, razem 21). Tabele: summary.csv, runs.csv.

### `fig:simple:arch_heatmap_test_accuracy`

simple: accuracy (test) (mediana) w zależności od liczby warstw ukrytych $L$ i neuronów w warstwie $B$.

![arch_heatmap_test_accuracy](simple/arch_heatmap_test_accuracy.png)

### `fig:simple:arch_lines_test_accuracy`

simple: accuracy (test) w zależności od $B$ dla każdej liczby warstw $L$ (mediana i kwartyle po ziarnach).

![arch_lines_test_accuracy](simple/arch_lines_test_accuracy.png)

### `fig:simple:activation_lr_test_accuracy`

simple: accuracy (test) dla różnych funkcji aktywacji w zależności od współczynnika uczenia (mediana i kwartyle; liczba przy krzyżyku nad wykresem to liczba rozbieżnych ziaren).

![activation_lr_test_accuracy](simple/activation_lr_test_accuracy.png)

### `fig:simple:activation_lr_test_macro_f1`

simple: macro-F1 (test) dla różnych funkcji aktywacji w zależności od współczynnika uczenia (mediana i kwartyle; liczba przy krzyżyku nad wykresem to liczba rozbieżnych ziaren).

![activation_lr_test_macro_f1](simple/activation_lr_test_macro_f1.png)

### `fig:simple:activation_curve_test_accuracy`

simple: przebieg accuracy (test) w trakcie uczenia dla różnych funkcji aktywacji (współczynnik uczenia 0.1).

![activation_curve_test_accuracy](simple/activation_curve_test_accuracy.png)

### `fig:simple:activation_curve_test_cost`

simple: przebieg koszt (test) w trakcie uczenia dla różnych funkcji aktywacji (współczynnik uczenia 0.1).

![activation_curve_test_cost](simple/activation_curve_test_cost.png)

### `fig:simple:predictions_relu`

simple: obszary decyzyjne sieci z aktywacją relu (ziarno 0). Punkty testowe w kolorze prawdziwej klasy, × oznacza pomyłkę (accuracy 0.995).

![predictions_relu](simple/predictions_relu.png)

### `fig:simple:predictions_sigmoid`

simple: obszary decyzyjne sieci z aktywacją sigmoid (ziarno 0). Punkty testowe w kolorze prawdziwej klasy, × oznacza pomyłkę (accuracy 0.997).

![predictions_sigmoid](simple/predictions_sigmoid.png)

### `fig:simple:predictions_tanh`

simple: obszary decyzyjne sieci z aktywacją tanh (ziarno 0). Punkty testowe w kolorze prawdziwej klasy, × oznacza pomyłkę (accuracy 0.995).

![predictions_tanh](simple/predictions_tanh.png)

### `fig:simple:confusion_relu_32x32`

simple: macierz pomyłek na zbiorze testowym, aktywacja relu, architektura 32x32, suma po 5 ziarnach (accuracy 0.991).

![confusion_relu_32x32](simple/confusion_relu_32x32.png)

### `fig:simple:confusion_sigmoid_32x32`

simple: macierz pomyłek na zbiorze testowym, aktywacja sigmoid, architektura 32x32, suma po 5 ziarnach (accuracy 0.996).

![confusion_sigmoid_32x32](simple/confusion_sigmoid_32x32.png)

### `fig:simple:confusion_tanh_32x32`

simple: macierz pomyłek na zbiorze testowym, aktywacja tanh, architektura 32x32, suma po 5 ziarnach (accuracy 0.990).

![confusion_tanh_32x32](simple/confusion_tanh_32x32.png)

### `fig:simple:loss_lr_test_accuracy`

simple: accuracy (test) dla różnych funkcji kosztu w zależności od współczynnika uczenia (mediana i kwartyle; liczba przy krzyżyku nad wykresem to liczba rozbieżnych ziaren).

![loss_lr_test_accuracy](simple/loss_lr_test_accuracy.png)

### `fig:simple:loss_lr_test_macro_f1`

simple: macro-F1 (test) dla różnych funkcji kosztu w zależności od współczynnika uczenia (mediana i kwartyle; liczba przy krzyżyku nad wykresem to liczba rozbieżnych ziaren).

![loss_lr_test_macro_f1](simple/loss_lr_test_macro_f1.png)

### `fig:simple:loss_curve_test_accuracy`

simple: przebieg accuracy (test) w trakcie uczenia dla różnych funkcji kosztu (współczynnik uczenia 0.1).

![loss_curve_test_accuracy](simple/loss_curve_test_accuracy.png)

### `fig:simple:bias`

simple: accuracy (test) sieci z biasami i bez biasów.

![bias](simple/bias.png)

### `fig:three_gauss:arch_heatmap_test_accuracy`

three_gauss: accuracy (test) (mediana) w zależności od liczby warstw ukrytych $L$ i neuronów w warstwie $B$.

![arch_heatmap_test_accuracy](three_gauss/arch_heatmap_test_accuracy.png)

### `fig:three_gauss:arch_lines_test_accuracy`

three_gauss: accuracy (test) w zależności od $B$ dla każdej liczby warstw $L$ (mediana i kwartyle po ziarnach).

![arch_lines_test_accuracy](three_gauss/arch_lines_test_accuracy.png)

### `fig:three_gauss:activation_lr_test_accuracy`

three_gauss: accuracy (test) dla różnych funkcji aktywacji w zależności od współczynnika uczenia (mediana i kwartyle; liczba przy krzyżyku nad wykresem to liczba rozbieżnych ziaren).

![activation_lr_test_accuracy](three_gauss/activation_lr_test_accuracy.png)

### `fig:three_gauss:activation_lr_test_macro_f1`

three_gauss: macro-F1 (test) dla różnych funkcji aktywacji w zależności od współczynnika uczenia (mediana i kwartyle; liczba przy krzyżyku nad wykresem to liczba rozbieżnych ziaren).

![activation_lr_test_macro_f1](three_gauss/activation_lr_test_macro_f1.png)

### `fig:three_gauss:activation_curve_test_accuracy`

three_gauss: przebieg accuracy (test) w trakcie uczenia dla różnych funkcji aktywacji (współczynnik uczenia 0.1).

![activation_curve_test_accuracy](three_gauss/activation_curve_test_accuracy.png)

### `fig:three_gauss:activation_curve_test_cost`

three_gauss: przebieg koszt (test) w trakcie uczenia dla różnych funkcji aktywacji (współczynnik uczenia 0.1).

![activation_curve_test_cost](three_gauss/activation_curve_test_cost.png)

### `fig:three_gauss:predictions_relu`

three_gauss: obszary decyzyjne sieci z aktywacją relu (ziarno 0). Punkty testowe w kolorze prawdziwej klasy, × oznacza pomyłkę (accuracy 0.933).

![predictions_relu](three_gauss/predictions_relu.png)

### `fig:three_gauss:predictions_sigmoid`

three_gauss: obszary decyzyjne sieci z aktywacją sigmoid (ziarno 0). Punkty testowe w kolorze prawdziwej klasy, × oznacza pomyłkę (accuracy 0.932).

![predictions_sigmoid](three_gauss/predictions_sigmoid.png)

### `fig:three_gauss:predictions_tanh`

three_gauss: obszary decyzyjne sieci z aktywacją tanh (ziarno 0). Punkty testowe w kolorze prawdziwej klasy, × oznacza pomyłkę (accuracy 0.933).

![predictions_tanh](three_gauss/predictions_tanh.png)

### `fig:three_gauss:confusion_relu_32x32`

three_gauss: macierz pomyłek na zbiorze testowym, aktywacja relu, architektura 32x32, suma po 5 ziarnach (accuracy 0.934).

![confusion_relu_32x32](three_gauss/confusion_relu_32x32.png)

### `fig:three_gauss:confusion_sigmoid_32x32`

three_gauss: macierz pomyłek na zbiorze testowym, aktywacja sigmoid, architektura 32x32, suma po 5 ziarnach (accuracy 0.933).

![confusion_sigmoid_32x32](three_gauss/confusion_sigmoid_32x32.png)

### `fig:three_gauss:confusion_tanh_32x32`

three_gauss: macierz pomyłek na zbiorze testowym, aktywacja tanh, architektura 32x32, suma po 5 ziarnach (accuracy 0.935).

![confusion_tanh_32x32](three_gauss/confusion_tanh_32x32.png)

### `fig:three_gauss:loss_lr_test_accuracy`

three_gauss: accuracy (test) dla różnych funkcji kosztu w zależności od współczynnika uczenia (mediana i kwartyle; liczba przy krzyżyku nad wykresem to liczba rozbieżnych ziaren).

![loss_lr_test_accuracy](three_gauss/loss_lr_test_accuracy.png)

### `fig:three_gauss:loss_lr_test_macro_f1`

three_gauss: macro-F1 (test) dla różnych funkcji kosztu w zależności od współczynnika uczenia (mediana i kwartyle; liczba przy krzyżyku nad wykresem to liczba rozbieżnych ziaren).

![loss_lr_test_macro_f1](three_gauss/loss_lr_test_macro_f1.png)

### `fig:three_gauss:loss_curve_test_accuracy`

three_gauss: przebieg accuracy (test) w trakcie uczenia dla różnych funkcji kosztu (współczynnik uczenia 0.1).

![loss_curve_test_accuracy](three_gauss/loss_curve_test_accuracy.png)

### `fig:three_gauss:bias`

three_gauss: accuracy (test) sieci z biasami i bez biasów.

![bias](three_gauss/bias.png)

### `fig:cube:arch_heatmap_test_in_mse`

cube: MSE (test w zakresie uczenia) (mediana) w zależności od liczby warstw ukrytych $L$ i neuronów w warstwie $B$.

![arch_heatmap_test_in_mse](cube/arch_heatmap_test_in_mse.png)

### `fig:cube:arch_lines_test_in_mse`

cube: MSE (test w zakresie uczenia) w zależności od $B$ dla każdej liczby warstw $L$ (mediana i kwartyle po ziarnach).

![arch_lines_test_in_mse](cube/arch_lines_test_in_mse.png)

### `fig:cube:arch_heatmap_test_out_mse`

cube: MSE (test poza zakresem, ekstrapolacja) (mediana) w zależności od liczby warstw ukrytych $L$ i neuronów w warstwie $B$.

![arch_heatmap_test_out_mse](cube/arch_heatmap_test_out_mse.png)

### `fig:cube:arch_lines_test_out_mse`

cube: MSE (test poza zakresem, ekstrapolacja) w zależności od $B$ dla każdej liczby warstw $L$ (mediana i kwartyle po ziarnach).

![arch_lines_test_out_mse](cube/arch_lines_test_out_mse.png)

### `fig:cube:activation_lr_test_in_mse`

cube: MSE (test w zakresie uczenia) dla różnych funkcji aktywacji w zależności od współczynnika uczenia (mediana i kwartyle; liczba przy krzyżyku nad wykresem to liczba rozbieżnych ziaren).

![activation_lr_test_in_mse](cube/activation_lr_test_in_mse.png)

### `fig:cube:activation_lr_test_out_mse`

cube: MSE (test poza zakresem, ekstrapolacja) dla różnych funkcji aktywacji w zależności od współczynnika uczenia (mediana i kwartyle; liczba przy krzyżyku nad wykresem to liczba rozbieżnych ziaren).

![activation_lr_test_out_mse](cube/activation_lr_test_out_mse.png)

### `fig:cube:activation_curve_test_in_mse`

cube: przebieg MSE (test w zakresie uczenia) w trakcie uczenia dla różnych funkcji aktywacji (współczynnik uczenia 0.1).

![activation_curve_test_in_mse](cube/activation_curve_test_in_mse.png)

### `fig:cube:predictions_relu`

cube: funkcja nauczona przez sieć z aktywacją relu (ziarno 0) na całym zakresie testu; zielone tło to zakres danych uczących.

![predictions_relu](cube/predictions_relu.png)

### `fig:cube:predictions_sigmoid`

cube: funkcja nauczona przez sieć z aktywacją sigmoid (ziarno 0) na całym zakresie testu; zielone tło to zakres danych uczących.

![predictions_sigmoid](cube/predictions_sigmoid.png)

### `fig:cube:predictions_tanh`

cube: funkcja nauczona przez sieć z aktywacją tanh (ziarno 0) na całym zakresie testu; zielone tło to zakres danych uczących.

![predictions_tanh](cube/predictions_tanh.png)

### `fig:cube:loss_lr_test_in_mse`

cube: MSE (test w zakresie uczenia) dla różnych funkcji kosztu w zależności od współczynnika uczenia (mediana i kwartyle; liczba przy krzyżyku nad wykresem to liczba rozbieżnych ziaren).

![loss_lr_test_in_mse](cube/loss_lr_test_in_mse.png)

### `fig:cube:loss_lr_test_in_mae`

cube: MAE (test w zakresie uczenia) dla różnych funkcji kosztu w zależności od współczynnika uczenia (mediana i kwartyle; liczba przy krzyżyku nad wykresem to liczba rozbieżnych ziaren).

![loss_lr_test_in_mae](cube/loss_lr_test_in_mae.png)

### `fig:cube:loss_curve_test_in_mse`

cube: przebieg MSE (test w zakresie uczenia) w trakcie uczenia dla różnych funkcji kosztu (współczynnik uczenia 0.1).

![loss_curve_test_in_mse](cube/loss_curve_test_in_mse.png)

### `fig:cube:bias`

cube: MSE (test w zakresie uczenia) sieci z biasami i bez biasów.

![bias](cube/bias.png)

### `fig:activation:arch_heatmap_test_in_mse`

activation: MSE (test w zakresie uczenia) (mediana) w zależności od liczby warstw ukrytych $L$ i neuronów w warstwie $B$.

![arch_heatmap_test_in_mse](activation/arch_heatmap_test_in_mse.png)

### `fig:activation:arch_lines_test_in_mse`

activation: MSE (test w zakresie uczenia) w zależności od $B$ dla każdej liczby warstw $L$ (mediana i kwartyle po ziarnach).

![arch_lines_test_in_mse](activation/arch_lines_test_in_mse.png)

### `fig:activation:arch_heatmap_test_out_mse`

activation: MSE (test poza zakresem, ekstrapolacja) (mediana) w zależności od liczby warstw ukrytych $L$ i neuronów w warstwie $B$.

![arch_heatmap_test_out_mse](activation/arch_heatmap_test_out_mse.png)

### `fig:activation:arch_lines_test_out_mse`

activation: MSE (test poza zakresem, ekstrapolacja) w zależności od $B$ dla każdej liczby warstw $L$ (mediana i kwartyle po ziarnach).

![arch_lines_test_out_mse](activation/arch_lines_test_out_mse.png)

### `fig:activation:activation_lr_test_in_mse`

activation: MSE (test w zakresie uczenia) dla różnych funkcji aktywacji w zależności od współczynnika uczenia (mediana i kwartyle; liczba przy krzyżyku nad wykresem to liczba rozbieżnych ziaren).

![activation_lr_test_in_mse](activation/activation_lr_test_in_mse.png)

### `fig:activation:activation_lr_test_out_mse`

activation: MSE (test poza zakresem, ekstrapolacja) dla różnych funkcji aktywacji w zależności od współczynnika uczenia (mediana i kwartyle; liczba przy krzyżyku nad wykresem to liczba rozbieżnych ziaren).

![activation_lr_test_out_mse](activation/activation_lr_test_out_mse.png)

### `fig:activation:activation_curve_test_in_mse`

activation: przebieg MSE (test w zakresie uczenia) w trakcie uczenia dla różnych funkcji aktywacji (współczynnik uczenia 0.1).

![activation_curve_test_in_mse](activation/activation_curve_test_in_mse.png)

### `fig:activation:predictions_relu`

activation: funkcja nauczona przez sieć z aktywacją relu (ziarno 0) na całym zakresie testu; zielone tło to zakres danych uczących.

![predictions_relu](activation/predictions_relu.png)

### `fig:activation:predictions_sigmoid`

activation: funkcja nauczona przez sieć z aktywacją sigmoid (ziarno 0) na całym zakresie testu; zielone tło to zakres danych uczących.

![predictions_sigmoid](activation/predictions_sigmoid.png)

### `fig:activation:predictions_tanh`

activation: funkcja nauczona przez sieć z aktywacją tanh (ziarno 0) na całym zakresie testu; zielone tło to zakres danych uczących.

![predictions_tanh](activation/predictions_tanh.png)

### `fig:activation:loss_lr_test_in_mse`

activation: MSE (test w zakresie uczenia) dla różnych funkcji kosztu w zależności od współczynnika uczenia (mediana i kwartyle; liczba przy krzyżyku nad wykresem to liczba rozbieżnych ziaren).

![loss_lr_test_in_mse](activation/loss_lr_test_in_mse.png)

### `fig:activation:loss_lr_test_in_mae`

activation: MAE (test w zakresie uczenia) dla różnych funkcji kosztu w zależności od współczynnika uczenia (mediana i kwartyle; liczba przy krzyżyku nad wykresem to liczba rozbieżnych ziaren).

![loss_lr_test_in_mae](activation/loss_lr_test_in_mae.png)

### `fig:activation:loss_curve_test_in_mse`

activation: przebieg MSE (test w zakresie uczenia) w trakcie uczenia dla różnych funkcji kosztu (współczynnik uczenia 0.1).

![loss_curve_test_in_mse](activation/loss_curve_test_in_mse.png)

### `fig:activation:bias`

activation: MSE (test w zakresie uczenia) sieci z biasami i bez biasów.

![bias](activation/bias.png)

### `fig:mnist:arch_heatmap_test_accuracy`

MNIST: accuracy (test) (mediana) w zależności od liczby warstw ukrytych $L$ i neuronów w warstwie $B$.

![arch_heatmap_test_accuracy](mnist/arch_heatmap_test_accuracy.png)

### `fig:mnist:arch_lines_test_accuracy`

MNIST: accuracy (test) w zależności od $B$ dla każdej liczby warstw $L$ (mediana i kwartyle po ziarnach).

![arch_lines_test_accuracy](mnist/arch_lines_test_accuracy.png)

### `fig:mnist:time_per_epoch`

MNIST: czas jednej epoki (uczenie i ewaluacja, jeden wątek) w zależności od architektury.

![time_per_epoch](mnist/time_per_epoch.png)

### `fig:mnist:activation_lr_test_accuracy`

MNIST: accuracy (test) dla różnych funkcji aktywacji w zależności od współczynnika uczenia (mediana i kwartyle; liczba przy krzyżyku nad wykresem to liczba rozbieżnych ziaren).

![activation_lr_test_accuracy](mnist/activation_lr_test_accuracy.png)

### `fig:mnist:activation_lr_test_macro_f1`

MNIST: macro-F1 (test) dla różnych funkcji aktywacji w zależności od współczynnika uczenia (mediana i kwartyle; liczba przy krzyżyku nad wykresem to liczba rozbieżnych ziaren).

![activation_lr_test_macro_f1](mnist/activation_lr_test_macro_f1.png)

### `fig:mnist:activation_curve_test_accuracy`

MNIST: przebieg accuracy (test) w trakcie uczenia dla różnych funkcji aktywacji (współczynnik uczenia 0.1).

![activation_curve_test_accuracy](mnist/activation_curve_test_accuracy.png)

### `fig:mnist:activation_curve_test_cost`

MNIST: przebieg koszt (test) w trakcie uczenia dla różnych funkcji aktywacji (współczynnik uczenia 0.1).

![activation_curve_test_cost](mnist/activation_curve_test_cost.png)

### `fig:mnist:confusion_relu_128`

MNIST: macierz pomyłek na zbiorze testowym, aktywacja relu, architektura 128, suma po 5 ziarnach (accuracy 0.979).

![confusion_relu_128](mnist/confusion_relu_128.png)

### `fig:mnist:confusion_sigmoid_128`

MNIST: macierz pomyłek na zbiorze testowym, aktywacja sigmoid, architektura 128, suma po 5 ziarnach (accuracy 0.961).

![confusion_sigmoid_128](mnist/confusion_sigmoid_128.png)

### `fig:mnist:confusion_tanh_128`

MNIST: macierz pomyłek na zbiorze testowym, aktywacja tanh, architektura 128, suma po 5 ziarnach (accuracy 0.977).

![confusion_tanh_128](mnist/confusion_tanh_128.png)

### `fig:mnist:loss_lr_test_accuracy`

MNIST: accuracy (test) dla różnych funkcji kosztu w zależności od współczynnika uczenia (mediana i kwartyle; liczba przy krzyżyku nad wykresem to liczba rozbieżnych ziaren).

![loss_lr_test_accuracy](mnist/loss_lr_test_accuracy.png)

### `fig:mnist:loss_lr_test_macro_f1`

MNIST: macro-F1 (test) dla różnych funkcji kosztu w zależności od współczynnika uczenia (mediana i kwartyle; liczba przy krzyżyku nad wykresem to liczba rozbieżnych ziaren).

![loss_lr_test_macro_f1](mnist/loss_lr_test_macro_f1.png)

### `fig:mnist:loss_curve_test_accuracy`

MNIST: przebieg accuracy (test) w trakcie uczenia dla różnych funkcji kosztu (współczynnik uczenia 0.1).

![loss_curve_test_accuracy](mnist/loss_curve_test_accuracy.png)

### `fig:mnist:bias`

MNIST: accuracy (test) sieci z biasami i bez biasów.

![bias](mnist/bias.png)
