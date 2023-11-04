# UMA

## Skład zespołu
* Ignacy Dąbkowski
* Ireneusz Okniński

## Treść zadania
Zastosowanie uczenia ze wzmocnieniem do ustawienia prawdopodobieństwa krzyżowania
i sposobu krzyżowania w algorytmie ewolucyjnym. Niech stanem będzie procent
sukcesów oraz średnia odległość pomiędzy osobnikami w aktualnej populacji, a 
akcją wybór wskazanych parametrów algorytmu. O sukcesie mówimy wtedy, gdy średnia
wartość funkcji oceny populacji potomnej jest lepsza niż populacji aktualnej. 
Liczbę sukcesów należy zsumować na przestrzeni X iteracji, gdzie X to np. 20, 
po czym podzielić przez X. Zarówno stany jak i akcje można zdyskretyzować. 
Funkcje do optymalizacji należy pobrać z benchmarku CEC2017, którego kod da się 
znaleźć w Pythonie, R i C.

## Rzeczy do zastanowienia się:
* Jakie wymiary funkcji z CEC2017 użyć?
* Q-learning?
* Jak oceniamy funkcje?
  * ilość wywołań funkcji celu = const?
  * pozostałe parametry?

## Rodzaje krzyżowania:
   * Krzyżowanie jednopunktowe. Wybieramy miejsce rozcięcia. 
     Potomek 1 ma kod rodzica 1 od początku do wybranego miejsca, a dalej ma 
     kod rodzica 2. Potomek 2 dostaje resztę materiału genetycznego.
   * Krzyżowania wymieniające – krzyżowanie równomierne. Dla każdego genu losujemy
     liczbę U(0, 1). Jeśli wyszło mniej niż pe (parametr operacji krzyżowania) to 
     bierzemy gen od rodzica 1. Gen pochodzi od rodzica 2 w przeciwnym przypadku.
   * Krzyżowanie uśredniające w wariancie podstawowym to ważone
     uśrednianie z losowymi współ-czynnikami (wagami)
   * Krzyżowanie uśredniające w wariancie rozszerzonym wprowadza niezależne wagi dla każdej
     współrzędnej (genu)
