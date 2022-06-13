# OpenCV_Lego_Shape_Recognition
# Metoda działania:
  #### Przetwarzanie obrazów:
  Obrazy po wczytaniu są przetwarzane poprzez użycie dwóch masek HLS oraz jednej HSV, w celu uzyskania obraz uzawierającego tylko obiekty bez tła i zakłóceń. Następnie dla każdego konturu spełniającego określone warunki dotyczące pola powierzchni zostaje stworzony element który zawiera prostokąt z konturem w środku. 
  #### Wykrywanie kształtów:
  Program posiada 5 zbiorów po jednym elemencie z każdego kształtu. Wczytywanie są jednocześnie dzięki wielowątkowemu działaniu programu. Następnie dla każdego elementu opisanego w poprzednim punkcie zostaje uruchomiony osobny wątek dla każdego z 5 zbiorów, który sprawdza wartość funkcji structural similarity. Następnie zapisany zostaje najwyzszy wynik, dodatkowym czynnikiem jest współczynnik propocji danego kształtu, pomocny przy rozrócznieniu kwadratów od prostokątów.
   #### Wykrywanie kolorów:
   Kolejnym etapem jest wykrywanie kolorów danych elementów na podstawie 5 dominujących kolorów, oraz sklasyfikowaniu ich wartości do odpowiednich kolorów.
## Źródła:
 https://github.com/codegiovanni/Dominant_colors/blob/main/dominant_colors.py
 https://scikit-image.org/docs/stable/auto_examples/transform/plot_ssim.html
 https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
