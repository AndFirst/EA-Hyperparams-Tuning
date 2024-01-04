def quality_reward(kwargs):
    """
    Oblicza nagrodę na podstawie poprawy jakości najlepszego punktu.

    Parametry:
    - kwargs (dict): Słownik zawierający niezbędne wartości.
      - "current_best_quality" (float): Aktualna najlepsza wartość jakości.
      - "last_best_quality" (float): Najlepsza wartość jakości z poprzedniej iteracji.

    Zwraca:
    float: Nagroda obliczona jako różnica między ostatnią najlepszą jakością a aktualną jakością.
    """
    current_quality = kwargs["current_best_quality"]
    last_best_quality = kwargs["last_best_quality"]

    return last_best_quality - current_quality


def successes_reward(kwargs):
    """
    Oblicza nagrodę na podstawie zmiany procentu sukcesów.

    Parametry:
    - kwargs (dict): Słownik zawierający niezbędne wartości.
      - "current_successes_bin" (float): Aktualna wartość sukcesów.
      - "last_successes_bin" (float): Wartość sukcesów z poprzedniej iteracji.

    Zwraca:
    float: Nagroda obliczona jako różnica między ostatnimi sukcesami a aktualnymi sukcesami, pomnożona przez 10.
    """
    current_successes_bin = kwargs["current_successes_bin"]
    last_successes_bin = kwargs["last_successes_bin"]

    return (current_successes_bin - last_successes_bin) * 10


def distance_reward(kwargs):
    """
    Oblicza nagrodę na podstawie zmiany średniej odległości populacji.

    Parametry:
    - kwargs (dict): Słownik zawierający niezbędne wartości.
      - "current_distance_bin" (float): Aktualna wartość odległości.
      - "last_distance_bin" (float): Wartość odległości z poprzedniej iteracji.

    Zwraca:
    float: Nagroda obliczona jako różnica między ostatnią odległością a aktualną odległością, pomnożona przez 10.
    """
    current_distance_bin = kwargs["current_distance_bin"]
    last_distance_bin = kwargs["last_distance_bin"]

    return (current_distance_bin - last_distance_bin) * 10
