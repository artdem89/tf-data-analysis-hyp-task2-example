import pandas as pd
import numpy as np
import scipy.stats as sps

chat_id = 157443210 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array, y: np.array) -> bool:
    # Измените код этой функции
    # Это будет вашим решением
    # Не меняйте название функции и её аргументы
    p_value = sps.cramervonmises_2samp(x, y).pvalue    
    alpha = 0.08
    return p_value < alpha # Ваш ответ, True или False


