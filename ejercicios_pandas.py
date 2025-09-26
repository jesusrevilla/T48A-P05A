import pandas as pd
import doctest

def crear_serie(lista):
    crear_serie([1, 2, 3]).tolist()
    [1, 2, 3]
    return pd.Series(lista)

def crear_dataframe(diccionario):
    df = crear_dataframe({'a': [1, 2], 'b': [3, 4]})
    df.shape
    (2, 2)
    list(df.columns)
    ['a', 'b']
