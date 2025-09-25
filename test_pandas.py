import unittest
import numpy as np
import pandas as pd
from ejercicios_pandas import (crear_serie, crear_dataframe, seleccionar_columna,
                               filtrar_por_condicion, agregar_columna,
                               crear_indexacion_jerarquica, suma_columnas,
                               rellenar_nulos, agrupar_y_promediar, ordenar_dataframe,
                               combinar_dataframes, concatenar_dataframes,
                               transformar_tipo)

class TestPandasExercises(unittest.TestCase):

    def test_crear_serie(self):
        assert crear_serie([1, 2, 3]).tolist() == [1, 2, 3]

    def test_crear_dataframe(self):
        df = crear_dataframe({'a': [1, 2], 'b': [3, 4]})
        assert df.shape == (2, 2)
        assert list(df.columns) == ['a', 'b']

    def test_seleccionar_columna(self):
        df = pd.DataFrame({'x': [10, 20], 'y': [30, 40]})
        assert seleccionar_columna(df, 'x').tolist() == [10, 20]

    def test_filtrar_por_condicion(self):
        df = pd.DataFrame({'edad': [15, 25, 35]})
        assert filtrar_por_condicion(df, 'edad', 20)['edad'].tolist() == [25, 35]
      
    def test_agregar_columna(self):
        df = pd.DataFrame({'a': [1, 2]})
        assert agregar_columna(df, 'b', [3, 4])['b'].tolist() == [3, 4] 
  
    def test_crear_indexacion_jerarquica(self):
        s = crear_indexacion_jerarquica()
        assert s.loc['a', 1] == 1

    def test_suma_columnas(self):
        df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
        assert suma_columnas(df, 'x', 'y').tolist() == [4, 6]

    def test_rellenar_nulos(self):
        df = pd.DataFrame({'a': [1, None, 3]})
        assert rellenar_nulos(df, 0)['a'].tolist() == [1.0, 0.0, 3.0]
        
    def test_agrupar_y_promediar(self):
        df = pd.DataFrame({'grupo': ['A', 'A', 'B', 'B'], 'valor': [10, 20, 30, 40]})
        resultado = agrupar_y_promediar(df, 'grupo', 'valor')
        assert resultado['A'] == 15 and resultado['B'] == 35

    def test_ordenar_dataframe(self):
        df = pd.DataFrame({'a': [3, 1, 2]})
        assert ordenar_dataframe(df, 'a')['a'].tolist() == [1, 2, 3]

    def test_combinar_dataframes(self):
        df1 = pd.DataFrame({'id': [1, 2], 'valor': ['A', 'B']})
        df2 = pd.DataFrame({'id': [1, 2], 'extra': ['X', 'Y']})
        combinado = combinar_dataframes(df1, df2, 'id')
        assert combinado.shape == (2, 3)

    def test_concatenar_dataframes(self):
        df1 = pd.DataFrame({'a': [1]})
        df2 = pd.DataFrame({'a': [2]})
        concatenado = concatenar_dataframes([df1, df2])
        assert concatenado['a'].tolist() == [1, 2]

    def test_transformar_tipo(self):
        df = pd.DataFrame({'a': ['1', '2']})
        assert transformar_tipo(df, 'a', int)['a'].tolist() == [1, 2]
        
