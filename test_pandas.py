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
        result = divide_arrays(arr1, arr2)
        expected = np.array([2, 2.5, 3])
        np.testing.assert_array_equal(result, expected)

    def test_stats(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = stats(arr)
        expected = (3.0, 3.0, 1.4142135623730951)
        self.assertEqual(result, expected)

    def test_first_5(self):
        arr = np.random.randint(0, 100, 10)
        result = first_5(arr)
        expected = arr[:5]
        np.testing.assert_array_equal(result, expected)

    def test_last_3(self):
        arr = np.random.randint(0, 100, 10)
        result = last_3(arr)
        expected = arr[-3:]
        np.testing.assert_array_equal(result, expected)

    def test_indices_2_4_6(self):
        arr = np.random.randint(0, 100, 10)
        result = indices_2_4_6(arr)
        expected = arr[[2, 4, 6]]
        np.testing.assert_array_equal(result, expected)

    def test_greater_50(self):
        arr = np.random.randint(0, 100, 10)
        result = greater_50(arr)
        expected = arr[arr > 50]
        np.testing.assert_array_equal(result, expected)

    def test_less_7(self):
        arr = np.random.randint(0, 10, 10)
        result = less_7(arr)
        expected = arr[arr <= 7]
        np.testing.assert_array_equal(result, expected)

    def test_reshape_2x6(self):
        arr = np.arange(12)
        result = reshape_2x6(arr)
        expected = arr.reshape(2, 6)
        np.testing.assert_array_equal(result, expected)

    def test_reshape_2x3x4(self):
        arr = np.arange(24)
        result = reshape_2x3x4(arr)
        expected = arr.reshape(2, 3, 4)
        np.testing.assert_array_equal(result, expected)

    def test_reshape_10x10(self):
        arr = np.arange(100)
        result = reshape_10x10(arr)
        expected = arr.reshape(10, 10)
        np.testing.assert_array_equal(result, expected)

    def test_reshape_10x10x10(self):
        arr = np.arange(1000)
        result = reshape_10x10x10(arr)
        expected = arr.reshape(10, 10, 10)
        np.testing.assert_array_equal(result, expected)

    def test_reshape_10x10x10x10(self):
        arr = np.arange(10000)
        result = reshape_10x10x10x10(arr)
        expected = arr.reshape(10, 10, 10, 10)
        np.testing.assert_array_equal(result, expected)

    def test_add_broadcast(self):
        arr1 = np.array([[1, 2, 3], [4, 5, 6]])
        arr2 = np.array([[1], [2]])
        result = add_broadcast(arr1, arr2)
        expected = arr1 + arr2
        np.testing.assert_array_equal(result, expected)

    def test_subtract_broadcast(self):
        arr1 = np.array([[1, 2], [3, 4], [5, 6]])
        arr2 = np.array([[1, 2, 3], [4, 5, 6]])
        result = subtract_broadcast(arr1, arr2)
        expected = arr1 - arr2.T
        np.testing.assert_array_equal(result, expected)

    def test_multiply_broadcast(self):
        arr1 = np.array([[1, 2, 3], [4, 5, 6]])
        arr2 = np.array([[1, 2], [3, 4], [5, 6]])
        result = multiply_broadcast(arr1, arr2)
        expected = arr1 @ arr2
        np.testing.assert_array_equal(result, expected)

    def test_divide_broadcast(self):
        arr1 = np.array([[1, 2, 3], [4, 5, 6]])
        arr2 = np.array([[1], [2]])
        result = divide_broadcast(arr1, arr2)
        expected = arr1 / arr2
        np.testing.assert_array_equal(result, expected)

    def test_element_wise_product(self):
        arr1 = np.array([[1, 2, 3], [4, 5, 6]])
        arr2 = np.array([[1, 2, 3], [4, 5, 6]])
        result = element_wise_product(arr1, arr2)
        expected = arr1 * arr2
        np.testing.assert_array_equal(result, expected)

    def test_temp_data(self):
        # Crear un arreglo de numpy con temperaturas de prueba
        temps = np.array([10, 20, 30, 5, 15, 25, 35, 12, 28])
        
        # Capturar la salida de la función
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        # Llamar a la función con los datos de prueba
        temp_data(temps)
        
        # Restaurar la salida estándar
        sys.stdout = sys.__stdout__
        
        # Obtener la salida capturada
        output = captured_output.getvalue()
        
        # Verificar que la salida sea la esperada
        assert "Temperaturas mayores a 25 grados: [30 35 28]" in output
        assert "Número de días con temperatura menor a 15 grados: 3" in output
        
        print("La prueba unitaria ha pasado exitosamente.")

    def test_rainfall_data(self):
        # Crear un arreglo 2D de numpy con datos de lluvia de prueba
        rainfall = np.array([
            [50, 120, 80],
            [110, 90, 130],
            [70, 60, 140]
        ])
        
        # Capturar la salida de la función
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        # Llamar a la función con los datos de prueba
        rainfall_data(rainfall)
        
        # Restaurar la salida estándar
        sys.stdout = sys.__stdout__
        
        # Obtener la salida capturada
        output = captured_output.getvalue()
        
        # Verificar que la salida sea la esperada
        assert "Índices de las ciudades con más de 100 mm de lluvia: [1 3 5 8]" in output
        
        print("La prueba unitaria ha pasado exitosamente.")

    def test_image_thresholding(self):
# Crear un arreglo 2D de numpy con datos de imagen de prueba
        image = np.array([
            [100, 150, 200],
            [50, 125, 175],
            [0, 255, 128]
        ])
        
        # Resultado esperado después del umbral
        expected_output = np.array([
            [0, 255, 255],
            [0, 0, 255],
            [0, 255, 255]
        ])
        
        # Llamar a la función con los datos de prueba
        output = image_thresholding(image)
        
        # Verificar que la salida sea la esperada
        assert np.array_equal(output, expected_output), f"Expected {expected_output}, but got {output}"
        
        print("La prueba unitaria ha pasado exitosamente.")

    def test_diagonals(self):
        # Crear un arreglo 2D de numpy de 5x5 con datos de prueba
        matrix = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25]
        ])
        

        expected_output = (np.array([1, 7, 13, 19, 25]), np.array([5, 9, 13, 17, 21]))
        
        # Llamar a la función con los datos de prueba
        output = matrix_diagonals(matrix)
        
        # Verificar que la salida sea la esperada
        assert np.array_equal(output[0], expected_output[0]), f"Expected {expected_output[0]}, but got {output[0]}"
        assert np.array_equal(output[1], expected_output[1]), f"Expected {expected_output[1]}, but got {output[1]}"
        
        print("La prueba unitaria ha pasado exitosamente.")
