import unittest
import numpy as np
import pandas as pd
from ejercicios_pandas import (crear_serie, crear_dataframe, seleccionar_columna,
                               filtrar_por_condicion, agregar_columna,
                               crear_indexacion_jerarquica, suma_columnas,
                               rellenar_nulos, agrupar_y_promediar, ordenar_dataframe,
                               combinar_dataframes, concatenar_dataframes,
                               transformar_tipo)

class TestNumpyExercises(unittest.TestCase):

    def test_rand_int(self):
        '''TODO: Reemplaza prueba por numeros enteros entre 0 100 y logitud de ndarray'''
        result = rand_int()
        assert result.size == 10
        assert result.dtype == np.int64

    def test_rand_float(self):
        '''TODO: reemplaza prueba por números flotantes entre 0 y 1 y logitud de ndarray'''
        np.random.seed(10)
        result = rand_float()
        assert result.size == 5
        assert result.dtype == np.float64

    def test_first_10_primes(self):
        result = first_10_primes()
        expected = np.array([ 2,  3,  5,  7, 11, 13, 17, 19, 23, 29])
        np.testing.assert_array_equal(result, expected)

    def test_squares(self):
        result = squares()
        expected = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
        np.testing.assert_array_equal(result, expected)

    def test_cubes(self):
        result = cubes()
        expected = np.array([1, 8, 27, 64, 125, 216, 343, 512, 729, 1000])
        np.testing.assert_array_equal(result, expected)

    def test_add_arrays(self):
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        result = add_arrays(arr1, arr2)
        expected = np.array([5, 7, 9])
        np.testing.assert_array_equal(result, expected)

    def test_subtract_arrays(self):
        arr1 = np.array([4, 5, 6])
        arr2 = np.array([1, 2, 3])
        result = subtract_arrays(arr1, arr2)
        expected = np.array([3, 3, 3])
        np.testing.assert_array_equal(result, expected)

    def test_multiply_arrays(self):
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        result = multiply_arrays(arr1, arr2)
        expected = np.array([4, 10, 18])
        np.testing.assert_array_equal(result, expected)

    def test_divide_arrays(self):
        arr1 = np.array([4, 5, 6])
        arr2 = np.array([2, 2, 2])
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
