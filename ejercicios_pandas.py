import pandas as pd
import doctest

## Funciones de Pandas
# --------------------------------------------------

def crear_serie(lista):
    """
    Crea una Serie de Pandas a partir de una lista.

    >>> crear_serie([1, 2, 3]).tolist()
    [1, 2, 3]
    """
    assert isinstance(lista, list), "La entrada debe ser una lista"
    return pd.Series(lista)

# --------------------------------------------------

def seleccionar_columna(df, columna):
    """
    Selecciona una columna de un DataFrame.

    >>> df = pd.DataFrame({'x': [10, 20], 'y': [30, 40]})
    >>> seleccionar_columna(df, 'x').tolist()
    [10, 20]
    """
    assert columna in df.columns, f"La columna '{columna}' no existe en el DataFrame"
    return df[columna]

# --------------------------------------------------

def filtrar_por_condicion(df, columna, valor):
    """
    Filtra filas donde la columna es mayor que el valor dado.

    >>> df = pd.DataFrame({'edad': [15, 25, 35]})
    >>> filtrar_por_condicion(df, 'edad', 20)['edad'].tolist()
    [25, 35]
    """
    assert columna in df.columns, f"La columna '{columna}' no existe en el DataFrame"
    return df[df[columna] > valor]

# --------------------------------------------------

def crear_indexacion_jerarquica():
    """
    Crea una Serie con indexación jerárquica (MultiIndex).

    >>> s = crear_indexacion_jerarquica()
    >>> s.loc['a', 1]
    np.int64(1)
    """
    s = pd.Series(range(1, 5), index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]])
    return s

# --------------------------------------------------

def suma_columnas(df, col1, col2):
    """
    Suma dos columnas de un DataFrame.

    >>> df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
    >>> suma_columnas(df, 'x', 'y').tolist()
    [4, 6]
    """
    assert col1 in df.columns and col2 in df.columns, f"Las columnas '{col1}' y '{col2}' no existen en el DataFrame"
    return df[col1] + df[col2]

# --------------------------------------------------

def rellenar_nulos(df, valor):
    """
    Rellena valores nulos con un valor dado.
    Se usa una copia del DataFrame para evitar modificar el original.

    >>> df = pd.DataFrame({'a': [1, None, 3]})
    >>> rellenar_nulos(df, 0)['a'].tolist()
    [1.0, 0.0, 3.0]
    """
    # Se elimina inplace=True para no modificar el df original del doctest
    return df.fillna(valor)

# --------------------------------------------------

def agrupar_y_promediar(df, columna_agrupacion, columna_valores):
    """
    Agrupa por una columna y calcula el promedio de otra.

    >>> df = pd.DataFrame({'grupo': ['A', 'A', 'B', 'B'], 'valor': [10, 20, 30, 40]})
    >>> resultado = agrupar_y_promediar(df, 'grupo', 'valor')
    >>> resultado['A'] == 15 and resultado['B'] == 35
    np.True_
    """
    assert columna_agrupacion in df.columns and columna_valores in df.columns, "Las columnas de agrupación o valores no existen."
    # El resultado de groupby.mean() es una Serie
    return df.groupby(columna_agrupacion)[columna_valores].mean()

# --------------------------------------------------

def ordenar_dataframe(df, columna):
    """
    Ordena un DataFrame por una columna.

    >>> df = pd.DataFrame({'a': [3, 1, 2]})
    >>> ordenar_dataframe(df, 'a')['a'].tolist()
    [1, 2, 3]
    """
    assert columna in df.columns, f"La columna '{columna}' no existe en el DataFrame"
    # Se agrega .reset_index(drop=True) para que el doctest sea robusto
    return df.sort_values(by=columna).reset_index(drop=True)

# --------------------------------------------------

def combinar_dataframes(df1, df2, clave):
    """
    Combina dos DataFrames usando merge por una clave.

    >>> df1 = pd.DataFrame({'id': [1, 2], 'valor': ['A', 'B']})
    >>> df2 = pd.DataFrame({'id': [1, 2], 'extra': ['X', 'Y']})
    >>> combinado = combinar_dataframes(df1, df2, 'id')
    >>> combinado.shape
    (2, 3)
    """
    return pd.merge(df1, df2, on=clave)

# --------------------------------------------------

def concatenar_dataframes(lista_df):
    """
    Concatena una lista de DataFrames.

    >>> df1 = pd.DataFrame({'a': [1], 'b': ['x']})
    >>> df2 = pd.DataFrame({'a': [2], 'b': ['y']})
    >>> concatenado = concatenar_dataframes([df1, df2])
    >>> concatenado['a'].tolist()
    [1, 2]
    """
    assert all(isinstance(df, pd.DataFrame) for df in lista_df), "Todos los elementos de la lista deben ser DataFrames"
    return pd.concat(lista_df)

# --------------------------------------------------

def transformar_tipo(df, columna, tipo):
    """
    Transforna el tipo de datos de una columna.

    >>> df = pd.DataFrame({'a': ['1', '2']})
    >>> transformar_tipo(df, 'a', int).tolist()
    [1, 2]
    """
    assert columna in df.columns, f"La columna '{columna}' no existe en el DataFrame"
    return df[columna].astype(tipo)

# --------------------------------------------------

# Ejecutar pruebas doctest
if __name__ == "__main__":
    doctest.testmod(verbose=True)
