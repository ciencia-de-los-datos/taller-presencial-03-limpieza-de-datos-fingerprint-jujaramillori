"""Taller evaluable presencial"""

import nltk
import pandas as pd

#Lea el archivo usando pandas y devuelva un DataFrame 

def load_data(input_file):
    """Lea el archivo usando pandas y devuelva un DataFrame"""
    df=pd.read_csv(input_file)
    return df

#print(load_data("input.txt"))


def create_fingerprint(df):
    """Cree una nueva columna en el DataFrame que contenga el fingerprint de la columna 'text'"""

    # 1. Copie la columna 'text' a la columna 'fingerprint'
    # 2. Remueva los espacios en blanco al principio y al final de la cadena
    # 3. Convierta el texto a minúsculas
    # 4. Transforme palabras que pueden (o no) contener guiones por su version sin guion.
    # 5. Remueva puntuación y caracteres de control
    # 6. Convierta el texto a una lista de tokens
    # 7. Transforme cada palabra con un stemmer de Porter
    # 8. Ordene la lista de tokens y remueve duplicados
    # 9. Convierta la lista de tokens a una cadena de texto separada por espacios

    df = df.copy()  # No modifique el DataFrame original
    df["key"] = df["text"]  # Copie la columna 'text' a la columna 
    df["key"] = df["key"].str.strip()  # Remueva los espacios en blanco al principio y al final de la cadena
    df["key"] = df["key"].str.lower()  # Convierta el texto a minúsculas
    df["key"] = df["key"].str.replace("-", "")  # Transforme palabras que pueden (o no) contener guiones por su version sin guion.
    df["key"] = df["key"].str.translate(
        str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    )
    df["key"] = df["key"].str.split()  # Convierta el texto a una lista de tokens
    stemmer = nltk.PorterStemmer()  # Cree un stemmer de Porter: Cambia las palabras parecidas a su raiz
    df["key"] = df["key"].apply(lambda x: [stemmer.stem(word) for word in x])  # Transforme cada palabra con un stemmer de Porter
    df["key"] = df["key"].apply(lambda x: sorted(list(set(x))))  # Ordene la lista de tokens y remueve duplicados
    df["key"] = df["key"].str.join(" ")  # Convierta la lista de tokens a una cadena de texto separada por espacios

    return df


# df = load_data("input.txt")
# df = create_fingerprint(df)
# print(df)         



def generate_cleaned_column(df):
    """Crea la columna 'cleaned' en el DataFrame"""
    # 1. Ordene el dataframe por 'fingerprint' y 'text'
    # 2. Seleccione la primera fila de cada grupo de 'fingerprint'
    # 3.  Cree un diccionario con 'fingerprint' como clave y 'text' como valor
    # 4. Cree la columna 'cleaned' usando el diccionario
    df = df.copy() # No modifique el DataFrame original
    df = df.sort_values(by=["key", "text"], ascending=[True, True]) # Ordene el dataframe por 'fingerprint' y 'text'
    keys = df.drop_duplicates(subset="key", keep="first") # Seleccione la primera fila de cada grupo de 'fingerprint'
    key_dict = dict(zip(keys["key"], keys["text"])) # Cree un diccionario con 'fingerprint' como clave y 'text' como valor
    df["cleaned"] = df["key"].map(key_dict) # Cree la columna 'cleaned' usando el diccionario

    return df # Devuelva el DataFrame modificado


# df = load_data("input.txt")
# df = create_fingerprint(df)
# df = generate_cleaned_column(df)
# print(df)
  

def save_data(df, output_file):
    """Guarda el DataFrame en un archivo"""
    # Solo contiene una columna llamada 'texto' al igual
    # que en el archivo original pero con los datos limpios
    df=df.copy()    
    df=df[["cleaned"]]
    df=df.rename(columns={"cleaned":"text"})
    df.to_csv(output_file, index=False)
    
  

def main(input_file, output_file):
    """Ejecuta la limpieza de datos"""

    df = load_data(input_file)
    df = create_fingerprint(df)
    df = generate_cleaned_column(df)
    df.to_csv("test.csv", index=False)
    save_data(df, output_file)


if __name__ == "__main__":
    main(
        input_file="input.txt",
        output_file="output.txt",
    )
