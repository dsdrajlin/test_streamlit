import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar datos y entrenar modelo
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Interfaz de usuario con Streamlit
st.title('Predicción con Machine Learning para Iris Dataset')

# Añadir controles de usuario
sepal_length = st.slider('Sepal Length', min_value=iris.data[:, 0].min(), max_value=iris.data[:, 0].max(), value=5.0)
sepal_width = st.slider('Sepal Width', min_value=iris.data[:, 1].min(), max_value=iris.data[:, 1].max(), value=3.0)
petal_length = st.slider('Petal Length', min_value=iris.data[:, 2].min(), max_value=iris.data[:, 2].max(), value=4.0)
petal_width = st.slider('Petal Width', min_value=iris.data[:, 3].min(), max_value=iris.data[:, 3].max(), value=1.3)

# Realizar la predicción con el modelo
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)

# Mostrar la predicción
st.subheader('Resultado:')
st.write(f'La predicción es: {iris.target_names[prediction][0]}')
