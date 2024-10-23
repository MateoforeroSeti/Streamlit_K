import streamlit as st
import joblib
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from unidecode import unidecode

def aumentar_peso_verbos(tokens, factor=3):
    nuevo_texto = []
    for palabra in tokens:
        nuevo_texto.append(palabra)
        # Si la palabra es un verbo, repetirla para darle más peso
        if palabra.startswith('permiso') or palabra.startswith('privilegio') or palabra.startswith('despliegue') or palabra.startswith('revision') or palabra.startswith('actualizacion') or palabra.startswith('asignacion') or palabra.startswith('validacion') or palabra.startswith('solicitud'):
            nuevo_texto.extend([palabra] * (factor - 1))
    return " ".join(nuevo_texto)
    
def preprocesar_texto(texto):
    palabras = word_tokenize(unidecode(texto.lower()), language='spanish')
    #pos_tags = [tag for _, tag in bigram_tagger.tag(palabras)]
    #nuevas_palabras = aumentar_peso_verbos(palabras, pos_tags, factor=3)
    nuevas_palabras = aumentar_peso_verbos(palabras, factor=3)
    palabras = word_tokenize(nuevas_palabras, language='spanish')
    #palabras_filtradas = [palabra.lower() for palabra in palabras if palabra.lower() not in stopwords_espanol and palabra.isalpha()]
    palabras_filtradas = [unidecode(palabra.lower()) for palabra in palabras if palabra.isalpha()]
    return " ".join(palabras_filtradas)

def main():

    st.title("Cluster Capacidad")
    st.write("Ingresa los datos para predecir el cluster según el modelo entrenado.")

    st.sidebar.markdown(
        """
        <style>
        .sidebar .sidebar-content{
            font-size: 52px
        }
        </style>
        """,unsafe_allow_html=True
    )

    menu = st.sidebar.radio("Modo de uso", ["Prediccion individual","Predicciones por archivo"])
        
    model = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    if menu == "Prediccion individual":

        #num_features = model.cluster_centers_.shape[1]  # Número de características esperadas por el modelo
        #     
        if 'texto' not in st.session_state:
            st.session_state.texto = ''

        col1, col2 = st.columns([1,3])

        with col1:
            opciones = ["Cambio","Incidente","Petición","Tarea de cambio","otro"]
            servicio = st.selectbox("Tipo de servicio", opciones)
    
        with col2:
            user_input_org = st.text_input('Titulo',value = st.session_state.texto)
            st.session_state.texto = user_input_org
    
        if st.button('Predecir Cluster'):
            if user_input_org:
                user_input = preprocesar_texto(user_input_org)
                user_vector = vectorizer.transform([user_input])
                df_titulos = pd.DataFrame(user_vector.toarray(), columns=vectorizer.get_feature_names_out())
                df_servicio = pd.DataFrame({"SERVICIO":[servicio]})
                datos_kluster = pd.concat([df_servicio, df_titulos],axis = 1)
                scaled_vector = scaler.transform(datos_kluster.select_dtypes(include=[float,int]))
                cluster = model.predict(scaled_vector)[0]
                tiempos = pd.read_csv(r"tiempos.csv", encoding='utf-8', delimiter=';')
                tiempo = tiempos.loc[tiempos['K'] == cluster, 'TIEMPO'].values[0]
                st.success(f"El conjunto de datos ingresado pertenece al cluster: {cluster}")
                st.success(f"El tiempo estimado de la tarea es de : {tiempo} minutos")
            else:
                st.warning("Titulo no definido")
            
    
    elif menu == "Predicciones por archivo":

        archivo = st.file_uploader("Carga de archivo CSV", type=["csv"])

        if st.button('Predecir Cluster'):

            if archivo is not None:
                st.session_state.archivo = archivo

                df = pd.read_csv(archivo, encoding='utf-8', delimiter=';')
                datos_filtrados = df.drop(['HORAS_CIERRE','ID_PETICION','HORA_APERTURA','HORA_CIERRE','DIAS_CIERRE','MIN_CIERRE','AÑO','MES_NUM','MES','ESTADO','CODIGO_CIERRE','ASIGNADO_A','USUARIO'],axis=1)
                datos_kluster = datos_filtrados
                titulos = datos_kluster['TITULO'] 
                documentos_procesados = [preprocesar_texto(doc) for doc in titulos]
                X = vectorizer.transform(documentos_procesados)
                df_titulos = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
                datos_kluster = pd.concat([datos_kluster.drop(['CLIENTE','MODELO_PETICION','TITULO','SUBCATEGORIA','GRUPO_ASIGNACION','CRITICIDAD','TIEMPO_SOLUCION'],axis=1), df_titulos],axis = 1)
                df_normalizado = scaler.transform(datos_kluster.select_dtypes(include=[float,int]))
                clusters = model.predict(df_normalizado)
                datos_filtrados['K']=clusters
                datos_filtrados['Tiempo_estimado']=datos_filtrados.groupby('K')['TIEMPO_SOLUCION'].transform(lambda x: int(x.mean()) if x.notna().any() else 450)
                #datos_filtrados.to_csv('resultado.csv',index=False,sep='|')
                csv = datos_filtrados.to_csv(index=False,sep='|')

                st.dataframe(datos_filtrados, use_container_width=True)

                st.download_button(
                    label="Descargar resultados",
                    data=csv,
                    file_name="datos.csv",
                    mime="texto/csv"
                )

            else:
                st.warning("Archivo no cargado")

    if st.button("Reiniciar"):
        st.session_state.texto = ''
        st.session_state.archivo = None
        st.rerun()

if __name__ == '__main__':
    main()
