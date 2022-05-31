import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle

def main():
    df = load_data()
    page = st.sidebar.selectbox("Choose a page", ["Homepage", "Exploration","Model"])
    
    with open('myfile.pkl', 'rb') as pkl_file:
        tree_from_file = pickle.load(pkl_file)

    if page == "Homepage":
        st.header("Данные для задачи классификации банкнот.")
        st.write("Данные для датасета были извлечены из изображений, снятых с подлинных и поддельных банкнотоподобных образцов. Для оцифровки использовалась промышленная камера, обычно используемая для проверки печати.Инструмент Wavelet Transform использовался для извлечения признаков из изображений.")
        st.write("1) variance of Wavelet Transformed image (дисперсия вейвлет-преобразованного изображения), тип вещественный.")
        st.write("2) skewness of Wavelet Transformed image (асимметрия вейвлет-преобразованного изображения), тип вещественный.")
        st.write("3) curtosis of Wavelet Transformed image (эксцесс преобразованного изображения), тип вещественный.")
        st.write("4) entropy of image (энтропия изображения), тип вещественный.")
        st.write(df)
    elif page == "Exploration":
        st.title("Data Exploration")
        visualize_data(df)
    else:
        st.title("Model")
        varience = st.slider("variance: ", min_value=df['variance'].min(),   
                       max_value=df['variance'].max(), value=0.0, step=0.01)
        skewness = st.slider("skewness: ", min_value=df['skewness'].min(),   
                       max_value=df['skewness'].max(), value=0.0, step=0.01)
        curtosis = st.slider("curtosis: ", min_value=df['curtosis'].min(),   
                       max_value=df['curtosis'].max(), value=0.0, step=0.01)
        entropy = st.slider("entropy: ", min_value=df['entropy'].min(),   
                       max_value=df['entropy'].max(), value=0.0, step=0.01)
        pred = tree_from_file.predict(np.array([[varience,skewness,curtosis,entropy]]))[0]
        if pred == 0:
            st.write("Эта купюра настоящая")
        else :
            st.write("Это подделка")
        

@st.cache
def load_data():
    df = pd.read_csv('preprocessing_data_banknote_authentication.csv')
    return df

def visualize_data(df):
    c = alt.Chart(df).mark_circle().encode(x='variance', y='skewness',  
                                       color='class')
    st.write(c)
    
    #fields = ['varience', 'skewness']
    #x = alt.Chart(df).mark_point().encode(
    #alt.X(alt.repeat("column"), type='quantitative'),
    #alt.Y(alt.repeat("row"), type='quantitative'),
    #color='class'
    #).properties(
    #width=200,
    #height=200
    #).repeat(
    #row=fields,
    #column=fields[::-1]
    #).interactive()
    #st.write(x)

if __name__ == "__main__":
    main()
