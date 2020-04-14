# $ streamlit run pandas_streamlit.py

import streamlit as st
import pandas as pd

def main():
    st.title('AceleraDev Data Science')
    st.image('logo.png')
    file_uploader = st.file_uploader('Upload your file', type='csv')

    if file_uploader is not None:
        slider = st.slider('Valores', 1,100)
        df = pd.read_csv(file_uploader)
        st.dataframe(df.head(slider))
        st.markdown('Markdown')
        st.table(df.head(slider))
        st.write(df.columns)
        st.table(df.groupby('species')['petal_width'])

if __name__ == '__main__':
    main()