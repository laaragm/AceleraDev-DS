import streamlit as st
import pandas as pd
import base64

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return href

def main():
    st.image('logo.png', width= 200)
    st.title('AceleraDev Data Science')
    st.subheader('Week 2 - Python Data Preprocessing Using Pandas')
    st.image('https://media.giphy.com/media/KyBX9ektgXWve/giphy.gif', width=200)
    file  = st.file_uploader('Select a .csv file', type = 'csv')

    if file is not None:
        st.subheader('Data Analysis')
        df = pd.read_csv(file)

        st.markdown('Number of lines:')
        st.markdown(df.shape[0])
        st.markdown('Number of columns:')
        st.markdown(df.shape[1])

        st.markdown('DataFrame Visualization')
        number = st.slider('Choose the number of instances you want to see', min_value=1, max_value=20)
        st.dataframe(df.head(number))
        st.markdown('Column names')
        st.markdown(list(df.columns))

        exploracao = pd.DataFrame({'nomes' : df.columns, 'tipos' : df.dtypes, 'NA #': df.isna().sum(), 'NA %' : (df.isna().sum()/df.shape[0]) * 100})
        
        st.markdown('Types of data:')
        st.write(exploracao.tipos.value_counts())
        st.markdown('int64 columns')
        st.markdown(list(exploracao[exploracao['tipos'] == 'int64']['nomes']))
        st.markdown('float64 columns')
        st.markdown(list(exploracao[exploracao['tipos'] == 'float64']['nomes']))
        st.markdown('object columns*')
        st.markdown(list(exploracao[exploracao['tipos'] == 'object']['nomes']))

        st.markdown('Missing data rate:')
        st.table(exploracao[exploracao['NA #'] != 0][['tipos', 'NA %']])

        st.subheader('Numerical data input:')
        percentual = st.slider('Choose the maximum rate of missing data for the columns you want to input data', min_value=0, max_value=100)
        lista_colunas = list(exploracao[exploracao['NA %']  < percentual]['nomes'])
        
        select_method = st.radio('Choose one:', ('Mean', 'Median'))
        st.markdown(str(select_method))
        if select_method == 'Mean':
            df_inputado = df[lista_colunas].fillna(df[lista_colunas].mean())
            exploracao_inputado = pd.DataFrame({'nomes': df_inputado.columns, 'tipos': df_inputado.dtypes, 'NA #': df_inputado.isna().sum(),
                                       'NA %': (df_inputado.isna().sum() / df_inputado.shape[0]) * 100})
            st.table(exploracao_inputado[exploracao_inputado['tipos'] != 'object']['NA %'])
            st.subheader('Download data below: ')
            st.markdown(get_table_download_link(df_inputado), unsafe_allow_html=True)
        
        if select_method == 'Median':
            df_inputado = df[lista_colunas].fillna(df[lista_colunas].mean())
            exploracao_inputado = pd.DataFrame({'nomes': df_inputado.columns, 'tipos': df_inputado.dtypes, 'NA #': df_inputado.isna().sum(),
                                       'NA %': (df_inputado.isna().sum() / df_inputado.shape[0]) * 100})
            st.table(exploracao_inputado[exploracao_inputado['tipos'] != 'object']['NA %'])
            st.subheader('Download data below: ')
            st.markdown(get_table_download_link(df_inputado), unsafe_allow_html=True)


if __name__ == '__main__':
	main()