# $ streamlit run intro_streamlit.py

import streamlit as st

def main():
    st.title('Hello World')
    
    st.markdown('Button')
    button = st.button('Button')
    if button:
        st.markdown('Clicked')

    check = st.checkbox('Checkbot')
    if check:
        st.markdown('Clicked')

    st.markdown('Radio')
    radio = st.radio('Choose one', ('Opt 1', 'Opt 2'))
    if radio == 'Opt 1':
        st.markdown('Opt 1')
    if radio == 'Opt 2':
        st.markdown('Opt 2')
    
    multi = st.multiselect('Choose', ('Opt 1', 'Opt 2'))
    if multi == 'Opt 1':
        st.markdown('Opt 1')
    if multi == 'Opt 2':
        st.markdown('Opt 2')

    st.markdown('Selectbox')
    select = st.selectbox('Choose opt', ('Opt 1', 'Opt 2'))
    if select == 'Opt 1':
        st.markdown('Opt 1')
    if select == 'Opt 2':
        st.markdown('Opt 2')
    
    st.markdown('File uploader')
    file_uploader = st.file_uploader('Choose your file', type='csv')
    if file_uploader is not None:
        st.markdown('File uploaded successfully')

if __name__ == '__main__':
    main()