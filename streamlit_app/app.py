import pathlib
import sys
import streamlit as st
from streamlit_chat import message
tool_path = f"{pathlib.Path().resolve().parents[0]}/tools"
sys.path.append(tool_path)
from tools import (get_wiki_entry,
                   get_tokens,
                   get_qa)


class WikiStreamlitApp:
    def __init__(self):
        self.build_app()
        self.initialize_session()
        self.clear_button = st.sidebar.button("Clear Conversation", key="clear")

    @staticmethod
    def initialize_session():
        """"""
        if 'generated' not in st.session_state:
            st.session_state['generated'] = []
        if 'past' not in st.session_state:
            st.session_state['past'] = []
        if 'messages' not in st.session_state:
            st.session_state['messages'] = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]
        if 'model_name' not in st.session_state:
            st.session_state['model_name'] = []
        if 'cost' not in st.session_state:
            st.session_state['cost'] = []
        if 'total_tokens' not in st.session_state:
            st.session_state['total_tokens'] = []
        if 'total_cost' not in st.session_state:
            st.session_state['total_cost'] = 0.0

    @staticmethod
    def build_app():
        st.set_page_config(page_title="Wiki explorer", page_icon=":robot_face:")
        st.markdown("<h1 style='text-align: center;'>Wiki RAG Bot</h1>", unsafe_allow_html=True)

    @staticmethod
    def generate_response(prompt):
        st.session_state['messages'].append({"role": "user", "content": prompt})

        #query = "who attacked Israel in 2023?"
        document = get_wiki_entry(prompt)
        tokens = get_tokens(document)
        qa = get_qa(tokens)
        generated_text = qa({"query": prompt})
        st.session_state['messages'].append({"role": "assistant", "content": generated_text['result']})
        return generated_text["result"]


if __name__ == '__main__':

    ms = WikiStreamlitApp()

    # container for chat history
    response_container = st.container()
    # container for text box
    container = st.container()
    if ms.clear_button:
        st.session_state['generated'] = []
        st.session_state['past'] = []
        st.session_state['messages'] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("You:", key='input', height=100)
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = WikiStreamlitApp.generate_response(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
