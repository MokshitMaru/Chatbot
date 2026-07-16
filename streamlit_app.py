from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
from pathlib import Path
import os
from dotenv import load_dotenv




BASE_DIR = Path(__file__).parent

load_dotenv(BASE_DIR / ".env")




db_name = str(BASE_DIR / "vector_pdf")


embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


vector = Chroma(
    persist_directory=db_name,
    embedding_function=embeddings
)


retriever = vector.as_retriever()



llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


SYSTEM_PROMPT = """
You are an expert PDF analyser and helpful assistant.

Use the context below to answer PDF-related questions.

If the question is related to the PDF and the answer is not present,
say "I don't know".

For general questions, casual conversation, jokes,
and greetings, answer normally.

Context:

{context}
"""



if "history" not in st.session_state:

    st.session_state.history = []




def answer_question(question):


    docs = retriever.invoke(question)


    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )


    messages = []


    messages.append(
        SystemMessage(
            content=SYSTEM_PROMPT.format(
                context=context
            )
        )
    )


    messages.extend(
        st.session_state.history
    )



    messages.append(
        HumanMessage(
            content=question
        )
    )


    response = llm.invoke(messages)


    return response.content


st.markdown(
    """
    <h1 style="text-align:center;">🧠 My Assistant</h1>
    <hr>
    """,
    unsafe_allow_html=True
)

question = st.text_input(
    "Ask your question"
)


if st.button("Submit"):


    if question:



        st.session_state.history.append(
            HumanMessage(
                content=question
            )
        )


        answer = answer_question(question)



        st.session_state.history.append(
            AIMessage(
                content=answer
            )
        )
st.write("\n")


st.write("## Conversation")


for message in st.session_state.history:


    if isinstance(message, HumanMessage):

        left, right = st.columns([1, 2])

        with right:
            st.write(
                "👤 User:"
            )

            st.write(
                message.content
            )


    elif isinstance(message, AIMessage):

        left, right = st.columns([2, 1])

        with left:
            st.write(
                "🤖 Assistant:"
            )

            st.write(
                message.content
            )

    st.write("\n")