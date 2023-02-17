from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from chat_chain import VectorChatVectorDBChain


prompt_template = """You are an AI assistant for answering questions about the most recent state of the union address.
You are given the following extracted parts of a long document, the chat history and a question. Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about the most recent state of the union, politely inform them that you are tuned to only answer questions about the most recent state of the union.

{context}

Chat History:
{chat_history}
Human: {question}
Assistant:"""
QA_PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question", "chat_history"],
)


def get_chain(vectorstore, m=0.8):
    llm = OpenAI(temperature=0)
    qa_chain = VectorChatVectorDBChain.from_llm(
        llm,
        vectorstore,
        qa_prompt=QA_PROMPT,
        k=3,
        search_kwargs={},
        m=m,
        return_vec=False,
    )
    return qa_chain
