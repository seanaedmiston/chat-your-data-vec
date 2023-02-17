"""Chain for chatting with a vector database."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
from operator import add
from pydantic import BaseModel, Field

from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate
from langchain.vectorstores.base import VectorStore

from langchain.prompts.prompt import PromptTemplate


prompt_template = """Use the following pieces of context to answer the question at the end of the conversation. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Chat History:
{chat_history}
Human: {question}
Assistant:"""
QA_PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question", "chat_history"],
)


def _get_chat_history(chat_history: List[Tuple[str, str]]) -> str:
    buffer = ""
    for human_s, ai_s in chat_history:
        human = "Human: " + human_s
        ai = "Assistant: " + ai_s
        buffer += "\n" + "\n".join([human, ai])
    return buffer


class VectorChatVectorDBChain(Chain, BaseModel):
    """Chain for chatting with a vector database."""

    vectorstore: VectorStore
    combine_docs_chain: BaseCombineDocumentsChain
    # question_generator: LLMChain
    output_key: str = "answer"
    k: int = 4
    """Number of documents to query for."""
    m: float = 0.8
    """Magic decay factor applied to vectors"""
    search_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Extra search args."""
    search_type: str = "similarity"
    """Search type to use over vectorstore. `similarity` or `mmr`."""
    return_vec: bool = False
    """Return the embedding of query"""

    @property
    def _chain_type(self) -> str:
        return "vector-chat-vector-db"

    @property
    def input_keys(self) -> List[str]:
        """Input keys."""
        return ["question", "chat_history"]

    @property
    def output_keys(self) -> List[str]:
        """Output keys."""
        _output_keys = [self.output_key]
        if self.return_vec:
            _output_keys = _output_keys + ["history_vec"]
        return _output_keys

    @classmethod
    def from_llm(
        cls,
        llm: BaseLLM,
        vectorstore: VectorStore,
        qa_prompt: BasePromptTemplate = QA_PROMPT,
        chain_type: str = "stuff",
        k: int = 4,
        m: float = 0.8,
        search_kwargs: Dict[str, Any] = Field(default_factory=dict),
        search_type: str = "similarity",
        return_vec: bool = False,
    ) -> VectorChatVectorDBChain:
        """Load chain from LLM."""
        doc_chain = load_qa_chain(
            llm,
            chain_type=chain_type,
            prompt=qa_prompt,
        )
        return cls(
            vectorstore=vectorstore,
            combine_docs_chain=doc_chain,
            k=k,
            m=m,
            search_kwargs=search_kwargs,
            search_type=search_type,
            return_vec=return_vec,
        )

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        question = inputs["question"]
        history_vec = inputs["history_vec"]
        chat_history_str = _get_chat_history(inputs["chat_history"])
        if chat_history_str:
            new_vec = list(
                map(
                    add,
                    self.vectorstore.embedding_function(question),
                    [q * self.m for q in history_vec],
                )
            )
        else:
            new_vec = self.vectorstore.embedding_function(question)
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search_by_vector(
                new_vec, k=self.k, **self.search_kwargs
            )
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search_by_vector(
                new_vec, k=self.k, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        new_inputs = inputs.copy()
        new_inputs["question"] = question
        new_inputs["chat_history"] = chat_history_str
        answer, _ = self.combine_docs_chain.combine_docs(docs, **new_inputs)
        if self.return_vec:
            return {self.output_key: answer, "history_vec": new_vec}
        else:
            return {self.output_key: answer}
