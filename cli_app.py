import pickle
from query_data import get_chain
from operator import add

M = 0.8

if __name__ == "__main__":
    with open("vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    qa_chain = get_chain(vectorstore, M)
    chat_history = []
    history_vec = []
    print("Chat with your docs!")
    while True:
        print("Human:")
        question = input()
        if question == "q":
            break
        elif question == "r":
            chat_history = []
            history_vec = []
            continue

        result = qa_chain(
            {
                "question": question,
                "chat_history": chat_history,
                "history_vec": history_vec,
            }
        )
        chat_history.append((question, result["answer"]))
        qa_vec = vectorstore.embedding_function(
            "Human: " + question + "\n" + "Assistant: " + result["answer"]
        )
        if len(history_vec):
            history_vec = list(
                map(
                    add,
                    qa_vec,
                    [q * M for q in history_vec],
                )
            )
        else:
            history_vec = qa_vec

        print("AI:")
        print(result["answer"])
