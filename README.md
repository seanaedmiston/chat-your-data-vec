# Vector memory

Demonstrates using a vector memory in a vector db qa chain using the chat-your-data challenge as a starting point.
Rough idea is that instead of searching against a direct embedding of a question, the search is against a 'history vector'
The history vector is formed by adding an embedding of the current question answer pair to a factor 'm' of the previous history
For m < 1 this creates a vector that has some memory of the entire conversation by prioritises recent history

# Chat-Your-Data

Create a ChatGPT like experience over your custom docs using [LangChain](https://github.com/hwchase17/langchain).

See [this blog post](https://blog.langchain.dev/tutorial-chatgpt-over-your-data/) for a more detailed explanation.

## Ingest data

Ingestion of data is done over the `state_of_the_union.txt` file. 
Therefor, the only thing that is needed is to be done to ingest data is run `python ingest_data.py`

## Query data
Custom prompts are used to ground the answers in the state of the union text file.

## Running the Application

By running `python app.py` from the command line you can easily interact with your ChatGPT over your own data.
