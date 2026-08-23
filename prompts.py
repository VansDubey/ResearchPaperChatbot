from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def create_history_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Given a chat history and the latest user question
                    which might reference context in the chat history, formulate a standalone question
                    which can be understood without the chat history. Do NOT answer the question,
                    just reformulate it if needed and otherwise return it as is.""",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


def create_qa_prompt(metadata):
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an assistant chatbot that helps with understanding research papers from Arxiv.
                    Tone: Concise, friendly, easy to understand, Human conversation-like, Talk in third person.
                    Rules: Be cautious while reading the Retrieved matching document as it may not be relevant all the time.
                    If you are asked random questions or you don't know something, please respectfully decline.""",
            ),
            (
                "system",
                f'Here is a research paper from Arxiv: Title: {metadata.get("Title")}, Authors: {metadata.get("Authors")}, Abstract: {metadata.get("Summary")}',
            ),
            (
                "system",
                "Retrieved matching document \n {context}",
            ),
            (
                "human",
                "{input}",
            ),
        ]
    )
