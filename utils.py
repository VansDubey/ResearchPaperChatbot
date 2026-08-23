import random
import string


def generate_random_id(length=5):
    # Generate a random string of letters and digits
    random_id = "".join(random.choices(string.ascii_letters + string.digits, k=length))
    return random_id


def ask_question(chain, query, session_id):
    config = {"configurable": {"session_id": session_id}}
    for chunk in chain.stream({"input": query}, config):

        if "answer" in chunk:
            yield chunk["answer"]
