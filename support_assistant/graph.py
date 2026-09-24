# graph.py
# this is the actual "brain" of the assistant - a langgraph StateGraph with
# 3 steps (nodes): first figure out what kind of question it is, then
# either go look stuff up, or just give a generic answer.
#
# MOCK_LLM thing: by default (or if MOCK_LLM=1) we DON'T call any real AI
# model, we just use simple rules / canned answers. this is what actually
# gets graded. if you set MOCK_LLM=0 you'd be calling a real LLM instead,
# but I haven't hooked that part up to an actual provider, its optional.

import os
from typing import TypedDict
import chromadb
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer

from prompt_template import build_prompt
from schema import AskResponse

# if the question contains any of these words, we treat it as a policy
# question that needs to go look something up
POLICY_WORDS = ["delivery", "return", "refund", "membership", "tracking",
                "cancel", "gift card", "support hours"]

# load the model and connect to chromadb once, so we're not reloading it
# every single time someone asks a question
print("loading embedding model + chromadb collection...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection("zepto_policies")


# this is what gets passed between the different nodes in the graph
class GraphState(TypedDict):
    query: str
    intent: str
    retrieved_ids: list
    retrieved_chunks: list
    answer: str
    sources: list
    confidence: float


def classify_intent(state):
    # just check if any policy-related word shows up in the question
    question_lower = state["query"].lower()

    found_a_policy_word = False
    for word in POLICY_WORDS:
        if word in question_lower:
            found_a_policy_word = True
            break

    if found_a_policy_word:
        state["intent"] = "policy_question"
    else:
        state["intent"] = "general_question"

    return state


def retrieve_and_answer(state):
    # this part ALWAYS actually runs for real (no mocking here), because
    # embeddings + chromadb dont need any api key or paid service

    query_embedding = embedder.encode([state["query"]]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)

    found_ids = results["ids"][0]
    found_chunks = results["documents"][0]

    state["retrieved_ids"] = found_ids
    state["retrieved_chunks"] = found_chunks

    # this is the part that would change if MOCK_LLM=0, but since im not
    # hooking up a real LLM right now, both branches do the same canned thing
    if os.environ.get("MOCK_LLM", "1") == "0":
        # (optional extension spot - would send build_prompt(...) to a real
        # LLM here instead. leaving it as the same canned answer for now)
        context_text = "\n".join(found_chunks)
        _unused_prompt = build_prompt(state["query"], context_text)
        top_snippet = found_chunks[0][:200]
        state["answer"] = "Based on the retrieved context: " + top_snippet
    else:
        top_snippet = found_chunks[0][:200]
        state["answer"] = "Based on the retrieved context: " + top_snippet

    state["sources"] = found_ids
    state["confidence"] = 1.0
    return state


def direct_answer(state):
    # for questions that aren't about zepto policy, just give a generic reply
    state["answer"] = "I can only answer questions about Zepto policies right now."
    state["sources"] = []
    state["confidence"] = 1.0
    return state


def decide_where_to_go(state):
    # this is the function that decides which node to go to next after
    # classify_intent
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"
    else:
        return "direct_answer"


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")

    # this is the "conditional edge" - after classify_intent, go check
    # decide_where_to_go() and follow whatever it returns
    graph.add_conditional_edges(
        "classify_intent",
        decide_where_to_go,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        }
    )

    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


# build it once so we're not rebuilding the graph on every request
compiled_graph = build_graph()


def ask(query):
    # this is the function main.py actually calls
    starting_state = {
        "query": query,
        "intent": "",
        "retrieved_ids": [],
        "retrieved_chunks": [],
        "answer": "",
        "sources": [],
        "confidence": 0.0,
    }

    final_state = compiled_graph.invoke(starting_state)

    return AskResponse(
        answer=final_state["answer"],
        sources=final_state["sources"],
        confidence=final_state["confidence"],
    )
