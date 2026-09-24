# main.py
# this wraps our graph in an actual web api using fastapi, so it can be
# called like a normal http endpoint instead of just running in python

from fastapi import FastAPI
from graph import ask
from schema import AskRequest, AskResponse

app = FastAPI(title="Zepto Support Assistant")


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest):
    # just pass the query straight into our graph function and return
    # whatever it gives back
    return ask(request.query)


@app.get("/health")
def health_check():
    # just a simple way to check the server is actually running
    return {"status": "ok"}
