"""Score delivery. Write to outbox.jsonl FIRST, then try to send, retry forever.

HINT: if the network is down when a round ends, where does the score live?
TODO: enqueue(event) ; flush(server_url) -> sends pending, removes only what the server accepted.
"""
