# Docs for v1 can be found by changing the above selector ^
from together import Together
import os

client = Together(
    api_key=os.environ.get("TOGETHER_API_KEY"),
)
for b in client.batches.list():
    try:
        print(b.id, b.status)
        client.batches.cancel(b.id)
    except Exception as e:
        print(f"Error cancelling batch {b.id}: {e}")