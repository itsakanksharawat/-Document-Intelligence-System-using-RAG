from quickstart import build_index
from src.rag import ask_rag


print("Building/loading index...")
index = build_index()

print("\nRunning RAG question...")

result = ask_rag(
    question="How many casual leaves does Google India provide?",
    index=index,
    company_id="google",
)

print("\n" + "=" * 80)
print("FINAL ANSWER")
print("=" * 80)

print(result["answer"])

print("\n" + "=" * 80)
print("SOURCES")
print("=" * 80)

for source in result["sources"]:
    print(f"- {source}")
    