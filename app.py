import json
import sqlite3
import numpy as np
from foundry_local_sdk import Configuration, FoundryLocalManager

DB_FILE = "rag_knowledge_base.db"


def init_db():
  conn = sqlite3.connect(DB_FILE)
  cursor = conn.cursor()
  cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            content TEXT,
            embedding TEXT
        )
    """)
  conn.commit()
  conn.close()


def save_chunk(source: str, content: str, embedding: list[float]):
  conn = sqlite3.connect(DB_FILE)
  cursor = conn.cursor()
  cursor.execute(
      "INSERT INTO documents (source, content, embedding) VALUES (?, ?, ?)",
      (source, content, json.dumps(embedding)),
  )
  conn.commit()
  conn.close()


def get_all_chunks():
  conn = sqlite3.connect(DB_FILE)
  cursor = conn.cursor()
  cursor.execute("SELECT id, source, content, embedding FROM documents")
  rows = cursor.fetchall()
  conn.close()
  chunks = []
  for row in rows:
    chunks.append({
        "id": row[0],
        "source": row[1],
        "content": row[2],
        "embedding": json.loads(row[3]),
    })
  return chunks


def cosine_similarity(v1, v2):
  vec1 = np.array(v1, dtype=np.float32)
  vec2 = np.array(v2, dtype=np.float32)
  norm1 = np.linalg.norm(vec1)
  norm2 = np.linalg.norm(vec2)
  if norm1 == 0 or norm2 == 0:
    return 0.0
  return float(np.dot(vec1, vec2) / (norm1 * norm2))


class LocalRAGAssistant:

  def __init__(self):
    print("\n[*] Foundry Local başlatılıyor...")
    config = Configuration(app_name="LocalRAG")
    FoundryLocalManager.initialize(config)
    self.manager = FoundryLocalManager.instance
    self.catalog = self.manager.catalog

    print("[*] Embedding modeli hazırlanıyor...")
    self.embed_model = self.catalog.get_model("qwen3-embedding-0.6b")
    self.embed_model.load()
    self.embed_client = self.embed_model.get_embedding_client()

    print("[*] Chat modeli (Phi-3.5) hazırlanıyor...")
    self.chat_model = self.catalog.get_model("phi-3.5-mini")
    self.chat_model.load()
    self.chat_client = self.chat_model.get_chat_client()

    init_db()

  def get_embedding(self, text: str) -> list[float]:
    response = self.embed_client.generate_embedding(text)
    return response.data[0].embedding

  def ingest_document(self, source_name: str, text: str):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    print(
        f"[*] '{source_name}' belgesi indeksleniyor ({len(paragraphs)} parça)..."
    )
    for chunk in paragraphs:
      emb = self.get_embedding(chunk)
      save_chunk(source=source_name, content=chunk, embedding=emb)
    print("[+] Belge SQLite veritabanına başarıyla kaydedildi.")

  def retrieve(self, query: str, top_k: int = 2) -> list[dict]:
    query_embedding = self.get_embedding(query)
    all_chunks = get_all_chunks()
    if not all_chunks:
      return []
    scored_chunks = []
    for ch in all_chunks:
      score = cosine_similarity(query_embedding, ch["embedding"])
      scored_chunks.append({
          "source": ch["source"],
          "content": ch["content"],
          "score": score,
      })
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks[:top_k]

  def answer_query(self, user_question: str) -> str:
    relevant_chunks = self.retrieve(user_question, top_k=2)
    if not relevant_chunks:
      return "Veritabanında henüz kayıtlı doküman bulunmuyor."

    context_parts = [
        f"[Kaynak: {c['source']}]\n{c['content']}" for c in relevant_chunks
    ]
    context_str = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "Sen yalnızca sana verilen bağlamdaki bilgileri kullanarak soruları"
        " yanıtlayan bir asistansın.\n"
        "Kurallar:\n"
        "1. Yanıtını sadece sağlanan bağlama dayandır.\n"
        "2. Bilgi bağlamda yoksa 'Bu bilgi belgelerde yer almıyor.' de.\n"
        "3. Hangi kaynaktan aldığını belirt.\n\n"
        f"BAĞLAM:\n{context_str}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
    ]

    resp = self.chat_client.complete_chat(messages=messages)
    if hasattr(resp, "choices"):
      return resp.choices[0].message.content
    elif hasattr(resp, "content"):
      return resp.content
    return str(resp)


def main():
  assistant = LocalRAGAssistant()

  if len(get_all_chunks()) == 0:
    sample_doc = (
        "Foundry Local, Microsoft tarafından sunulan ve modelleri yerel"
        " cihazda çalıştıran bir araçtır.\n\n"
        "İnternet bağlantısına ihtiyaç duymaz, veriler bilgisayarda kalır ve"
        " CPU/NPU hızlandırması kullanır.\n\n"
        "Bu projede yerel depolama için sunucusuz SQLite veritabanı"
        " kullanılmaktadır."
    )
    assistant.ingest_document("ornek_not.txt", sample_doc)

  print("\n" + "=" * 45)
  print(" Yerel RAG Asistanı Hazır! (Çıkış için 'q')")
  print("=" * 45)

  while True:
    query = input("\nSorunuz: ").strip()
    if not query:
      continue
    if query.lower() in ["q", "exit", "quit"]:
      print("Kapatılıyor...")
      break
    print("[*] Yanıt aranıyor...")
    try:
      answer = assistant.answer_query(query)
      print(f"\nCevap:\n{answer}\n")
    except Exception as e:
      print(f"[!] Hata oluştu: {e}")


if __name__ == "__main__":
  main()