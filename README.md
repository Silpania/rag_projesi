# Local RAG Assistant (Foundry Local & SQLite)

Tamamen yerel cihazda (offline) çalışan, Microsoft Foundry Local modelleri ve SQLite tabanlı vektör arama mekanizmasını kullanan RAG (Retrieval-Augmented Generation) asistanı.

## Özellikler
- **Modeller:** `phi-3.5-mini` (Chat) ve `qwen3-embedding-0.6b` (Embedding)
- **Veritabanı:** SQLite (Vektörler JSON formatında saklanır)
- **Arama Yöntemi:** Kosinüs Benzerliği (Cosine Similarity - NumPy)
- **Gizlilik:** Veriler cihaz dışına çıkmaz, internet bağlantısı gerektirmez.

## Kurulum ve Çalıştırma

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install numpy foundry-local-sdk
