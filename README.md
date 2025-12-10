# PolyRAG - Modüler RAG Framework

**PolyRAG**, esnek, modüler ve yüksek performanslı bir RAG (Retrieval-Augmented Generation) framework'üdür. Hexagonal Architecture (Ports and Adapters) prensiplerine göre tasarlanmış olup, farklı LLM, Vector Database ve Embedding sağlayıcıları arasında kolayca geçiş yapmanızı sağlar.

## 🚀 Özellikler

*   **Çoklu LLM Desteği**: Ollama, OpenAI, Claude, Gemini gibi popüler modellerle entegrasyon.
*   **Esnek Veritabanı**: Qdrant (Vektör) ve Neo4j (Graph) veritabanı desteği.
*   **Multimodal Yetenekler**: Metin, Görsel ve Tablo verileriyle çalışabilme.
*   **Gelişmiş Retrieval**: ColBERT, ColPali ve Hybrid arama teknikleri.
*   **Hexagonal Mimari**: Bağımlılıkları izole eden, test edilebilir ve sürdürülebilir kod yapısı.
*   **Streaming**: Token-by-token yanıt üretimi.

---

## 🛠️ Kurulum

PolyRAG'i kullanmaya başlamak için öncelikle Python 3.10 veya üzeri bir sürüme ihtiyacınız vardır.

1.  **Projeyi Klonlayın:**
    ```bash
    git clone https://github.com/polyrag/polyrag.git
    cd polyrag
    ```

2.  **Sanal Ortam Oluşturun (Önerilen):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

3.  **Bağımlılıkları Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Not: Geliştirme modunda kurulum için `pip install -e .` komutunu kullanabilirsiniz.)*

4.  **Gerekli Servisleri Ayağa Kaldırın:**
    Örnekleri çalıştırmak için Ollama ve Qdrant'ın yerel makinenizde çalışıyor olması gerekir.
    *   **Ollama:** [ollama.com](https://ollama.com) adresinden indirin.
    *   **Qdrant:** Docker ile hızlıca başlatın:
        ```bash
        docker run -p 6333:6333 qdrant/qdrant
        ```

---

## ⚡ Hızlı Başlangıç

Aşağıdaki örnek, basit bir metin belgesi üzerinden RAG akışının nasıl oluşturulacağını gösterir.

```python
from polyrag.interface.builder import PipelineBuilder
from polyrag.interface.factory import AdapterFactory

def main():
    # 1. Adapter'ları Oluşturun
    llm = AdapterFactory.create_llm("ollama", model="llama3.2")
    embedding = AdapterFactory.create_embedding("fastembed")
    vector_store = AdapterFactory.create_vector_store("qdrant")
    loader = AdapterFactory.create_document_loader("text")
    chunker = AdapterFactory.create_chunker("fixed_size", chunk_size=500, chunk_overlap=50)

    # 2. Pipeline'ı İnşa Edin
    pipeline = (
        PipelineBuilder()
        .with_llm(llm)
        .with_embedding(embedding)
        .with_vector_store(vector_store)
        .with_document_loader(loader)
        .with_chunker(chunker)
        .with_collection_name("my_rag_collection")
        .build()
    )

    # 3. Veri Yükleyin (Ingestion)
    # 'data.txt' adında bir dosyanız olduğunu varsayalım.
    pipeline.ingest("data.txt")

    # 4. Soru Sorun (Querying)
    question = "Bu belgenin ana fikri nedir?"
    print(f"Soru: {question}\nCevap:")
    
    for chunk in pipeline.query_stream(question):
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    main()
```

---

## 🏗️ Detaylı Mimari

PolyRAG, **Hexagonal Architecture** (Ports ve Adapters) yapısını benimser. Bu mimari, iş mantığını (Core Domain) dış dünyadan (Veritabanları, API'ler, Framework'ler) ayırır.

### Temel Katmanlar

1.  **Interface Layer (Arayüz Katmanı)**:
    *   Kullanıcının sistemle etkileşime girdiği yerdir.
    *   `PipelineBuilder`: Akışkan (fluent) bir arayüz ile pipeline oluşturmayı sağlar.
    *   `PolyRAGPipeline`: Ingestion ve Query süreçlerini yöneten ana orkestratördür.

2.  **Core Domain (Çekirdek Katman)**:
    *   Sistemin kalbidir. Hiçbir dış kütüphaneye bağımlı değildir.
    *   **Ports (Arayüzler)**: `LLMPort`, `VectorStorePort` gibi soyut sınıfları tanımlar. Adapter'lar bu portları implemente etmek zorundadır.
    *   **Models**: `Document`, `Chunk`, `RetrievalResult` gibi veri yapılarını içerir.

3.  **Adapters Layer (Adaptör Katmanı)**:
    *   Dış teknolojilerle Core katmanı arasındaki köprüdür.
    *   Örneğin: `OllamaAdapter`, `QdrantAdapter`, `FastEmbedAdapter`.
    *   Yeni bir teknoloji eklemek için sadece yeni bir adapter yazmak yeterlidir; Core kodunu değiştirmeye gerek yoktur.

### Mimari Şeması

```mermaid
graph TD
    User[Kullanıcı / Uygulama] --> Interface[Interface Layer\n(Pipeline, Builder)]
    
    subgraph Core Domain
        Ports[Ports\n(Abstract Interfaces)]
        Models[Domain Models\n(Document, Chunk)]
    end
    
    Interface --> Ports
    Interface --> Models
    
    subgraph Adapters Layer
        LLM[LLM Adapter\n(Ollama, OpenAPI)]
        VectorDB[Vector Store\n(Qdrant, Chroma)]
        Embed[Embedding\n(FastEmbed, OpenAI)]
        Loader[Doc Loader\n(PDF, Text)]
    end
    
    LLM -. implements .-> Ports
    VectorDB -. implements .-> Ports
    Embed -. implements .-> Ports
    Loader -. implements .-> Ports
```

### Akış Diyagramları

**Ingestion (Veri Yükleme) Akışı:**
`Dosya` -> `Document Loader` -> `Chunker` -> `Embedder` -> `Vector Store`

**Query (Sorgulama) Akışı:**
`Soru` -> `Embedder` -> `Retriever (Vector Store)` -> `Reranker (Opsiyonel)` -> `Context Builder` -> `LLM`

---

## 📂 Proje Yapısı

```
polyrag/
├── core/
│   ├── ports/           # Tüm soyut arayüzler (LLMPort, vb.)
│   ├── models/          # Veri sınıfları (Document, Chunk, vb.)
│   └── services/        # Temel servis mantıkları
├── adapters/
│   ├── llm/             # LLM implementasyonları
│   ├── embedding/       # Embedding model entegrasyonları
│   ├── vector_store/    # Vektör veritabanı sürücüleri
│   ├── document_loader/ # Dosya okuyucular
│   └── chunking/        # Metin parçalama algoritmaları
├── interface/
│   ├── pipeline.py      # Ana çalışma mantığı
│   └── builder.py       # Pipeline oluşturucu
└── examples/            # Örnek senaryolar
```
