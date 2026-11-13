"""
Code Explorer — інтерактивна система RAG для пошуку у коді за допомогою
OllamaEmbeddings (bge-m3), локальної бази LanceDB та LangChain.
"""

import os
import sys
import lancedb
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document

def main():
    # 1) Читаємо текст із файлів у папці my_project
    project_dir = "my_project"
    if not os.path.exists(project_dir):
        print(f"Помилка: Папка {project_dir} не існує. Створіть її та покладіть туди файли коду.")
        return

    documents = []
    
    # Шукаємо всі Python-файли у папці my_project
    for filename in os.listdir(project_dir):
        if filename.endswith(".py"):
            filepath = os.path.join(project_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    code_content = f.read()
                
                # 5) Створюємо документ та додаємо метадані з іменем файлу
                doc = Document(page_content=code_content, metadata={"source": filename})
                documents.append(doc)
            except Exception as exc:
                print(f"Помилка читання файлу {filename}: {exc}")

    if not documents:
        print("Не знайдено жодного Python-файлу в папці my_project.")
        return

    print(f"Знайдено {len(documents)} Python-файлів у папці '{project_dir}'.")

    # 3) Чанкінг за допомогою RecursiveCharacterTextSplitter для мови Python
    print("Розбиваємо код на чанки...")
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=400,
        chunk_overlap=50
    )
    
    chunks = splitter.split_documents(documents)
    print(f"Створено {len(chunks)} чанків коду.")

    # 4) Ембедінги та База
    print("Ініціалізуємо OllamaEmbeddings(model='bge-m3')...")
    embeddings = OllamaEmbeddings(model="bge-m3")

    print("Підключаємо локальну базу LanceDB (папка .lancedb)...")
    db = lancedb.connect(".lancedb")
    table_name = "code_chunks"

    # 5) Завантаження (Ingestion) через LangChain конектор
    print("Зберігаємо чанки коду у LanceDB...")
    vectorstore = LanceDB.from_documents(
        chunks,
        embeddings,
        connection=db,
        table_name=table_name
    )
    print("Дані успішно завантажено в базу!")

    # 6) Пошук (Retrieval)
    print("\n=== КОДОВИЙ ПРОВІДНИК ГОТОВИЙ ДО РОБОТИ ===")
    print("Введіть запит для пошуку в коді (наприклад, 'temperature' або 'datetime').")
    
    while True:
        try:
            query = input("\nЩо ви хочете знайти у коді? (або 'exit' для виходу): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nВихід...")
            break

        if not query:
            continue

        if query.lower() in ["exit", "вихід", "quit"]:
            print("Дякуємо за використання Кодового Провідника. Бувайте!")
            break

        print(f"Шукаємо: '{query}'...")
        
        # 7) Similarity search по базі (ТОП-2 релевантних шматки)
        try:
            results = vectorstore.similarity_search(query, k=2)
            
            if not results:
                print("Збігів не знайдено.")
                continue

            print(f"\nЗнайдено {len(results)} найрелевантніших збіги:")
            
            for i, doc in enumerate(results, 1):
                source_file = doc.metadata.get("source", "Невідомий файл")
                print(f"\n[{i}] Джерело: {source_file}")
                print("-" * 60)
                print(doc.page_content)
                print("-" * 60)
                
        except Exception as exc:
            print(f"Помилка пошуку: {exc}")

if __name__ == "__main__":
    main()
