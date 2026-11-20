"""
Pro RAG — пайплайн із Step-back Prompting (переписування запиту) та Reranking.

Схема:
  user_question
      -> rewrite_chain (LLM переписує питання у пошуковий запит)
      -> compression_retriever (LanceDB retriever + FlashrankRerank)
      -> final LLM (відповідь лише на основі знайденого контексту)
"""

import os
import sys

import lancedb
from dotenv import load_dotenv

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    sys.exit("Помилка: OPENROUTER_API_KEY не знайдено у .env")

os.environ["OPENAI_API_KEY"] = API_KEY

DB_DIR = ".lancedb"
TABLE_NAME = "pro_rag_facts"

# 10-15 коротких фактів про історію Apple та Microsoft
FACTS = [
    "Apple була заснована 1 квітня 1976 року Стівом Джобсом, Стівом Возняком і Рональдом Вейном.",
    "Перший комп'ютер Apple — Apple I — був представлений у квітні 1976 року і продавався за 666,66 долара.",
    "Apple II, випущений у 1977 році, став одним із перших масових персональних комп'ютерів.",
    "Macintosh був випущений 24 січня 1984 року і став першим масовим комп'ютером із графічним інтерфейсом.",
    "iPhone був представлений 9 січня 2007 року.",
    "iPad був представлений 27 січня 2010 року.",
    "Стів Джобс залишив Apple у 1985 році і повернувся у 1997 році.",
    "Microsoft була заснована 4 квітня 1975 року Біллом Гейтсом і Полом Алленом.",
    "Перший продукт Microsoft — Altair BASIC — вийшов у 1975 році.",
    "MS-DOS була випущена у 1981 році для комп'ютерів IBM PC.",
    "Windows 1.0 була випущена 20 листопада 1985 року.",
    "Windows 95 була випущена 24 серпня 1995 року.",
    "Microsoft випустила перший пристрій Surface у 2012 році — свій перший власний планшет.",
    "Xbox була випущена у 2001 році.",
    "Microsoft придбала LinkedIn у 2016 році та Activision Blizzard у 2023 році.",
]


def build_vectorstore():
    embeddings = OllamaEmbeddings(model="bge-m3")
    documents = [
        Document(page_content=fact, metadata={"source": "facts", "id": i})
        for i, fact in enumerate(FACTS, 1)
    ]

    db = lancedb.connect(DB_DIR)
    try:
        if TABLE_NAME in db.table_names():
            db.drop_table(TABLE_NAME)
    except Exception:
        pass

    vectorstore = LanceDB.from_documents(
        documents,
        embeddings,
        connection=db,
        table_name=TABLE_NAME,
    )
    return vectorstore


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        model="deepseek/deepseek-chat",
        temperature=0.0,
    )

    vectorstore = build_vectorstore()
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    # 4) Reranking: FlashrankRerank як компресор для базового retriever
    compressor = FlashrankRerank(top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

    # 3) Step-back Prompting: власний LCEL-ланцюжок переписування запиту
    rewrite_prompt = ChatPromptTemplate.from_template(
        "Користувач задав питання: {question}. "
        "Перепиши його так, щоб воно звучало як ідеальний пошуковий запит для енциклопедії. "
        "Поверни ТІЛЬКИ текст нового запиту."
    )
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    # Фінальна відповідь лише з контексту
    final_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Ти відповідаєш на питання, використовуючи ВИКЛЮЧНО наданий контекст. "
                "Якщо у контексті немає відповіді, скажи: 'Я не маю достатньо інформації.' "
                "Відповідай українською, коротко і по суті.",
            ),
            ("human", "Контекст:\n{context}\n\nПитання: {question}"),
        ]
    )

    # 5) Загальний пайплайн:
    # user_question -> rewrite_chain -> compression_retriever -> final LLM
    overall_chain = (
        RunnableParallel(
            context=rewrite_chain | compression_retriever | format_docs,
            question=RunnablePassthrough(),
        )
        | final_prompt
        | llm
        | StrOutputParser()
    )

    user_question = "Коли вони випустили свій перший комп'ютер?"

    # Показуємо проміжний переписаний запит для наочності
    rewritten = rewrite_chain.invoke(user_question).strip()
    print(f"Оригінальне питання: {user_question}")
    print(f"Переписаний запит: {rewritten}\n")

    docs = compression_retriever.invoke(rewritten)
    print("Знайдений контекст після реранкінгу:")
    for i, doc in enumerate(docs, 1):
        print(f"  {i}. {doc.page_content}")

    answer = overall_chain.invoke(user_question)
    print(f"\nВідповідь:\n{answer}")


if __name__ == "__main__":
    main()
