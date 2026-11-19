"""
FAQ Router RAG — бот підтримки, який відповідає ЛИШЕ на основі бази знань
faq_router.txt. Побудовано на LCEL: Retriever -> Prompt -> LLM -> Parser.
Векторна база: LanceDB + OllamaEmbeddings(model="bge-m3").
"""

import os
import sys

import lancedb
from dotenv import load_dotenv

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    sys.exit("Помилка: OPENROUTER_API_KEY не знайдено у .env")

# langchain-openai під капотом шукає OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = API_KEY

FAQ_FILE = "faq_router.txt"
DB_DIR = ".lancedb"
TABLE_NAME = "faq_router"

REFUSAL = "Вибачте, я можу допомогти лише з налаштуванням роутера."

# Дуже жорсткий System Prompt: відповідати ТІЛЬКИ з контексту
SYSTEM_PROMPT = (
    "Ти — бот служби підтримки, який відповідає ВИКЛЮЧНО на питання про "
    "налаштування роутера.\n\n"
    "ЖОРСТКІ ПРАВИЛА:\n"
    "1. Використовуй ТІЛЬКИ наданий Контекст. Заборонено використовувати будь-які "
    "знання поза Контекстом.\n"
    "2. Якщо у Контексті немає відповіді на питання, або питання не стосується "
    "налаштування роутера, відповідай рівно однією фразою: "
    f'"{REFUSAL}"\n'
    "3. Не вигадуй факти, кроки, адреси чи номери телефонів, яких немає в Контексті.\n"
    "4. Якщо питання містить кілька частин і всі вони є в Контексті — відповідай "
    "на кожну частину окремо.\n"
    "5. Відповідай українською мовою, коротко і по суті."
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_vectorstore(embeddings):
    """Читає FAQ, ріже на чанки й завантажує у LanceDB."""
    if not os.path.exists(FAQ_FILE):
        sys.exit(f"Помилка: файл {FAQ_FILE} не знайдено.")

    with open(FAQ_FILE, "r", encoding="utf-8") as fh:
        faq_text = fh.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(faq_text)
    documents = [
        Document(page_content=chunk, metadata={"source": FAQ_FILE})
        for chunk in chunks
    ]

    db = lancedb.connect(DB_DIR)
    # Перебудовуємо таблицю з нуля, щоб не дублювати чанки
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
    print(f"Завантажено {len(documents)} чанків у LanceDB (таблиця '{TABLE_NAME}').")
    return vectorstore


def build_rag_chain(vectorstore):
    """LCEL-ланцюжок: Retriever -> Prompt -> LLM -> Parser."""
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        model="deepseek/deepseek-chat",
        temperature=0.0,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Контекст:\n{context}\n\nПитання: {question}"),
        ]
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def main():
    print("Ініціалізація OllamaEmbeddings(model='bge-m3')...")
    embeddings = OllamaEmbeddings(model="bge-m3")

    vectorstore = build_vectorstore(embeddings)
    rag_chain = build_rag_chain(vectorstore)

    print("\n=== FAQ ROUTER BOT ГОТОВИЙ ===")
    print("Задайте питання про налаштування роутера (або 'exit' для виходу).")

    while True:
        try:
            question = input("\nВи: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nВихід...")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "вихід"}:
            print("Бот: До побачення!")
            break

        try:
            answer = rag_chain.invoke(question).strip()
        except Exception as exc:
            print(f"Помилка запиту: {exc}")
            continue

        print(f"Бот: {answer}")


if __name__ == "__main__":
    main()


# ============================================================================
# ТЕСТУВАННЯ — вивід терміналу (3 запити)
# ============================================================================
# Ініціалізація OllamaEmbeddings(model='bge-m3')...
# Завантажено 12 чанків у LanceDB (таблиця 'faq_router').
#
# === FAQ ROUTER BOT ГОТОВИЙ ===
# Задайте питання про налаштування роутера (або 'exit' для виходу).
#
# Ви: Що робити, якщо горить червоний?
# Бот: Якщо на роутері горить червоний індикатор, виконайте наступні кроки:
# 1. Перевірте, чи щільно вставлений кабель провайдера в порт WAN на задній панелі роутера.
# 2. Переконайтеся, що тариф оплачений і немає аварій на боці провайдера.
# 3. Перезавантажте роутер: вимкніть живлення з розетки на 30 секунд і ввімкніть знову.
# 4. Зачекайте 2-3 хвилини, поки роутер повністю завантажиться.
# 5. Якщо червоний індикатор не зникає після перезавантаження, зателефонуйте до служби підтримки.
#
# Ви: Як скинути роутер, і який номер підтримки?
# Бот: Щоб скинути роутер до заводських налаштувань:
# 1. Знайдіть на задній панелі кнопку Reset (зазвичай це маленький отвір, підписаний Reset або WPS/Reset).
# 2. Використовуйте скріпку або голку, натисніть і утримуйте кнопку Reset протягом 10 секунд, поки всі індикатори не блимнуть.
# 3. Відпустіть кнопку і зачекайте 1-2 хвилини, поки роутер перезавантажиться.
# Номер підтримки: 0 800 123 456.
#
# Ви: Яка висота Евересту?
# Бот: Вибачте, я можу допомогти лише з налаштуванням роутера.
# ============================================================================
