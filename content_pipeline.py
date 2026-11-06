"""
Content Pipeline — скрипт, який демонструє об'єднання кількох ланцюжків (Chains)
в єдиний конвеєр (Pipeline) за допомогою LangChain Expression Language (LCEL).

Конвеєр складається з:
1. Ланцюжка 1 (Генератор ідей): генерує яскравий заголовок на тему {topic}.
2. Ланцюжка 2 (Письменник): пише короткий абзац-вступ для отриманого заголовка {title}.
"""

import os
import sys
from dotenv import load_dotenv

# Завантажуємо змінні середовища з файлу .env
load_dotenv()

# Налаштовуємо API-ключ для сумісності з langchain-openai
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    sys.exit("Помилка: OPENROUTER_API_KEY не знайдено у файлі .env")

# LangChain OpenAI під капотом шукає змінну OPENAI_API_KEY, тому задаємо її
os.environ["OPENAI_API_KEY"] = api_key

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def main():
    print("=== Ініціалізація моделей та конвеєра LangChain ===")
    
    # Створюємо базову модель через OpenRouter
    model = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        model="deepseek/deepseek-chat",
        temperature=0.7,
    )

    # 2) Ланцюжок 1: Генератор ідей
    prompt1 = ChatPromptTemplate.from_messages([
        ("system", "You are a creative editor. Generate exactly one catchy and inspiring article headline on the given topic. Do not wrap it in quotes, write just the headline text in Ukrainian."),
        ("user", "Тема для заголовка: {topic}")
    ])
    
    chain1 = prompt1 | model | StrOutputParser()

    # 3) Ланцюжок 2: Письменник
    prompt2 = ChatPromptTemplate.from_messages([
        ("system", "You are a professional copywriter. Write a short, engaging introduction paragraph (3-4 sentences) for an article with the given headline. Write in Ukrainian."),
        ("user", "Заголовок статті: {title}")
    ])
    
    chain2 = prompt2 | model | StrOutputParser()

    # 4) Магія LCEL: Об'єднання ланцюжків в один загальний конвеєр!
    # Результат chain1 автоматично передається як значення для змінної {title} у chain2
    overall_chain = {"title": chain1} | chain2

    topic_to_generate = "Майбутнє космічного туризму"
    print(f"\n[Запуск конвеєра для теми]: \"{topic_to_generate}\"")
    print("Генерація контенту...")

    # 5) Запуск спільного конвеєра
    try:
        final_result = overall_chain.invoke({"topic": topic_to_generate})
        
        # 6) Виведення фінального тексту
        print("\n=== Фінальний згенерований вступ ===")
        print(final_result)
        
    except Exception as exc:
        print(f"\n[Помилка під час роботи конвеєра]: {exc}")

if __name__ == "__main__":
    main()
