"""
Research Agent — автономний ReAct-агент на базі LangGraph та OpenRouter,
який вміє шукати інформацію у Вікіпедії та записувати результати у файл.
"""

import os
import sys
from dotenv import load_dotenv

# Завантажуємо змінні середовища з .env
load_dotenv()

# Налаштовуємо API-ключ для langchain-openai
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    sys.exit("Помилка: OPENROUTER_API_KEY не знайдено у .env")

os.environ["OPENAI_API_KEY"] = api_key

import wikipedia
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


# 2) Інструмент 1: Пошук у Вікіпедії
@tool
def search_wikipedia(query: str) -> str:
    """Шукає у Вікіпедії інформацію за запитом та повертає коротке резюме (summary) статті."""
    try:
        # Спершу пробуємо шукати українською
        wikipedia.set_lang("uk")
        return wikipedia.summary(query, sentences=4)
    except wikipedia.exceptions.DisambiguationError as e:
        # Якщо знайдено кілька варіантів, беремо перший
        try:
            return wikipedia.summary(e.options[0], sentences=4)
        except Exception:
            return f"Знайдено кілька значень для '{query}'. Варіанти: {', '.join(e.options[:5])}"
    except wikipedia.exceptions.PageError:
        # Якщо українською не знайдено, шукаємо англійською
        try:
            wikipedia.set_lang("en")
            return wikipedia.summary(query, sentences=4)
        except Exception:
            return f"Статтю про '{query}' не знайдено у Вікіпедії."
    except Exception as exc:
        return f"Помилка пошуку у Вікіпедії: {exc}"


# 3) Інструмент 2: Запис у файл
@tool
def save_to_file(filename: str, content: str) -> str:
    """Створює текстовий файл із заданим іменем і записує туди вміст (content). Повертає 'Успішно збережено'."""
    try:
        # Безпечне ім'я файлу (без відносних шляхів на кшталт ../)
        safe_filename = os.path.basename(filename)
        with open(safe_filename, "w", encoding="utf-8") as fh:
            fh.write(content)
        return "Успішно збережено"
    except Exception as exc:
        return f"Помилка запису у файл: {exc}"


def main():
    print("=== Ініціалізація ReAct-агента ===")
    
    # 4) Ініціалізуємо LLM через OpenRouter
    model = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        model="deepseek/deepseek-chat",
        temperature=0.3, # Температура 0.3 для оптимальної творчості у написанні есе
    )

    # Список доступних інструментів
    tools = [search_wikipedia, save_to_file]

    # Створюємо ReAct-агента за допомогою LangGraph
    agent = create_react_agent(model, tools)

    # 5) Тестування автономності: Запит для агента
    query = (
        "Знайди у Вікіпедії інформацію про компанію SpaceX. "
        "Напиши коротке есе на основі знайденого (2 абзаци) і збережи його у файл 'spacex_report.txt'."
    )

    print(f"\n[Запит до агента]: \"{query}\"")
    print("\nЗапуск агента (спостерігайте за кроками Thought/Action у консолі)...")
    print("-" * 60)

    # Запускаємо агента
    try:
        response = agent.invoke({"messages": [("user", query)]})
        
        # 6) Виводимо лог спілкування та кроків у консоль
        for message in response["messages"]:
            message.pretty_print()
            
    except Exception as exc:
        print(f"Помилка під час роботи агента: {exc}")
        return

    print("-" * 60)
    
    # 7) Перевіряємо, чи з'явився файл spacex_report.txt
    report_file = "spacex_report.txt"
    if os.path.exists(report_file):
        print(f"\n[Успіх!] Файл '{report_file}' успішно створено у поточній папці.")
        try:
            with open(report_file, "r", encoding="utf-8") as fh:
                content = fh.read()
            print("\nВміст файлу 'spacex_report.txt':")
            print("=" * 60)
            print(content)
            print("=" * 60)
        except Exception as exc:
            print(f"Помилка читання створеного файлу: {exc}")
    else:
        print(f"\n[Помилка] Файл '{report_file}' не знайдено.")


if __name__ == "__main__":
    main()
