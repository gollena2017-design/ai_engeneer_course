"""
Moderator Graph — система модерації коментарів для блогу за допомогою LangGraph.
Аналізує коментар, категоризує його та направляє відповідному обробнику або завершує роботу.
"""

import os
import sys
from typing import TypedDict
from dotenv import load_dotenv

# Завантажуємо .env файл
load_dotenv()

# Налаштовуємо API-ключ для сумісності з OpenAI SDK, який використовується LangChain
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    sys.exit("Помилка: OPENROUTER_API_KEY не знайдено у файлі .env")

os.environ["OPENAI_API_KEY"] = api_key

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# 1) Налаштування AgentState (TypedDict)
class AgentState(TypedDict):
    comment: str
    category: str
    final_response: str

# Створюємо модель через OpenRouter
model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-chat",
    temperature=0.0,  # 0.0 для високої точності класифікації
)

# 2) Вузол 1: Аналізатор
def analyze_comment(state: AgentState) -> dict:
    comment = state["comment"]
    prompt = (
        "You are an automated blog comment moderator.\n"
        "Analyze the following comment and classify it into exactly ONE category.\n"
        "Respond with only a single word: TOXIC, QUESTION, or NORMAL.\n\n"
        "Rules:\n"
        "- TOXIC: if the comment is offensive, rude, uses bad words, insults the author or others.\n"
        "- QUESTION: if the user asks a question seeking support, help, explanation or technical information.\n"
        "- NORMAL: if it is a polite feedback, expression of gratitude, or general neutral comment.\n\n"
        f"Comment: {comment}"
    )
    
    response = model.invoke([HumanMessage(content=prompt)])
    category = response.content.strip().upper()
    
    # Очищаємо відповідь про всяк випадок
    if "TOXIC" in category:
        resolved_category = "TOXIC"
    elif "QUESTION" in category:
        resolved_category = "QUESTION"
    else:
        resolved_category = "NORMAL"
        
    return {"category": resolved_category}

# 3) Умовне Ребро (Router)
def route_comment(state: AgentState) -> str:
    category = state.get("category", "NORMAL")
    if category == "TOXIC":
        return "toxic_handler"
    elif category == "QUESTION":
        return "qa_handler"
    else:
        return "end"

# 4) Вузли-обробники
def toxic_handler(state: AgentState) -> dict:
    return {"final_response": "Ваш коментар видалено через порушення правил."}

def qa_handler(state: AgentState) -> dict:
    comment = state["comment"]
    prompt = (
        "You are a helpful and polite blog manager assistant.\n"
        "Write a concise, friendly, and polite response to the following user question in Ukrainian. "
        "Keep it to 1-2 sentences.\n\n"
        f"Question: {comment}"
    )
    response = model.invoke([HumanMessage(content=prompt)])
    return {"final_response": response.content.strip()}

# Збірка графа
workflow = StateGraph(AgentState)

# Додаємо вузли
workflow.add_node("analyze_comment", analyze_comment)
workflow.add_node("toxic_handler", toxic_handler)
workflow.add_node("qa_handler", qa_handler)

# Задаємо точку входу
workflow.set_entry_point("analyze_comment")

# Додаємо умовні переходи
workflow.add_conditional_edges(
    "analyze_comment",
    route_comment,
    {
        "toxic_handler": "toxic_handler",
        "qa_handler": "qa_handler",
        "end": END
    }
)

# Додаємо прямі переходи з обробників до кінця графа
workflow.add_edge("toxic_handler", END)
workflow.add_edge("qa_handler", END)

# Компіляція графа
app = workflow.compile()

# 5) Запуск графа тричі з різними коментарями
def main():
    test_comments = [
        "Ця стаття повний відстій, автор дурень!",
        "А як встановити цю бібліотеку на Windows?",
        "Дякую, дуже цікаво!"
    ]
    
    print("=== ЗАПУСК СИСТЕМИ МОДЕРАЦІЇ КОМЕНТАРІВ ===")
    
    for idx, comment in enumerate(test_comments, 1):
        print(f"\n--- Тест №{idx} ---")
        print(f"Вхідний коментар: \"{comment}\"")
        
        # Початковий стан
        initial_state = {
            "comment": comment,
            "category": "",
            "final_response": ""
        }
        
        # Виконання графа
        final_state = app.invoke(initial_state)
        
        # 6) Виведення фінального стану в термінал
        print("Фінальний стан графа:")
        print(f"  - comment: {final_state.get('comment')}")
        print(f"  - category: {final_state.get('category')}")
        print(f"  - final_response: {final_state.get('final_response') or '(коментар схвалено, відповіді не потрібно)'}")

if __name__ == "__main__":
    main()
