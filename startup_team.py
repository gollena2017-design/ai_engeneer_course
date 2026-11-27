"""
Startup Team — багатоагентна система із Супервізором (Supervisor Pattern)
на базі LangGraph та OpenRouter.

Команда складається з:
1. Software Architect (Архітектор): планує структуру та логіку.
2. Senior Python Developer (Кодер): пише код за планом архітектора.
3. Supervisor (Супервізор/Менеджер): керує чергою кроків (Architect -> Coder -> FINISH).
"""

import os
import sys
import operator
from typing import Annotated, List, Literal, TypedDict
from dotenv import load_dotenv

# Завантажуємо змінні середовища з .env
load_dotenv()

# Налаштовуємо API-ключ для сумісності з OpenAI SDK
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    sys.exit("Помилка: OPENROUTER_API_KEY не знайдено у .env")

os.environ["OPENAI_API_KEY"] = api_key

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# 1) Налаштування стану команди (State)
class TeamState(TypedDict):
    # Нові повідомлення автоматично додаються до списку
    messages: Annotated[List[BaseMessage], operator.add]
    next: str


# Визначаємо структуру відповіді Супервізора для моделі
class RouteResponse(BaseModel):
    next: Literal["Architect", "Coder", "FINISH"] = Field(
        description="Хто має виконувати наступний крок: Architect, Coder або FINISH."
    )


# Ініціалізуємо базову LLM через OpenRouter
model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-chat",
    temperature=0.1, # Дуже низька температура для стабільного планування та кодування
)

# Створюємо супервізор-модель із структурованим виводом
supervisor_model = model.with_structured_output(RouteResponse)


# 2) Агент 1: Software Architect (Системний промпт)
ARCHITECT_SYSTEM_PROMPT = (
    "Ти Software Architect. Твоя задача - прочитати бізнес-ідею і скласти чітку технічну структуру: "
    "які класи та функції треба написати і яку логіку використати. "
    "Увага: ТОБІ СУВОРО ЗАБОРОНЕНО ПИСАТИ БУДЬ-ЯКИЙ КОД НА PYTHON (включаючи приклади коду, класи або функції). "
    "Опиши архітектуру лише текстом, списками та специфікаціями алгоритмів!"
)

def architect_node(state: TeamState) -> dict:
    print("\n[Нода: Architect] Складаю архітектурний план...")
    messages = [SystemMessage(content=ARCHITECT_SYSTEM_PROMPT)] + state["messages"]
    response = model.invoke(messages)
    content = f"[Software Architect Plan]:\n{response.content}"
    return {
        "messages": [AIMessage(content=content, name="Architect")]
    }


# 3) Агент 2: Coder (Системний промпт)
CODER_SYSTEM_PROMPT = (
    "Ти Senior Python Developer. Твоя задача - написати ПОВНИЙ, готовий та працездатний Python код "
    "виключно на основі архітектурного плану від Архітектора. Тобі потрібно повністю "
    "реалізувати всі класи, функції та логіку, які описав Архітектор. Твоя відповідь має містити "
    "повністю готовий код гри, готовий до запуску в терміналі. Пиши чистий код із коментарями."
)

def coder_node(state: TeamState) -> dict:
    print("\n[Нода: Coder] Пишу код за архітектурним планом...")
    messages = [SystemMessage(content=CODER_SYSTEM_PROMPT)] + state["messages"]
    response = model.invoke(messages)
    content = f"[Senior Python Developer Code]:\n{response.content}"
    return {
        "messages": [AIMessage(content=content, name="Coder")]
    }


# 4) Супервізор
SUPERVISOR_PROMPT = (
    "Ти — менеджер проектів (Супервізор) у стартап-команді.\n"
    "Твоя роль — аналізувати поточний стан проекту та вирішувати, кому передати слово наступним:\n"
    "- 'Architect': якщо у розмові є лише запит користувача, але технічний план від '[Software Architect Plan]' ще не створений. Спочатку завжди викликай Architect!\n"
    "- 'Coder': якщо технічний план від '[Software Architect Plan]' готовий, але готовий Python код від '[Senior Python Developer Code]' ще не написаний. Coder має написати код.\n"
    "- 'FINISH': якщо у нас вже є і технічний план від '[Software Architect Plan]', і готовий Python код від '[Senior Python Developer Code]'. Задача виконана!\n\n"
    "Важливо: якщо в історії повідомлень вже є код від '[Senior Python Developer Code]', ти обов'язково ПОВИНЕН обрати 'FINISH'. Не викликай Coder повторно!"
)

def supervisor_node(state: TeamState) -> dict:
    print("\n[Нода: Supervisor] Аналізую розмову і приймаю рішення щодо наступного кроку...")
    
    # Додатковий програмний захист від нескінченних циклів
    has_architect = any("[Software Architect Plan]" in msg.content for msg in state["messages"])
    has_coder = any("[Senior Python Developer Code]" in msg.content for msg in state["messages"])
    
    if has_architect and has_coder:
        print("-> [Захист від зациклення] Виявлено і план, і код. Завершуємо роботу.")
        return {"next": "FINISH"}
        
    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    response = supervisor_model.invoke(messages)
    print(f"-> Рішення супервізора: {response.next}")
    return {"next": response.next}


# 5) Збірка графа (StateGraph)
workflow = StateGraph(TeamState)

# Додаємо ноди
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Architect", architect_node)
workflow.add_node("Coder", coder_node)

# Налаштовуємо точку входу
workflow.set_entry_point("Supervisor")

# Прямі переходи назад до Супервізора
workflow.add_edge("Architect", "Supervisor")
workflow.add_edge("Coder", "Supervisor")

# Функція умовного переходу від Супервізора
def route_next(state: TeamState) -> str:
    next_step = state.get("next", "FINISH")
    if next_step == "FINISH":
        return "FINISH"  # Повертаємо назву ключа з мапінгу ("FINISH"), а не END (який є "__end__")
    return next_step

workflow.add_conditional_edges(
    "Supervisor",
    route_next,
    {
        "Architect": "Architect",
        "Coder": "Coder",
        "FINISH": END
    }
)

# ... компілюємо граф ...
app = workflow.compile()


# 6) Тестування
def main():
    user_goal = "Я хочу консольну гру 'Хрестики-нулики' на Python."
    
    print("=== ЗАПУСК СТАРТАП-КОМАНДИ ===")
    print(f"Мета проекту: \"{user_goal}\"")
    
    # Початковий стан із запитом користувача
    initial_state = {
        "messages": [HumanMessage(content=user_goal)],
        "next": ""
    }
    
    # Виконуємо граф
    try:
        final_state = app.invoke(initial_state)
    except Exception as exc:
        print(f"Помилка виконання графа: {exc}")
        return

    # 7) Виводимо фінальний результат у консоль
    print("\n" + "=" * 60)
    print("=== ФІНАЛЬНИЙ РЕЗУЛЬТАТ РОБОТИ СТАРТАП-КОМАНДИ ===")
    print("=" * 60)
    
    for message in final_state["messages"]:
        sender = message.name if hasattr(message, "name") and message.name else "Користувач"
        # Визначаємо відправника по префіксу вмісту
        if "[Software Architect Plan]" in message.content:
            sender = "Software Architect"
        elif "[Senior Python Developer Code]" in message.content:
            sender = "Senior Python Developer"
            
        print(f"\n[{sender}]:")
        print("-" * 60)
        print(message.content)
        print("-" * 60)

if __name__ == "__main__":
    main()
