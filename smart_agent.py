"""
Smart Agent — скрипт, який демонструє роботу з інструментами (Tool Calling)
через OpenRouter та OpenAI SDK. Реалізовано виклик функції check_inventory,
опис схеми у форматі JSON-схеми із параметрами strict та additionalProperties,
а також потоковий вивід (Stream) фінальної відповіді.
"""

import os
import sys
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    sys.exit("Помилка: OPENROUTER_API_KEY не знайдено в .env")

# Ініціалізація клієнта
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

# 2) Функція перевірки залишку товару
def check_inventory(item_name: str) -> str:
    inventory = {"iphone 15": 24, "macbook air": 5, "airpods": 0}
    return f"Залишок товару {item_name}: {inventory.get(item_name.lower(), 'Не знайдено')}"

# 3) Опис функції у форматі JSON-схеми
tools = [
    {
        "type": "function",
        "function": {
            "name": "check_inventory",
            "description": "Перевіряє наявність та точну кількість залишку товару на складі за його назвою.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "Назва товару для перевірки (наприклад: 'macbook air', 'airpods', 'iphone 15')."
                    }
                },
                "required": ["item_name"],
                "additionalProperties": False  # забороняє створювати нові поля
            },
            "strict": True  # заставляє створювати JSON точно за форматом
        }
    }
]

def main():
    user_prompt = "Привіт! Чи є у вас в наявності macbook air та airpods?"
    
    messages = [
        {"role": "user", "content": user_prompt}
    ]
    
    print(f"Користувач: {user_prompt}\n")
    print("Надсилаємо запит до моделі...")
    
    # 4) Перший запит до моделі
    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=messages,
            tools=tools,
        )
    except Exception as exc:
        print(f"Помилка першого запиту: {exc}")
        return

    assistant_message = response.choices[0].message
    
    # Додаємо повідомлення асистента до історії
    messages.append(assistant_message)
    
    # Перевіряємо, чи повернула модель запит на виклик інструменту (tool_calls)
    if assistant_message.tool_calls:
        print("\n[Модель запросила виклик інструментів]:")
        
        for tool_call in assistant_message.tool_calls:
            print(f"- Виклик функції '{tool_call.function.name}' з аргументами: {tool_call.function.arguments}")
            
            if tool_call.function.name == "check_inventory":
                # Парсимо аргументи
                try:
                    args = json.loads(tool_call.function.arguments)
                    item_name = args.get("item_name", "")
                except json.JSONDecodeError:
                    item_name = ""
                    print("  [Помилка] Не вдалося розпарсити JSON-аргументи.")
                
                # Програмно викликаємо mock-функцію
                result = check_inventory(item_name)
                print(f"  Результат функції: \"{result}\"")
                
                # Додаємо результати у повідомлення з роллю tool
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": result
                })
        
        print("\n[Надсилаємо фінальний запит для генерації відповіді користувачу (Stream)]...")
        
        # Відправляємо фінальний запит із stream=True
        try:
            final_stream = client.chat.completions.create(
                model="deepseek/deepseek-chat",
                messages=messages,
                stream=True
            )
            
            print("Асистент (Stream): ", end="", flush=True)
            for chunk in final_stream:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
            print()
            
        except Exception as exc:
            print(f"Помилка фінального запиту (Stream): {exc}")
            
    else:
        # Якщо модель відповіла відразу без виклику функцій
        print(f"Асистент: {assistant_message.content}")

if __name__ == "__main__":
    main()
