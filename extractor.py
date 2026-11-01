"""
Extractor — симуляція обробки повідомлення служби підтримки через OpenRouter та OpenAI SDK.
Використовує System Prompt та JSON Mode для структурування даних.
"""

import os
import sys
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    sys.exit("Помилка: OPENROUTER_API_KEY не знайдено у .env")

# Ініціалізація клієнта OpenAI для роботи з OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

# Симуляція повідомлення від клієнта за замовчуванням
DEFAULT_MESSAGE = (
    "Добрий день! Я вчора отримала своє замовлення #40291 (шкіряний рюкзак), але "
    "колір взагалі не той, що на фото! Я замовляла темно-синій, а прийшов яскраво-блакитний. "
    "До того ж застібка заїдає. Дуже розчарована, це мав бути подарунок мамі на ювілей... "
    "Хочу зробити повернення коштів або обмін. Мій телефон: +380671112233. Олена."
)

SYSTEM_PROMPT = """
You are an expert customer support AI. Analyze the customer's message and extract the following structured details.
You must respond strictly with a single JSON object.

JSON Schema:
{
  "customer_name": string or null,
  "order_id": string or null,
  "category": string (e.g., "wrong_item", "damaged_item", "delivery_delay", "refund_request", "other"),
  "sentiment": string (e.g., "angry", "frustrated", "neutral", "happy"),
  "urgency": string (e.g., "low", "medium", "high"),
  "summary_ua": string (1-sentence summary of the problem in Ukrainian),
  "contact_phone": string or null,
  "requested_action": string (e.g., "refund", "replacement", "information", "other")
}
"""

def extract_support_info(customer_message: str) -> str:
    # Використовуємо модель deepseek/deepseek-chat та вмикаємо JSON mode
    response = client.chat.completions.create(
        model="deepseek/deepseek-chat",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": customer_message}
        ],
        temperature=0.0, # Мінімальна температура для точного видобутку інформації
    )
    return response.choices[0].message.content

def main():
    print("=== Симуляція обробки звернення клієнта ===")
    print(f"\n[Повідомлення за замовчуванням]:\n\"{DEFAULT_MESSAGE}\"")
    
    user_input = input("\nВведіть власне повідомлення або натисніть Enter, щоб використати за замовчуванням: ").strip()
    message_to_process = user_input if user_input else DEFAULT_MESSAGE

    print("\nАналізуємо повідомлення...")
    
    try:
        raw_result = extract_support_info(message_to_process)
        
        # Спроба розпарсити відповідь як JSON для перевірки валідності та красивого виведення
        parsed_json = json.loads(raw_result)
        
        print("\n=== Результат обробки (JSON) ===")
        # Виводимо у красивому відформатованому вигляді
        print(json.dumps(parsed_json, indent=4, ensure_ascii=False))
        
    except json.JSONDecodeError as exc:
        print(f"\n[Помилка]: Отримано невалідний JSON від моделі. {exc}")
        print("Сира відповідь моделі:")
        print(raw_result)
    except Exception as exc:
        print(f"\n[Помилка під час запиту]: {exc}")

if __name__ == "__main__":
    main()
