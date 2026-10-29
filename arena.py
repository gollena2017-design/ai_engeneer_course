"""
AI Arena — порівняння двох моделей через OpenRouter.

Один промпт вводиться з терміналу, робляться два запити:
  • DeepSeek  — temperature 0.0 (детерміновано, мінімум "фантазії")
  • Llama     — temperature 0.9 (висока варіативність)

Спостереження (за результатами запуску):
  1. DeepSeek при temperature=0.0 відповідає стабільно й коротко: повторні
     запуски з тим самим промптом дають майже ідентичний текст.
  2. Llama при temperature=0.9 частіше змінює формулювання, структуру та
     додає більше прикладів/деталей; інколи відповідь менш точна.
  3. Для фактичних/математичних задач краще низька temperature, для
     креативних — висока.
"""

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("OPENROUTER_API_KEY")
URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = {
    "DeepSeek (temp 0.0)": {"model": "deepseek/deepseek-chat", "temperature": 0.0},
    "Llama (temp 0.9)": {"model": "meta-llama/llama-3.3-70b-instruct", "temperature": 0.9},
}


def ask(model: str, temperature: float, prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = requests.post(URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def main() -> None:
    if not API_KEY:
        sys.exit("Помилка: OPENROUTER_API_KEY не знайдено у .env")

    prompt = input("Введіть промпт: ").strip()
    if not prompt:
        sys.exit("Порожній промпт.")

    for label, cfg in MODELS.items():
        print(f"\n=== {label} ({cfg['model']}) ===")
        try:
            print(ask(cfg["model"], cfg["temperature"], prompt))
        except requests.HTTPError as exc:
            print(f"Помилка запиту: {exc}\n{exc.response.text}")
        except Exception as exc:  # noqa: BLE001
            print(f"Помилка: {exc}")


if __name__ == "__main__":
    main()
