from datetime import datetime

time_str = input("Введіть час у форматі ГГ:ХХ (24-годинний формат): ")

try:
    t = datetime.strptime(time_str, "%H:%M").time()
except ValueError:
    print("Неправильний формат часу. Використовуйте ГГ:ХХ, наприклад 14:30.")
    raise SystemExit(1)

hour = t.hour

if 5 <= hour < 12:
    greeting = "Доброго ранку"
elif 12 <= hour < 17:
    greeting = "Доброго дня"
elif 17 <= hour < 22:
    greeting = "Доброго вечора"
else:
    greeting = "Доброї ночі"

print(f"{greeting}! ({t.strftime('%H:%M')})")