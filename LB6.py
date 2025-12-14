def read_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            value = float(line.strip().split(';')[1])
            data.append(value)
    return data


def holt_winters(data, season_length, alpha1, alpha2, alpha3):
    level = data[0]
    trend = data[1] - data[0]

    seasonals = [0] * season_length
    for i in range(season_length):
        seasonals[i] = data[i] - level

    for t in range(len(data)):
        if t >= season_length:
            last_level = level
            level = alpha1 * (data[t] - seasonals[t % season_length]) + (1 - alpha1) * (level + trend)
            trend = alpha2 * (level - last_level) + (1 - alpha2) * trend
            seasonals[t % season_length] = alpha3 * (data[t] - level) + (1 - alpha3) * seasonals[t % season_length]

    return level, trend, seasonals


def forecast(level, trend, seasonals, season_length, periods):
    result = []
    for i in range(1, periods + 1):
        value = level + i * trend + seasonals[(i - 1) % season_length]
        result.append(value)
    return result


# === ЗАГРУЗКА ДАННЫХ ===
data = read_data("LAB6.csv")

season_length = 12  # месяцы

alpha1 = 0.4
alpha2 = 0.3
alpha3 = 0.3

level, trend, seasonals = holt_winters(data, season_length, alpha1, alpha2, alpha3)

forecast_periods = 2 if len(data) >= 7 else 1
prediction = forecast(level, trend, seasonals, season_length, forecast_periods)

print("Параметры модели Хольта–Уинтерса:")
print(f"Уровень (a): {level:.2f}")
print(f"Тренд (b): {trend:.2f}")

print("\nСезонные коэффициенты:")
for i, s in enumerate(seasonals, 1):
    print(f"Месяц {i}: {s:.2f}")

print("\nПрогноз:")
for i, p in enumerate(prediction, 1):
    print(f"Прогноз на период {i}: {p:.2f}")
