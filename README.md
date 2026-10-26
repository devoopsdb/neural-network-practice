# Практическая работа: основы нейронных сетей

Три части в одном репозитории: перцептрон на ООП, CLI для подготовки данных
и параллельная аугментация. Все части самостоятельные, но используют общий
каталог `data/` и общий файл зависимостей.

## Структура

```
neural_network_practice/
├── part1_mlp/          # многослойный перцептрон и консольное меню
├── part2_prepare/      # CLI prepare/doctor, сборка массива из файлов
├── part3_parallel/     # producer-consumer, пул процессов, замеры
├── data/input/         # исходные данные (CSV, изображения)
├── data/output/        # веса, массивы, отчёты
├── logs/               # журналы и графики
├── tools/              # генератор демонстрационных данных
├── requirements.txt
├── .env.example
└── .gitignore
```

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Подготовка данных

```bash
python tools/make_sample_data.py --kind all
python tools/make_sample_data.py --kind images --count 240 --side 160 --images-name aug_images
```

Первая команда создаёт CSV с заявками, примеры для предсказания, таблицы и
48 картинок 64x64, вторая - 240 картинок 160x160 для замеров производительности.

## Запуск частей

```bash
# часть 1
cd part1_mlp && python -m app.main

# часть 2
cd part2_prepare && python -m app.cli prepare --path ../data/input/images --ext .png --output images_dataset.npy
cd part2_prepare && python -m app.cli doctor

# часть 3
cd part3_parallel && python -m app.cli --path ../data/input/aug_images --ext .png --workers 4 --processes 4
```

## Настройки

Все пути и параметры берутся из переменных окружения и аргументов командной
строки, значения по умолчанию совпадают с примером в `.env.example`.
Захардкоженных абсолютных путей в коде нет.

## Результаты

| Артефакт | Где лежит |
|----------|-----------|
| Веса обученной сети | `data/output/part1_weights.json` |
| График ошибки | `logs/part1_loss.png` |
| Массивы после `prepare` | `data/output/*.npy` и `*.json` |
| Отчёт о производительности | `data/output/performance_report.json` |
| Журналы | `logs/prepare.log`, `logs/parallel.log` |
