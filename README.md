# Guarded Chatbot (Groq) — прототип для midterm

Мини веб-чат. Фронтенд открывается в браузере.
Пайплайн: **guard (вход) → целевая модель → guard (выход)**.

### Через Groq (`PROVIDER=groq` в `.env`)
`gpt-oss-safeguard-20b` → `gpt-oss-20b` → `gpt-oss-safeguard-20b`.
Llama-2 и Llama Guard на Groq нет, поэтому guard получает таксономию Llama Guard
(S1–S14) в system-промпте и отвечает в том же формате `safe` / `unsafe\nS<номер>`.

### Локально (Ollama, без `PROVIDER` в `.env`)
Llama Guard 3 → Llama-2-7b-chat → Llama Guard 3 (как в статье CoP).
```bash
ollama pull llama2:7b
ollama pull llama-guard3:8b
python app.py
```
Модели можно поменять в `.env`: `BASE_MODEL=...`, `GUARD_MODEL=...`
(например, `GUARD_MODEL=llama-guard3:1b`, если не хватает памяти).


Это прототип для проекта по статье **CoP** (AI red-teaming). Галочка «Включить
Llama-Guard» переключает защиту — это основа будущего эксперимента
(сравнение Attack Success Rate: без защиты vs с защитой).

## Структура

```
groq_chat/
├── app.py              # бэкенд: Flask + Groq + Llama-Guard
├── templates/
│   └── index.html      # интерфейс чата
├── requirements.txt    # зависимости
├── .env.example        # шаблон для API-ключа
└── README.md
```

## Запуск в VS Code — по шагам

### 1. Получить ключ Groq
1. Зайди на https://console.groq.com (можно через Google).
2. Раздел **API Keys** → **Create API Key** → скопируй ключ (начинается с `gsk_`).

### 2. Открыть проект
- VS Code → File → Open Folder → выбери папку `groq_chat`.

### 3. Вставить ключ
- Переименуй `.env.example` в `.env`.
- Внутри впиши свой ключ:
  ```
  GROQ_API_KEY=gsk_твой_настоящий_ключ
  ```

### 4. Создать виртуальное окружение и поставить зависимости
Открой терминал в VS Code (Terminal → New Terminal) и выполни:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 5. Запустить
```bash
python app.py
```
Увидишь строку вроде `Running on http://127.0.0.1:5000`.
Открой этот адрес в браузере — там твой чат.

## Что проверить
- Напиши «Привет» — бот ответит через Groq.
- Сними галочку Llama-Guard и напиши снова — увидишь, что фильтр выключен (stage: no_guard).
- Под каждым ответом видно `stage` и вердикт guard — это пригодится для эксперимента.

## Что дальше (эксперимент)
Следующий шаг — скрипт, который прогоняет список тестовых промптов дважды
(`use_guard=False` и `use_guard=True`), считает Attack Success Rate и строит
таблицу baseline vs protected — как Table 2 в статье CoP.
