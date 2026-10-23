"""
Прогон 100 вредных промптов JailbreakBench (JBB-Behaviors) через пайплайн
guard -> целевая модель -> guard (один раз, guard включён).
Для каждого промпта сохраняется вердикт guard и категория S1-S14.

Запуск: python run_jbb.py
Результат: results/jbb_guarded.csv + сводка в терминале.
"""

import csv
import os
import time
from collections import Counter

from datasets import load_dataset

from app import BASE_MODEL, GUARD_MODEL, PROVIDER, guarded_chat

# Названия категорий — как в SAFEGUARD_POLICY / Llama Guard 3
S_NAMES = {
    "S1": "Violent Crimes", "S2": "Non-Violent Crimes", "S3": "Sex-Related Crimes",
    "S4": "Child Sexual Exploitation", "S5": "Defamation", "S6": "Specialized Advice",
    "S7": "Privacy", "S8": "Intellectual Property", "S9": "Indiscriminate Weapons",
    "S10": "Hate", "S11": "Suicide & Self-Harm", "S12": "Sexual Content",
    "S13": "Elections", "S14": "Code Interpreter Abuse",
}
OUT_PATH = "results/jbb_guarded.csv"


def parse_categories(guard_raw: str | None) -> list[str]:
    """'unsafe\\nS1,S9' -> ['S1', 'S9']; 'safe' -> []."""
    if not guard_raw or not guard_raw.lower().startswith("unsafe"):
        return []
    lines = guard_raw.split("\n", 1)
    if len(lines) < 2:
        return []
    return [c.strip().upper() for c in lines[1].replace(" ", ",").split(",") if c.strip()]


def call_with_retry(goal: str, attempts: int = 6):
    """Groq free tier ~30 запросов/мин — при 429 или сбое сети ждём и повторяем."""
    for i in range(attempts):
        try:
            return guarded_chat(goal, use_guard=True)
        except Exception as e:
            msg = str(e).lower()
            retryable = any(s in msg for s in ("429", "rate", "connection", "timed out"))
            if not retryable or i == attempts - 1:
                raise
            wait = 10 * (i + 1)
            print(f"   {e} — жду {wait} с…")
            time.sleep(wait)


def main():
    jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    harmful = jbb["harmful"]  # 100 вредных промптов (есть ещё jbb["benign"])
    print(f"{PROVIDER}: guard={GUARD_MODEL}, target={BASE_MODEL}, промптов: {len(harmful)}\n")

    # Уже готовые строки (не error) из прошлого запуска не отправляем повторно
    done = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, encoding="utf-8") as f:
            done = {r["index"]: r for r in csv.DictReader(f) if r["stage"] != "error"}
        print(f"Из прошлого запуска уже готово: {len(done)}, догоняю остальные\n")

    os.makedirs("results", exist_ok=True)
    rows = []
    for i, item in enumerate(harmful, 1):
        goal = item["Goal"]
        if str(item["Index"]) in done:
            rows.append(done[str(item["Index"])])
            continue
        try:
            reply, meta = call_with_retry(goal)
        except Exception as e:
            reply, meta = f"[Ошибка API: {e}]", {"stage": "error", "guard": None}
        cats = parse_categories(meta["guard"])
        rows.append({
            "index": item["Index"],
            "jbb_category": item["Category"],
            "goal": goal,
            "stage": meta["stage"],
            "guard_raw": (meta["guard"] or "").replace("\n", " "),
            "s_categories": ",".join(cats),
            "reply": reply,
        })
        print(f"[{i:3}/{len(harmful)}] {meta['stage']:15} {','.join(cats) or '-':8} {goal[:60]}")

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # --- Сводка ---
    stages = Counter(r["stage"] for r in rows)
    s_counts = Counter(c for r in rows for c in r["s_categories"].split(",") if c)
    n = len(rows)
    print("\n=== Итог ===")
    for stage, cnt in stages.most_common():
        print(f"{stage:15} {cnt:3}  ({cnt / n:.0%})")
    print("\n=== Категории S1-S14 (по заблокированным) ===")
    for s in sorted(s_counts, key=lambda s: int(s[1:]) if s[1:].isdigit() else 99):
        print(f"{s:4} {S_NAMES.get(s, '?'):26} {s_counts[s]:3}")
    print(f"\nСохранено: {OUT_PATH}")


if __name__ == "__main__":
    main()
