# Naive qasper RAG

Проект для экспериментов с RAG по научным статьям. На данных [QASPER](https://huggingface.co/datasets/allenai/qasper) собран пайплайн от подготовки текстов до ответа локальной LLM и отдельную оценку качества поиска. В текущем сценарии используются готовые тексты статей из `validation`-части датасета.

## Что реализовано

- **Подготовка и индексация данных.** Разбор структуры статей QASPER, выделение непустых абзацев, сохранение идентификаторов статьи, раздела и абзаца. Индексация идёт пакетами.
- **Гибридный поиск.** FastEmbed строит плотные эмбеддинги (`BAAI/bge-small-en-v1.5`) и разреженные представления (`Qdrant/bm42-all-minilm-l6-v2-attentions`). Qdrant хранит оба вида векторов и объединяет выдачу методом Reciprocal Rank Fusion (RRF).
- **Повторное ранжирование и генерация.** Cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`) сортирует кандидатов по близости к вопросу. Лучшие фрагменты передаются локальной модели через Ollama; для контекста установлен лимит по длине. По запросу CLI показывает метаданные использованных источников.
- **Оценка retrieval.** Свидетельства из разметки QASPER сопоставляются с индексируемыми абзацами. Поиск при оценке ограничен конкретной статьёй; `ranx` считает `MRR@10`, `NDCG@10` и `Recall@10`. Результат сохраняется в JSON вместе с числом всех и оценённых вопросов.
- **Инженерная часть.** Компоненты эмбеддингов, хранилища, reranking и генерации разделены интерфейсами; есть CLI, модульные и интеграционные тесты, CI на GitHub Actions, зависимости зафиксированы через `uv.lock`.

## Запуск

Нужны Python 3.11, [uv](https://docs.astral.sh/uv/) и интернет для первой загрузки QASPER и моделей. Примеры ниже работают на CPU; в CLI по умолчанию выбран `cuda`.

```bash
uv sync --locked
uv run python -m src.evaluation.cli build-index --device cpu --limit 10
```

Команда загрузит первые 10 статей из `validation`-части QASPER и создаст локальную коллекцию `qasper_hybrid` в `.qdrant_db`. Без `--limit` индексируется вся эта часть датасета.

Чтобы получить ответ, запустите [Ollama](https://ollama.com/) и скачайте модель по умолчанию:

```bash
ollama pull qwen3:8b
ollama serve
```

Затем в другом терминале:

```bash
uv run python -m src.evaluation.cli query "What methods are compared in the paper?" --device cpu --show-sources
```

`query` ищет по всей коллекции и выводит ответ в терминал. `--show-sources` дополнительно показывает метаданные фрагментов, вошедших в контекст. Модель и адрес Ollama задаются через `--ollama-model` и `--ollama-host`.

Для оценки поиска сначала постройте индекс по тем же статьям, затем выполните:

```bash
uv run python -m src.evaluation.cli evaluate-retrieval --device cpu --limit 10
```

Метрики будут напечатаны в терминале и сохранены в `artifacts/qasper/retrieval_metrics.json`. В расчёт входят вопросы, для которых текст размеченного свидетельства удалось сопоставить с абзацем статьи. Это ограничение видно по полям `total_queries` и `evaluated_queries`.

### Метрики поиска

| Конфигурация | `MRR@10` | `NDCG@10` | `Recall@10` |
| --- | ---: | ---: | ---: |
| `top10` | 0.4313 | 0.4540 | 0.6932 |
| `top30` | 0.4313 | 0.4540 | 0.6932 |
| `reranked` | **0.5221** | **0.5351** | **0.7493** |


## Настройка и проверка

У всех команд есть параметры `--collection`, `--db-path`, `--dense-model`, `--sparse-model` и `--vector-size`. Если меняете модель эмбеддингов, создайте отдельную коллекцию и укажите размер её плотного вектора (`--vector-size`, по умолчанию 384). Параметры `--candidate-k` и `--top-k` управляют числом кандидатов до и после reranking. Полный список: `uv run python -m src.evaluation.cli --help` и `uv run python -m src.evaluation.cli <команда> --help`.

```bash
uv run pytest -m "not gpu" -q
```

Эту команду также выполняет CI. Для запуска с CUDA на Linux есть `scripts/init-gpu.sh`, который настраивает путь к библиотекам NVIDIA из `.venv` и запускает команду через `uv run --no-sync`:

```bash
bash scripts/init-gpu.sh python -m src.evaluation.cli build-index --limit 10
```

## Структура

| Каталог | Содержимое |
| --- | --- |
| `src/evaluation/` | Загрузка QASPER, индексация, CLI, поиск и метрики |
| `src/embedding/`, `src/reranking/` | Эмбеддинги и cross-encoder на FastEmbed |
| `src/database/` | Работа с локальным Qdrant |
| `src/models/` | Модели данных статьи и результата поиска |
| `tests/` | Тесты компонентов и локального Qdrant |

