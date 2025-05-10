import os
import re
import time
import pdfplumber
import torch
import pandas as pd
import psycopg2
from tqdm.auto import tqdm
from torch import nn, optim
from transformers import AutoTokenizer, BertPreTrainedModel, BertModel


# Класс для мониторинга обучения
class TrainingMonitor:
    def __init__(self):
        self.progress_bar = None
        self.epoch_start_time = None
    def start_epoch(self, epoch, epochs, file):
        self.epoch_start_time=time.time()
        print(f"\n[Подготовка] Начало эпохи {epoch+1}/{epochs} для файла {file}")
        self.progress_bar = tqdm(
            total=1,  # Один шаг на файл
            desc=f"Прогресс эпохи {epoch + 1}",
            unit="file",
            postfix={"loss": "?"},
            dynamic_ncols=True,
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"
        )

    def update_batch(self, loss):
        self.progress_bar.set_postfix({"loss": f"{loss:.4f}"})
        self.progress_bar.update(1)

    def end_epoch(self, epoch, train_loss):
        self.progress_bar.close()
        elapsed = time.time() - self.epoch_start_time
        print(f"[Результаты] Эпоха {epoch + 1} завершена за {elapsed:.1f} сек")
        print(f"Train Loss: {train_loss:.4f}")
        print("-"*70)

class TrainingConfig:
    csv_path= "books.csv"
    db_config= {
        'host': 'localhost',
        'database': 'BBK_index',
        'user': 'postgres',
        'password': 'Dima2003',
        'port': 5432
    }
    model_name= "DeepPavlov/rubert-base-cased-sentence"
    max_length= 512
    epochs_per_file= 3 # количество эпох для каждого файла
    learning_rate = 2e-5
    save_dir = "hierarchical_bbk_model"
    checkpoint: int = 10  # Частота сохранения модели (по количеству файлов)

# Класс для работы с данными
class BBKProcessor:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data = pd.read_csv(config.csv_path, dtype={'id': str})
        self.hierarchy, self.descriptions, self.keywords = self._load_metadata()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.num_labels = self._calculate_label_space()

    def _load_metadata(self):
        conn = psycopg2.connect(**self.config.db_config)
        cursor = conn.cursor()

        # Запрос для построения иерархии
        cursor.execute("""
            WITH RECURSIVE hierarchy AS (
                SELECT path::text, NULL::text AS parent, 0 AS level 
                FROM index_bbk 
                WHERE nlevel(path) = 1
                UNION ALL
                SELECT i.path::text, h.path AS parent, h.level + 1 
                FROM index_bbk i
                JOIN hierarchy h ON subpath(i.path, 0, -1)::text = h.path
            )
            SELECT path, parent, level FROM hierarchy
        """)
        hierarchy = {}
        rows = cursor.fetchall()
        with tqdm(total=len(rows), desc="Обработка иерархии", unit="rows") as pbar:
            for path, parent, level in rows:
                hierarchy[path] = {'parent': parent, 'level': level}
                pbar.update(1)
        cursor.execute("SELECT path::text, title, definition FROM index_bbk")
        rows = cursor.fetchall()
        descriptions = {}
        with tqdm(total=len(rows), desc="Обработка описаний", unit="rows") as pbar:
            for path, title, definition in rows:
                descriptions[path] = f"{title}. {definition}"
                pbar.update(1)
        cursor.execute("SELECT path::text, value FROM keywords_bbk")
        rows = cursor.fetchall()
        keywords = {}
        with tqdm(total=len(rows), desc="Обработка ключ.слов", unit="rows") as pbar:
            for path, value in rows:
                keywords.setdefault(path,[]).append(value)
                pbar.update(1)

        return hierarchy, descriptions, keywords

    def _calculate_label_space(self):
        levels = max([meta['level'] for meta in self.hierarchy.values()]) + 1
        label_counts = [0] * levels
        for path, meta in self.hierarchy.items():
            level = meta['level']
            label_id = int(path.split('.')[level])
            if label_id >= label_counts[level]:
                label_counts[level] = label_id + 1
        return label_counts

    def _load_and_process_text(self, pdf_path: str) -> str:
        if not os.path.exists(pdf_path):
            print(f"Файл не найден: {pdf_path}")
            return ""
        text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in tqdm(pdf.pages, desc=f"Обработка {os.path.basename(pdf_path)}", unit="page"):
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return self._clean_text(' '.join(text))

    def _clean_text(self, text) :
        text = re.sub(r'\b(рис|рисунок|табл)\.?\s*\d*[\.,]?\d*\b', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'[^\w\s.,!?\-—:;()%&§©®℗℠™°×÷π²√∅≈≠≤≥±→←↑↓∆ℓ∈∉∩∪∏∑−∛₀₁₂₃₄₅₆₇₈₉]', ' ', text)
        words = [word.lower() for word in text.split()]
        return " ".join(words)

    def _enrich_text(self, text, bbk_id):
        description = self.descriptions.get(bbk_id, "")
        keywords = ' '.join(self.keywords.get(bbk_id, []))
        return f"{description} [KEYWORDS] {keywords} [TEXT] {text}"

    def _get_hierarchical_labels(self, bbk_id):
        labels = []
        parts = bbk_id.split('.')
        for i in range(len(self.num_labels)):
            labels.append(int(parts[i]) if i < len(parts) else 0)
        return labels

    def process_file(self, idx):
        row = self.data.iloc[idx]
        pdf_path = row['path']
        bbk_id = str(row['id'])
        text = self._load_and_process_text(pdf_path)
        labels = self._get_hierarchical_labels(bbk_id)
        inputs = self.tokenizer(
            self._enrich_text(text, bbk_id),
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# Модель
class HierarchicalBERT(BertPreTrainedModel):
    def __init__(self, config, num_labels_per_level):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifiers = nn.ModuleList([
            nn.Linear(config.hidden_size, num_labels)
            for num_labels in num_labels_per_level
        ])

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.last_hidden_state[:, 0, :])
        logits = [cls(pooled_output) for cls in self.classifiers]
        loss = None
        if labels is not None:
            loss = 0
            for i, logit in enumerate(logits):
                loss += nn.CrossEntropyLoss()(logit, labels[i].unsqueeze(0))
            loss /= len(logits)
        return {'loss': loss, 'logits': logits}

# Функция обучения
def train(config: TrainingConfig = TrainingConfig()):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.save_dir, exist_ok=True)
    processor = BBKProcessor(config)
    model = HierarchicalBERT.from_pretrained(
        config.model_name,
        num_labels_per_level=processor.num_labels
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    monitor = TrainingMonitor()

    print("\n[Обучение] Старт цикла обучения по файлам...")
    for file_idx in range(len(processor.data)):
        file_data = processor.process_file(file_idx)
        file_name = os.path.basename(processor.data.iloc[file_idx]['path'])
        print(f"[Файл] Обработка файла {file_idx + 1}/{len(processor.data)}: {file_name}")

        for epoch in range(config.epochs_per_file):
            monitor.start_epoch(epoch, config.epochs_per_file, file_name)
            model.train()
            optimizer.zero_grad()

            inputs = {
                'input_ids': file_data['input_ids'].unsqueeze(0).to(device),
                'attention_mask': file_data['attention_mask'].unsqueeze(0).to(device),
                'labels': file_data['labels'].to(device)
            }
            outputs = model(**inputs)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()

            monitor.update_batch(loss.item())
            monitor.end_epoch(epoch, loss.item())

        if (file_idx + 1) % config.checkpoint == 0:
            save_path = f"{config.save_dir}/checkpoint_file_{file_idx + 1}"
            model.save_pretrained(save_path)
            print(f"[Сохранение] Чекпоинт: {save_path}")

    final_save_path = f"{config.save_dir}/final_model"
    model.save_pretrained(final_save_path)
    processor.tokenizer.save_pretrained(final_save_path)
    print(f"\n[Завершение] Модель сохранена в: {final_save_path}")


if __name__ == "__main__":
    print("=== Запуск обучения ===")
    train()
    print("=== Обучение завершено ===")