import torch
import psycopg2
from transformers import AutoTokenizer
from torch import nn, optim
from RBERTTEST.ml.Training import HierarchicalBERT


class IncrementalBBKTrain:
    def __init__(self, model_path, db_config):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HierarchicalBERT.from_pretrained(model_path, [29, 830, 24]).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.db_config = db_config
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-6)
        self.counter = 0

    def _bbk_to_labels(self, bbk_code):
        """Преобразует BBK в иерархию меток"""
        return [int(level) for level in bbk_code.split('.')]

    def _get_topic_metadata(self, bbk_code):
        """Получает метаданные для ББК из базы"""
        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT title, definition FROM index_bbk WHERE path = %s",
                    (bbk_code,)
                )
                title, definition = cursor.fetchone()

                cursor.execute(
                    "SELECT value FROM keywords_bbk WHERE path = %s",
                    (bbk_code,)
                )
                keywords = [row[0] for row in cursor.fetchall()]

        return {
            "title": title,
            "definition": definition,
            "keywords": keywords
        }

    def enrich_text(self, text, bbk_code):
        """Обогащает текст метаданными"""
        meta = self._get_topic_metadata(bbk_code)
        return (
            f"{meta['title']} [DEFINITION] {meta['definition']} "
            f"[KEYWORDS] {' '.join(meta['keywords'])} "
            f"[CONTENT] {text}"
        )

    def train_on_example(self, text, true_bbk):
        """Один шаг обучения на примере"""
        # Подготовка данных
        enriched_text = self.enrich_text(text, true_bbk)
        labels = self._bbk_to_labels(true_bbk)

        # Токенизация
        inputs = self.tokenizer(
            enriched_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        # Forward pass
        self.model.train()
        outputs = self.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=torch.tensor([labels]).to(self.device)
        )

        # Backward pass
        loss = outputs['loss']
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.counter += 1
        return float(loss.item())

    def save_updated_model(self, output_dir):
        """Сохраняет обновлённую модель"""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model updated and saved to {output_dir}")

    def get_update_count(self):
        return self.counter