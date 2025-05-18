import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import psycopg2
import RBERTTEST.ml.contextboost as contextboost
from torch import nn
from sklearn.base import BaseEstimator, RegressorMixin
import logging
import os
import pdfplumber

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger("pdfplumber").setLevel(logging.ERROR)


class ModernBertWithAttention(nn.Module):
    """Класс для динамического взвешивания контекста и выделения нужных терминов (улучшенный берт)"""
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size  # автоматическое определение размерности

        self.attention = nn.Sequential( # слои
            nn.Linear(self.hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        """Прямой обход"""
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = bert_outputs.last_hidden_state # извлечение последнего слоя
        attention_weights = torch.softmax(
            self.attention(hidden_states).squeeze(-1), #механизм внимания
            dim=1
        )

        # Возвращаем словарь с двумя выходами
        return {
            'pooled_output': torch.sum(attention_weights.unsqueeze(-1) * hidden_states, dim=1),
            'hidden_states': hidden_states
        }



class DocumentProcessor(BaseEstimator, RegressorMixin):
    """Класс по обработке документа"""
    def __init__(
            self,
            model_name = "answerdotai/ModernBERT-base",
            #model_name="C:\\AI_determinant_bbk_index\\RBERTTEST\\hierarchical_bbk_model\\final_model",
            stopwords_file= "stopwords-ru.txt",
            max_seq_length = 512,
            chunk_overlap = 64, # перекрытие между соседними чанками
            bert_weight = 0.5, # вес эмбедингов из BERT
            db_config = None
    ):
        self.model_name = model_name
        self.stopwords_file = stopwords_file
        self.max_seq_length = max_seq_length
        self.chunk_overlap = chunk_overlap
        self.bert_weight = bert_weight
        self.db_config = db_config or {
            'host': 'localhost',
            'database': 'BBK_index',
            'user': 'postgres',
            'password': 'Dima2003',
            'port': 5432
        }
        self.top = 0  # Уровень классификации (0, 1, 2)
        self._initialize_components()
        self._res ={}

    def _initialize_components(self):
        """Инициализация всех компонентов системы"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = ModernBertWithAttention(self.model_name)
            self.stopwords = self._load_stopwords()
            self.reference_topics = self._load_reference_topics()
        except Exception as e:
            raise

    def _load_stopwords(self):
        """Загрузка стопслов"""
        try:
            with open(self.stopwords_file, 'r', encoding='utf-8') as f:
                return {line.strip() for line in f if line.strip()}
        except Exception as e:
            return set()


    def _load_reference_topics(self, our_index = None):
        """Загрузка тем из БД"""
        try:
            connection = psycopg2.connect(**self.db_config)
            dict_theme = {}
            if self.top == 0:
                query = r"SELECT * FROM index_bbk WHERE length(regexp_replace(path::text, '[^0-9]', '', 'g')) = 2"
            if self.top == 1:

                query = rf"SELECT * FROM index_bbk WHERE path::text ~ '^{our_index+'.'}\d$' AND length(regexp_replace(path::text, '[^0-9]', '', 'g')) = 3"
            elif self.top == 2:
                query = rf"SELECT * FROM index_bbk WHERE path::text ~ '^{our_index}\d$' AND length(regexp_replace(path::text, '[^0-9]', '', 'g')) = 4"
            elif self.top == 3:
                query = rf"SELECT * FROM index_bbk WHERE path::text ~ '^{our_index}\d$' AND length(regexp_replace(path::text, '[^0-9]', '', 'g')) = 5"

            with connection.cursor() as cursor:
                cursor.execute(query)
                for row in cursor.fetchall():
                    dict_theme[f"{row[0]} {row[1]}"]=f"{row[1]}. {row[2]}"

            return dict_theme
        except Exception as e:
            return {}
        finally:
            if 'connection' in locals():
                connection.close()



    def preprocess_text(self, text):
        """Предобработка текста"""
        try:
            # Удаление лишних символов и графических элементов
            text = re.sub(r'\b(рис|рисунок|табл)\.?\s*\d*[\.,]?\d*\b', ' ', text, flags=re.IGNORECASE)
            text = re.sub(r'[^\w\s.,!?\-—:;()%&§©®℗℠™°×÷π²√∅≈≠≤≥±→←↑↓∆ℓ∈∉∩∪∏∑−∛₀₁₂₃₄₅₆₇₈₉]', ' ', text)
            words = [word.lower() for word in text.split()]
            return " ".join(words)
        except Exception as e:
            return ""

    def tokenize_and_chunk(self, text, window_size=512, stride=256):
        """Разбиение текста на чанки"""

        tokens = self.tokenizer.tokenize(text)
        chunks = []

        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i:i + window_size]
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)

            if i > 0 and self.chunk_overlap > 0:
                overlap_tokens = tokens[max(0, i - self.chunk_overlap):i] # берутся токены из предыдущего чанка (перекрытие)
                chunk_text = self.tokenizer.convert_tokens_to_string(overlap_tokens + chunk_tokens) # конвертация токенов в текст

            chunks.append(chunk_text)

        return chunks

    def extract_keywords(self, text):
        """Увеличение влияния ключевых слов, которые обозначены в тексте (не очень помогает)"""
        keywords = []
        patterns = [
            r"(?:КЛЮЧЕВЫЕ СЛОВА|Ключевые слова|Keywords)[:\s]*([^.]+)",
            r"Keywords:[ ]*(.+?)(?=\n|\.)",
            r"[К|к]лючевые слова[:\s]*([^.;]+)"
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            for match in matches:
                cleaned =re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9\-—.,]', ' ', match)
                keywords.extend([word.strip().lower() for word in cleaned.split(',') if word.strip()])

        return list(set(keywords))

    @lru_cache(maxsize=100) # сохранение результатов для повторяющихся текстов
    def get_embedding(self, text):
        inputs = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # получение эмбеддинга чанка
        with torch.no_grad():  #отключаю вычисление градиентов, т.к. модель тестируется
            outputs = self.model(**inputs)
            hidden_states = outputs['hidden_states']
            input_mask = inputs['attention_mask']

            #вычисление mean pooling
            masked_states = hidden_states * input_mask.unsqueeze(-1)
            sum_embeddings = torch.sum(masked_states, dim=1)
            sum_mask = torch.clamp(input_mask.sum(1), min=1e-9)

        # Убрали batch-размерность и преобразуем в numpy
        return (sum_embeddings / sum_mask.unsqueeze(-1)).squeeze(0).numpy()



    def analyze_document(self, text = None):
        """Метод для анализа текста"""
        with open('..\\web\\res.txt', 'w') as f:
            f.write('')
        try:
            if text is None:
                    raise ValueError()
            #Обработка
            clean_text = self.preprocess_text(text)
            explicit_keywords = self.extract_keywords(text)
            enhanced_text = clean_text + " " + " ".join(explicit_keywords * 3)

            #Эмбендингирование
            doc_embedding = self.get_embedding(enhanced_text)
            doc_words = set(enhanced_text.split())
            self.explicit_keywords_set = set(explicit_keywords)

            #Для каждой темы из БД вычисляется её эмбеддинг,
            #к осинусная схожесть между вектором документа и темы умножается на вес берт в финальной оценке
            results = {}
            for topic, desc in self.reference_topics.items():
                topic_emb = self.get_embedding(desc)
                #print(doc_embedding.shape, topic_emb.shape)
                try:
                    cos_sim = cosine_similarity(doc_embedding.reshape(1, -1),
                                                topic_emb.reshape(1, -1))[0][0]
                    results[topic] = (self.bert_weight * cos_sim)
                except Exception as e:
                    print(e)

            final_scores = {
                k: v for k, v in results.items()
            }
            #Усиление контекста тем, которые содержат ключевые термины

            if final_scores:
                 cb = contextboost.ContextBoost(final_scores, doc_words, self.explicit_keywords_set, self.tokenizer, self.model, doc_embedding)
                 if self.top== 0:
                     cb.processingTop0()
                 elif self.top == 1:
                     cb.processingTop1()
                 elif self.top == 2:
                     cb.processingTop2()
                 elif self.top == 3:
                     cb.processingTop3()
                 final_scores = cb.getfinal_scores()

            # Рекурсивная иерархическая классификация
            if self.top < 3 and final_scores:
                self.top += 1

                # Выбор лучшей темы, в чей узел спуститься рекурсия
                best_topic = max(final_scores.items(), key=lambda x: x[1])
                our_index = best_topic[0].split()[0]
                self.reference_topics = self._load_reference_topics(our_index)
                self.recursive_scores = self.analyze_document(text=text)

                #Запись тем в файл, инфа с которого будет выводиться на экран (надо сделать вывод более информативным)
                if self.top ==  3:
                    with open("..\\web\\res.txt", "a", encoding="utf-8") as f:
                        f.write("<br>".join([f"{k}: {v:.4f}" for k, v in sorted(self._normalize_scores(final_scores).items(), key=lambda item: item[1], reverse=True)[:4]]) + "<br><br>")
                else:
                    with open("..\\web\\res.txt", "a", encoding="utf-8") as f:
                        f.write("<br>".join([f"{k}: {v:.4f}" for k, v in
                                           sorted(self._normalize_scores(final_scores).items(),
                                                  key=lambda item: item[1], reverse=True)[:4]]) + "<br><br>")
                self.top -= 1

                return
            with open("..\\web\\res.txt", "a", encoding="utf-8") as f:
                f.write("<br>".join([f"{k}: {v:.4f}" for k, v in
                                   sorted(self._normalize_scores(final_scores).items(), key=lambda item: item[1],
                                          reverse=True)[:4]]) + "<br><br>")
            return

        except Exception as e:
            return {}

    def _normalize_scores(self, scores):
        """Нормализация оценок"""
        total = sum(scores.values())
        return {k: v / total for k, v in scores.items()} if total else {}

def extract_text( pdf_path, start_page=0):
    """Извлечение текста из PDF с обработкой ошибок"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text_pages = []
            for i, page in enumerate(pdf.pages[start_page:], start=start_page):
                text = page.extract_text()
                if text:
                    text_pages.append(text)
            result = " ".join(text_pages) if text_pages else ""
            return result
    except Exception as e:
        return ""

if __name__ == "__main__":
    processor = DocumentProcessor()
    book_text = extract_text('books\\22.123.pdf')
    processor.analyze_document(book_text)
