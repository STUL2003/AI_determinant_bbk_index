import re
import pdfplumber
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from functools import lru_cache
import psycopg2
import contextboost


class DocumentProcessor:
    def __init__(self, pdf_path, model_name = "DeepPavlov/rubert-base-cased-sentence", stopwords_file = "stopwords-ru.txt",
                 max_seq_length= 512, chunk_overlap = 64):

        self.top = 0
        self.tokenizer, self.model = self._initialize_model(model_name) # Загрузка модели
        self.stopwords = self._load_stopwords(stopwords_file) # загрузка стопслов
        self.reference_topics = self._get_reference_topics() # загрузка тем из бд
        self.max_seq_length = max_seq_length # макс. длина текста
        self.chunk_overlap = chunk_overlap # чанк для перекрытия

        self.raw_text = self.extract_text(pdf_path)
        self.clean_text = self.preprocess_text(self.raw_text)



    @staticmethod
    def _initialize_model(model_name): # загрузка модеи
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        return tokenizer, model


    @staticmethod
    def _load_stopwords(file_path):  # Загрузка стопслов с файла наа гитхабе
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return {line.strip() for line in f if line.strip()}
        except Exception as e:
            print(f"Нет файла: {e}")
            return set()

    def _get_reference_topics(self, our_index = None):
        connection = psycopg2.connect(
            host="localhost",
            database="BBK_index",
            user="postgres",
            password="Dima2003",
            port=5432
        )
        dict_theme = {}
        query = ''
        if self.top == 0: query = r"SELECT * FROM index_bbk WHERE path::text ~ '^[0-9]+\.[0-9]$';"
        elif self.top == 1: query = rf"SELECT * FROM index_bbk WHERE path::text ~ '^{our_index}\d$'AND length(regexp_replace(path::text, '[^0-9]', '', 'g')) = 4"
        elif self.top == 2: query = rf"SELECT * FROM index_bbk WHERE path::text ~ '^{our_index}\d$'AND length(regexp_replace(path::text, '[^0-9]', '', 'g')) = 5"

        with connection.cursor() as cursor:
            cursor.itersize = 1000  # сколько строк подгружать за раз
            cursor.execute(query)
            for row in cursor.fetchall():
                dict_theme[f"{row[0]} {row[1]}"] = f"{row[1]}. {row[2]}"

        return dict_theme

    def extract_text(self, pdf_path, start_page = 0): # извлечение текста
        with pdfplumber.open(pdf_path) as pdf:
            text_pages = []
            for i, page in enumerate(pdf.pages[start_page:], start=start_page):
                text = page.extract_text()
                if text: text_pages.append(text)

            return " ".join(text_pages) if text_pages else ""

    # Пример модификации:
    def preprocess_text(self, text):
        # Сохраняем латинские названия и формулы
        text = re.sub(r'\b(рис|рисунок|табл)\.?\s*\d*[\.,]?\d*\b', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'[^\w\s.,!?\-—:;()%&§©®℗℠™°×÷π²√∅≈≠≤≥±→←↑↓∆ℓ∈∉∩∪∏∑−∛₀₁₂₃₄₅₆₇₈₉]', ' ', text)

        keep_words = {"атф", "днк", "рнк", "корень", "стебель", "фотосинтез", "гаметофит"}
        words = [
            word.lower() for word in text.split()
            if (word.lower() not in self.stopwords or word in keep_words)
               or len(word) >= 2  # Разрешаем 2-буквенные термины
        ]

        return " ".join(words)

    def tokenize_and_chunk(self, text):
        # Разбиваем по предложениям, а не фиксированной длине
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []

        for sent in sentences:
            tokens = self.tokenizer.tokenize(sent)
            if len(current_chunk) + len(tokens) <= self.max_seq_length:
                current_chunk.extend(tokens)
            else:
                chunks.append(self.tokenizer.convert_tokens_to_string(current_chunk))
                current_chunk = tokens[-self.chunk_overlap:]  # Перекрытие

        if current_chunk:
            chunks.append(self.tokenizer.convert_tokens_to_string(current_chunk))

        return chunks

    @lru_cache(maxsize=100)
    def get_embedding(self, text):
        chunks = self.tokenize_and_chunk(text)
        cls_embeddings = []

        for chunk in chunks:
            inputs = self.tokenizer(chunk, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_seq_length)

            with torch.no_grad():
                outputs = self.model(**inputs)
                cls_embedding = outputs.last_hidden_state[0, 0, :].numpy()
                cls_embeddings.append(cls_embedding)

        doc_embedding = np.mean(cls_embeddings, axis=0)
        return normalize(doc_embedding.reshape(1, -1))[0]

    def analyze_document(self):
        try:

            chunks = self.tokenize_and_chunk(self.clean_text)
            if not chunks:
                return {}

            cls_embeddings = []
            for chunk in chunks:
                inputs = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_seq_length
                )

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    cls_embedding = outputs.last_hidden_state[0, 0, :].numpy()
                    cls_embeddings.append(cls_embedding)

            # Взвешенное усреднение (больший вес последним чанкам)
            weights = np.linspace(0.5, 1.5, len(cls_embeddings))
            doc_embedding = np.average(cls_embeddings, axis=0, weights=weights)
            doc_embedding = normalize(doc_embedding.reshape(1, -1))[0]

            # Анализ ключевых слов
            doc_words = set(self.clean_text.split())
            results = {}

            for topic, desc in self.reference_topics.items():
                # Комбинированная схожесть
                topic_embedding = self.get_embedding(desc)
                topic_words = set(desc.split())
                cos_sim = cosine_similarity([doc_embedding], [topic_embedding])[0][0]
                jaccard_sim = len(doc_words & topic_words) / len(doc_words | topic_words) if doc_words else 0

                combined_score = 0.6 * cos_sim + 0.4 * jaccard_sim
                results[topic] = combined_score

            # пороговая фильтрация
            max_score = max(results.values()) if results else 0
            final_scores = {}
            for topic, score in results.items():
                if score >= 0.7 * max_score and score > 0.2:  # Абсолютный и относительный пороги
                    final_scores[topic] = round(score, 3)

            cb = contextboost.ContextBoost(final_scores, doc_words)
            if self.top == 0:
                cb.processingTop1()
            elif self.top == 1:
                cb.processingTop2()

            final_scores = cb.getfinal_scores()

            # Нормализация
            total = sum(final_scores.values())
            if total == 0:
                return {}

            normalized_scores = {k: v / total for k, v in final_scores.items()}

            if self.top == 0 or self.top == 1:
                self.top += 1
                our_index = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)[0][0].split()[0]
                self.reference_topics = self._get_reference_topics(our_index)

                # Сохраняем результат рекурсии
                recursive_scores = self.analyze_document()

                # Возвращаем комбинированный результат (или только рекурсивный)
                return recursive_scores  # Или {**normalized_scores, **recursive_scores}
            return normalized_scores
        except Exception as e:
            raise


def main():
    processor = DocumentProcessor("mikro2.pdf")

    try:
        results = processor.analyze_document()
        if not results:
            print("No results generated")
            return

        print("\nDocument Topics:")
        for topic, score in sorted(results.items(),
                                   key=lambda x: -x[1]):
            print(f"- {topic}: {score:.3f}")

    except KeyboardInterrupt:
        print("\nProcessing interrupted")
    except Exception as e:
        print(f"Critical error: {e}")


if __name__ == "__main__":
    main()

