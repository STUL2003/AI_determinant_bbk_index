import re
import pdfplumber
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from functools import lru_cache
import psycopg2
import RBERTTEST.ml.contextboost as contextboost
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn
from sklearn.base import BaseEstimator, RegressorMixin
import warnings
import logging
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger("pdfplumber").setLevel(logging.ERROR)

class BertWithAttention(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.attention = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask, return_dict=True)
        hidden_states = outputs.last_hidden_state
        weights = torch.softmax(self.attention(hidden_states).squeeze(-1), dim=1)
        return torch.sum(weights.unsqueeze(-1) * hidden_states, dim=1)


class DocumentProcessor(BaseEstimator, RegressorMixin):
    def __init__(
            self,
            model_name = "DeepPavlov/rubert-base-cased-sentence",
            #model_name="C:\\AI_determinant_bbk_index\\RBERTTEST\\hierarchical_bbk_model\\final_model",
            stopwords_file= "stopwords-ru.txt",
            max_seq_length = 512,
            chunk_overlap = 64,
            bert_weight = 0.5,
            keyword_weight = 0.5,
            relative_threshold = 0.7,
            absolute_threshold = 0.2,
            use_attention = True,
            db_config = None
    ):
        self.model_name = model_name
        self.stopwords_file = stopwords_file
        self.max_seq_length = max_seq_length
        self.chunk_overlap = chunk_overlap
        self.bert_weight = bert_weight
        self.keyword_weight = keyword_weight
        self.relative_threshold = relative_threshold
        self.absolute_threshold = absolute_threshold
        self.use_attention = use_attention
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
            self.model = BertWithAttention(self.model_name) if self.use_attention else AutoModel.from_pretrained(
                self.model_name)
            self.stopwords = self._load_stopwords()
            self._init_tfidf()
            self.reference_topics = self._load_reference_topics()
        except Exception as e:
            raise

    def _load_stopwords(self):
        try:
            with open(self.stopwords_file, 'r', encoding='utf-8') as f:
                return {line.strip() for line in f if line.strip()}
        except Exception as e:
            return set()

    def _init_tfidf(self):
        """Инициализация TF-IDF векторайзера"""
        self.tfidf = TfidfVectorizer(
            tokenizer=lambda x: x.split(),
            binary=False,
            min_df=2,
            max_df=0.95,
            max_features=10000,
            stop_words=list(self.stopwords)
        )
        self.tfidf_fitted = False

    def _load_reference_topics(self, our_index = None):
        """Загрузка тем из БД с обработкой ошибок"""
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
        try:
            # Удаление лишних символов и графических элементов
            text = re.sub(r'\b(рис|рисунок|табл)\.?\s*\d*[\.,]?\d*\b', ' ', text, flags=re.IGNORECASE)
            text = re.sub(r'[^\w\s.,!?\-—:;()%&§©®℗℠™°×÷π²√∅≈≠≤≥±→←↑↓∆ℓ∈∉∩∪∏∑−∛₀₁₂₃₄₅₆₇₈₉]', ' ', text)

            words = [
                word.lower() for word in text.split()
            ]
            return " ".join(words)
        except Exception as e:
            return ""

    def tokenize_and_chunk(self, text):
        if not text:
            return []

        try:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks = []
            current_chunk = []

            for sent in sentences:
                tokens = self.tokenizer.tokenize(sent)
                if len(current_chunk) + len(tokens) <= self.max_seq_length:
                    current_chunk.extend(tokens)
                else:
                    chunks.append(self.tokenizer.convert_tokens_to_string(current_chunk))
                    current_chunk = tokens[-self.chunk_overlap:] if self.chunk_overlap > 0 else []

            if current_chunk:
                chunks.append(self.tokenizer.convert_tokens_to_string(current_chunk))

            return chunks
        except Exception as e:
            return []

    def extract_keywords(self, text):

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

    @lru_cache(maxsize=100)
    def get_embedding(self, text):
        chunks = self.tokenize_and_chunk(text)
        if not chunks:
            return np.zeros((self.model.bert.config.hidden_size,))

        embeddings = []
        for chunk in chunks:
            try:
                inputs = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_seq_length
                )
                if 'token_type_ids' in inputs:
                    del inputs['token_type_ids']

                with torch.no_grad():
                    if self.use_attention:
                        chunk_emb = self.model(**inputs).numpy()[0]
                    else:
                        outputs = self.model(**inputs)
                        chunk_emb = outputs.last_hidden_state[0, 0, :].numpy()
                    embeddings.append(chunk_emb)
            except Exception as e:
                continue

        if not embeddings:
            return np.zeros((self.model.bert.config.hidden_size,))

        doc_embedding = np.mean(embeddings, axis=0)
        return normalize(doc_embedding.reshape(1, -1))[0]

    def _fit_tfidf(self, texts):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.tfidf.fit(texts)
                self.tfidf_fitted = True
        except Exception as e:
            self.tfidf_fitted = False

    def _get_tfidf_weight(self, word):
        if not self.tfidf_fitted:
            return 0.0


        try:
            return self.tfidf.idf_[self.tfidf.vocabulary_[word.lower()]]
        except (KeyError, AttributeError):
            return 0.0

    def jaccard_tfidf(self, doc_words, topic_words):
        if not doc_words or not topic_words or not self.tfidf_fitted:
            return 0.0
        try:
            common = doc_words & topic_words
            union = doc_words | topic_words
            keyword_boost = 2.0

            numerator = sum(self._get_tfidf_weight(w) * (keyword_boost if w in self.explicit_keywords_set else 1)
                            for w in common)

            denominator = sum(self._get_tfidf_weight(w) for w in union)

            return numerator / denominator if denominator else 0.0
        except Exception as e:
            return 0.0

    def analyze_document(self, text = None):
        with open('res.txt', 'w') as f:
            f.write('')
        try:
            if text is None:
                    raise ValueError()

            clean_text = self.preprocess_text(text)
            explicit_keywords = self.extract_keywords(text)
            enhanced_text = clean_text + " " + " ".join(explicit_keywords * 3)

            doc_embedding = self.get_embedding(enhanced_text)
            doc_words = set(enhanced_text.split())
            self.explicit_keywords_set = set(explicit_keywords)
            if not hasattr(self, 'tfidf_fitted'):
                self._fit_tfidf([clean_text] + list(self.reference_topics.values()))

            results = {}
            for topic, desc in self.reference_topics.items():
                topic_emb = self.get_embedding(desc)
                topic_words = set(desc.split())

                cos_sim = cosine_similarity([doc_embedding], [topic_emb])[0][0]
                jaccard_sim = self.jaccard_tfidf(doc_words, topic_words)
                combined = (self.bert_weight * cos_sim +
                            self.keyword_weight * jaccard_sim)
                results[topic] = combined


            max_score = max(results.values(), default=0)# Фильтрация с учетом порогов
            final_scores = {
                k: v for k, v in results.items()
                if v>=self.relative_threshold * max_score
                   and v>self.absolute_threshold
            }

            if final_scores:
                 cb = contextboost.ContextBoost(final_scores, doc_words, self.explicit_keywords_set)
                 if self.top== 0:
                     cb.processingTop0()
                 elif self.top == 1:
                     cb.processingTop1()
                 elif self.top == 2:
                     cb.processingTop2()
                 elif self.top == 3:
                     cb.processingTop3()
                 final_scores = cb.getfinal_scores()

            if self.top < 3 and final_scores:
                self.top += 1
                best_topic = max(final_scores.items(), key=lambda x: x[1])
                our_index = best_topic[0].split()[0]
                self.reference_topics = self._load_reference_topics(our_index)
                self.recursive_scores = self.analyze_document(text=text)
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
            with open("RBERTTEST\\web\\res.txt", "a", encoding="utf-8") as f:
                f.write("<br>".join([f"{k}: {v:.4f}" for k, v in
                                   sorted(self._normalize_scores(final_scores).items(), key=lambda item: item[1],
                                          reverse=True)[:4]]) + "<br><br>")
            return

        except Exception as e:
            return {}

    def _normalize_scores(self, scores):
        total = sum(scores.values())
        return {k: v / total for k, v in scores.items()} if total else {}

# def main():
#     try:
#         processor = DocumentProcessor()
#         with open('res.txt', 'w') as f:f.write('')
#         processor.analyze_document("books\\26.111.pdf")
#
#     except Exception as e:
#         print(e)
#
#
# if __name__ == "__main__":
#     main()