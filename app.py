import faiss
import numpy as np
import torch
import feedparser
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
from urllib.parse import quote_plus

# Налаштування констант
MONGO_URI = 'mongodb://localhost:27017/'
DB_NAME = 'articles_db'
COLLECTION_NAME = 'articles'
FAISS_INDEX_PATH = 'backend/index/vectors_index'
ID_MAPPING_PATH = 'backend/index/id_mapping.npy'
MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'

def get_vector(query):
    """
    Обчислює вектор запиту за допомогою моделі MPNet.

    :param query: Текст запиту.
    :return: NumPy-масив з вектором запиту.
    """
    inputs = tokenizer_mpnet(
        query,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model_mpnet(**inputs)
    last_hidden = outputs.last_hidden_state
    # Маска для врахування attention
    mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
    # Обчислення середнього значення embedding з урахуванням маски
    sum_embeddings = torch.sum(last_hidden * mask, dim=1)
    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
    query_vector = sum_embeddings / sum_mask
    return query_vector.cpu().numpy()

def search_articles(query_vector, index, k=10):
    """
    Пошук k найближчих статей за запитом.

    :param query_vector: NumPy-масив з вектором запиту.
    :param index: Faiss індекс.
    :param k: Кількість результатів.
    :return: Кортеж (індекси статей, відстані).
    """
    query_vector = query_vector.astype('float32').reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    return indices[0], distances[0]

# Ініціалізація Flask додатку
app = Flask(__name__)
CORS(app)

# Підключення до MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
articles_collection = db[COLLECTION_NAME]

# Завантаження моделі та токенізатора MPNet
tokenizer_mpnet = AutoTokenizer.from_pretrained(MODEL_NAME)
model_mpnet = AutoModel.from_pretrained(MODEL_NAME)

# Завантаження Faiss індексу та налаштування параметрів
vectors_index = faiss.read_index(FAISS_INDEX_PATH, faiss.IO_FLAG_MMAP)
vectors_index.hnsw.efSearch = 150

# Завантаження мапінгу ідентифікаторів
id_mapping = np.load(ID_MAPPING_PATH, allow_pickle=True)

@app.route('/api/search', methods=['POST'])
def init_search_papers():
    """
    API для пошуку статей.
    Приймає JSON: {"query": "<текст запиту>"}.
    Повертає JSON зі списком статей та їх релевантністю.
    """
    data = request.json
    query = data.get('query', '')

    # Обмеження: не більше 384 слів у запиті
    if len(query.split()) > 384:
        return jsonify({'error': 'Abstract exceeds 384 words limit'}), 400

    query_vector = get_vector(query)
    query_vector = normalize(query_vector, norm='l2', axis=1)
    paper_indices, scores = search_articles(query_vector, vectors_index, k=10)

    # Розрахунок релевантності у відсотках
    relevance_percentages = [score * 100 for score in scores]

    # Отримання arXiv ID з мапінгу (декодування байтових рядків)
    arxiv_ids = [id_mapping[i].decode() for i in paper_indices]

    # Запит до MongoDB для отримання документів
    papers = list(articles_collection.find({"id": {"$in": arxiv_ids}}))

    # Створення словника для швидкого доступу до документів
    paper_dict = {paper['id']: paper for paper in papers}

    # Сортування статей відповідно до списку arxiv_ids та релевантності
    sorted_papers = []
    for arxiv_id, relevance in zip(arxiv_ids, relevance_percentages):
        paper = paper_dict.get(arxiv_id, {"id": arxiv_id, "error": "Document not found"})
        paper['relevance'] = float(round(relevance, 2))
        # Видалення непотрібного поля MongoDB
        paper.pop('_id', None)
        sorted_papers.append(paper)

    return jsonify({'papers': sorted_papers}), 200

@app.route('/api/get_pdf_links', methods=['POST'])
def get_pdf_links():
    """
    API для отримання посилань на PDF.
    Приймає JSON: {"arxiv_ids": [list of arxiv id strings]}.
    Повертає JSON: {"pdf_links": {arxiv_id: pdf_url, ...}}.
    """
    data = request.json
    arxiv_ids = data.get('arxiv_ids', [])

    # Якщо arxiv_ids порожній, повертаємо порожній словник
    if not arxiv_ids:
        return jsonify({'pdf_links': {}}), 200

    base_url = "http://export.arxiv.org/api/query?"
    # Формування запиту до ArXiv API через оператор "OR"
    search_query = "+OR+".join([f"id:{quote_plus(arxiv_id)}" for arxiv_id in arxiv_ids])
    query_url = f"{base_url}search_query={search_query}&start=0&max_results={len(arxiv_ids)}"

    response = feedparser.parse(query_url)
    # Ініціалізація словника для посилань
    pdf_links = {arxiv_id: None for arxiv_id in arxiv_ids}

    if response.status == 200:
        for entry in response.entries:
            # Отримання базового arXiv ID (без версії)
            full_id = entry.id.split('/')[-1]  # Наприклад, "1701.00123v1"
            base_id = full_id.split('v')[0]    # Наприклад, "1701.00123"
            # Якщо base_id співпадає з запитаним arxiv_id, зберігаємо посилання
            for arxiv_id in arxiv_ids:
                if arxiv_id.strip() == base_id:
                    for link in entry.get('links', []):
                        if link.get('type') == 'application/pdf':
                            pdf_links[arxiv_id] = link.get('href')
                            break
    else:
        for arxiv_id in arxiv_ids:
            pdf_links[arxiv_id] = None

    return jsonify({'pdf_links': pdf_links}), 200

if __name__ == '__main__':
    app.run(debug=True)
