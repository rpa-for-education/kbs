import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
from datetime import datetime
import time

class DepartmentClassifierAPI:
    def __init__(self):
        # API endpoints
        self.comment_api = "https://api.rpa4edu.shop/api_binh_luan.php"
        self.post_api = "https://api.rpa4edu.shop/api_bai_viet.php"
        self.dictionary_api = "https://api.rpa4edu.shop/api_tu_dien.php"
        self.result_api = "https://api.rpa4edu.shop/api_bai_viet_binh_luan_don_vi.php"
        
        # Initialize PhoBERT
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_auth_token=False)
        self.model = AutoModel.from_pretrained("vinai/phobert-base", use_auth_token=False)
        
        # Load initial keywords
        self.departments = self._load_keywords()
        self.department_embeddings = self._calculate_department_embeddings()

    def _load_keywords(self):
        try:
            response = requests.get(self.dictionary_api)
            keywords_data = response.json()
            
            departments = {}
            for item in keywords_data:
                dept_id = item['id_don_vi']
                if dept_id not in departments:
                    departments[dept_id] = []
                departments[dept_id].append(item['tu_khoa'])
            
            return departments
            
        except Exception as e:
            print(f"Error loading keywords: {str(e)}")
            return {}

    def _calculate_department_embeddings(self):
        department_embeddings = {}
        for dept_id, keywords in self.departments.items():
            dept_emb = self._get_embeddings(" ".join(keywords))
            department_embeddings[dept_id] = dept_emb.mean(axis=0)
        return department_embeddings

    def _get_embeddings(self, text):
        encoded = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**encoded)
            embeddings = outputs.last_hidden_state.numpy()
        return embeddings[0]

    def classify_text(self, text):
        text_emb = self._get_embeddings(text)
        text_emb_mean = text_emb.mean(axis=0)
        
        results = {}
        for dept_id, dept_emb in self.department_embeddings.items():
            similarity = cosine_similarity([text_emb_mean], [dept_emb])[0][0]
            if similarity >= 0.6:  # Only keep results >= 60%
                results[dept_id] = float(similarity * 100)
        
        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    def process_content(self):
        # Get existing results
        existing_results = self._get_existing_results()
        
        # Process posts and comments
        posts = requests.get(self.post_api).json()
        comments = requests.get(self.comment_api).json()
        
        # Create mapping of posts to comments
        post_comments = {}
        for comment in comments:
            post_id = comment['id_bai_viet']
            if post_id not in post_comments:
                post_comments[post_id] = []
            post_comments[post_id].append(comment)

        # Process each post
        for post in posts:
            post_id = post['id_bai_viet']
            
            if post_id in post_comments:
                # Process comments for this post
                for comment in post_comments[post_id]:
                    self._process_item(post, comment, existing_results)
            else:
                # Process post without comments
                self._process_item(post, None, existing_results)

    def _process_item(self, post, comment, existing_results):
        content = comment['noi_dung_binh_luan'] if comment else post['noi_dung_bai_viet']
        classifications = self.classify_text(content)
        
        for dept_id, similarity in classifications.items():
            result_id = self._get_result_id(post['id_bai_viet'], 
                                          comment['id_binh_luan'] if comment else None,
                                          dept_id,
                                          existing_results)
            
            if result_id:
                # Update existing result
                self._update_result(result_id, similarity)
            else:
                # Create new result
                self._create_result(post['id_bai_viet'],
                                  comment['id_binh_luan'] if comment else None,
                                  dept_id,
                                  similarity)

    def _get_existing_results(self):
        try:
            response = requests.get(self.result_api)
            return response.json()
        except:
            return []

    def _get_result_id(self, post_id, comment_id, dept_id, existing_results):
        for result in existing_results:
            if (result['id_bai_viet'] == post_id and
                result['id_binh_luan'] == (comment_id if comment_id else "0") and
                result['id_don_vi'] == dept_id):
                return result['id_bai_viet_binh_luan']
        return None

    def _create_result(self, post_id, comment_id, dept_id, similarity):
        data = {
            "id_bai_viet": post_id,
            "id_binh_luan": comment_id if comment_id else "0",
            "id_don_vi": dept_id,
            "phan_tram_lien_quan": str(round(similarity, 2)),
            "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        requests.post(self.result_api, json=data)

    def _update_result(self, result_id, similarity):
        data = {
            "id_bai_viet_binh_luan": result_id,
            "phan_tram_lien_quan": str(round(similarity, 2)),
            "modified_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        requests.put(f"{self.result_api}/{result_id}", json=data)

    def update_keywords(self):
        # Reload keywords and recalculate embeddings
        self.departments = self._load_keywords()
        self.department_embeddings = self._calculate_department_embeddings()
        
        # Reprocess all content with new keywords
        self.process_content()

def main():
    try:
        print("Initializing Department Classifier...")
        classifier = DepartmentClassifierAPI()
        
        while True:
            print("\nProcessing content...")
            classifier.process_content()
            
            print("Checking for keyword updates...")
            classifier.update_keywords()
            
            print("Waiting for next cycle...")
            time.sleep(300)  # Wait 5 minutes before next check
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()