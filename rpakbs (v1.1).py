import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
from datetime import datetime
import time
import uuid


class DepartmentClassifierAPI:
    def __init__(self):
        self.comment_api = "https://api.rpa4edu.shop/api_binh_luan.php"
        self.post_api = "https://api.rpa4edu.shop/api_bai_viet.php"
        self.dictionary_api = "https://api.rpa4edu.shop/api_tu_dien.php"
        self.result_api = "https://api.rpa4edu.shop/api_bai_viet_binh_luan_don_vi.php"

        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_auth_token=False)
        self.model = AutoModel.from_pretrained("vinai/phobert-base", use_auth_token=False)

        self.departments = self._load_keywords()
        self.department_embeddings = self._calculate_department_embeddings()
        self.processed_items = set()  # Lưu trữ các cặp (post_id, comment_id) đã xử lý

    def _load_keywords(self):
        try:
            response = requests.get(self.dictionary_api)
            if response.status_code != 200:
                print(f"[{uuid.uuid4()}] Error: Failed to fetch keywords, status code: {response.status_code}")
                return {}
            keywords_data = response.json()
            if not isinstance(keywords_data, list):
                print(f"[{uuid.uuid4()}] Error: Keywords data is not a list, got: {keywords_data}")
                return {}

            departments = {}
            for item in keywords_data:
                if not isinstance(item, dict) or 'id_don_vi' not in item or 'tu_khoa' not in item:
                    print(f"[{uuid.uuid4()}] Warning: Invalid keyword item: {item}")
                    continue
                dept_id = item['id_don_vi']
                if dept_id not in departments:
                    departments[dept_id] = []
                departments[dept_id].append(item['tu_khoa'])
            return departments
        except Exception as e:
            print(f"[{uuid.uuid4()}] Error loading keywords: {str(e)}")
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
            if similarity >= 0.6:
                results[dept_id] = float(similarity * 100)
        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    def process_content(self):
        existing_results = self._get_existing_results()
        if not isinstance(existing_results, list):
            print(f"[{uuid.uuid4()}] Error: Existing results is not a list, skipping processing")
            return False

        posts_response = requests.get(self.post_api)
        if posts_response.status_code != 200:
            print(f"[{uuid.uuid4()}] Error: Failed to fetch posts, status code: {posts_response.status_code}")
            return False
        try:
            posts = posts_response.json()
            if not isinstance(posts, list):
                print(f"[{uuid.uuid4()}] Error: Posts data is not a list, got: {posts}")
                return False
        except Exception as e:
            print(f"[{uuid.uuid4()}] Error parsing posts JSON: {str(e)}")
            return False

        comments_response = requests.get(self.comment_api)
        if comments_response.status_code != 200:
            print(f"[{uuid.uuid4()}] Error: Failed to fetch comments, status code: {comments_response.status_code}")
            return False
        try:
            comments = comments_response.json()
            if not isinstance(comments, list):
                print(f"[{uuid.uuid4()}] Error: Comments data is not a list, got: {comments}")
                return False
        except Exception as e:
            print(f"[{uuid.uuid4()}] Error parsing comments JSON: {str(e)}")
            return False

        # Nếu không có bài viết hoặc bình luận mới, dừng lại
        if not posts and not comments:
            print(f"[{uuid.uuid4()}] Info: No new posts or comments to process. Stopping.")
            return False

        post_comments = {}
        for comment in comments:
            if not isinstance(comment, dict) or 'id_bai_viet' not in comment:
                print(f"[{uuid.uuid4()}] Warning: Invalid comment item: {comment}")
                continue
            post_id = comment['id_bai_viet']
            if post_id not in post_comments:
                post_comments[post_id] = []
            post_comments[post_id].append(comment)

        has_new_data = False
        for post in posts:
            if not isinstance(post, dict) or 'id_bai_viet' not in post:
                print(f"[{uuid.uuid4()}] Warning: Invalid post item: {post}")
                continue
            post_id = post['id_bai_viet']
            if post_id in post_comments:
                for comment in post_comments[post_id]:
                    if self._process_item(post, comment, existing_results):
                        has_new_data = True
            else:
                if self._process_item(post, None, existing_results):
                    has_new_data = True

        return has_new_data

    def _process_item(self, post, comment, existing_results):
        if not isinstance(post, dict) or 'noi_dung_bai_viet' not in post or 'id_bai_viet' not in post:
            print(f"[{uuid.uuid4()}] Error: Invalid post data: {post}")
            return False

        post_id = post['id_bai_viet']
        comment_id = comment['id_binh_luan'] if comment and isinstance(comment,
                                                                       dict) and 'id_binh_luan' in comment else "0"
        item_key = (post_id, comment_id)

        # Nếu item đã được xử lý trước đó, bỏ qua
        if item_key in self.processed_items:
            return False

        if comment and (not isinstance(comment, dict) or 'noi_dung_binh_luan' not in comment):
            print(f"[{uuid.uuid4()}] Error: Invalid comment data: {comment}")
            return False

        content = comment['noi_dung_binh_luan'] if comment else post['noi_dung_bai_viet']
        classifications = self.classify_text(content)

        has_changes = False
        for dept_id, similarity in classifications.items():
            result_id = self._get_result_id(post_id, comment_id, dept_id, existing_results)
            if result_id:
                # Kiểm tra xem phan_tram_lien_quan có thay đổi không
                for result in existing_results:
                    if result.get('id_bai_viet_binh_luan') == result_id:
                        old_similarity = float(result.get('phan_tram_lien_quan', 0))
                        if abs(old_similarity - similarity) > 0.01:  # Ngưỡng thay đổi nhỏ
                            self._update_result(result_id, similarity)
                            has_changes = True
                        break
            else:
                self._create_result(post_id, comment_id, dept_id, similarity)
                has_changes = True

        if has_changes:
            self.processed_items.add(item_key)
        return has_changes

    def _get_existing_results(self):
        try:
            response = requests.get(self.result_api)
            if response.status_code != 200:
                print(f"[{uuid.uuid4()}] Error: Failed to fetch existing results, status code: {response.status_code}")
                return []
            results = response.json()
            if not isinstance(results, list):
                print(f"[{uuid.uuid4()}] Error: Existing results is not a list, got: {results}")
                return []
            return results
        except Exception as e:
            print(f"[{uuid.uuid4()}] Error fetching existing results: {str(e)}")
            return []

    def _get_result_id(self, post_id, comment_id, dept_id, existing_results):
        for result in existing_results:
            if not isinstance(result, dict):
                print(f"[{uuid.uuid4()}] Warning: Invalid result item: {result}")
                continue
            if (result.get('id_bai_viet') == post_id and
                    result.get('id_binh_luan') == (comment_id if comment_id else "0") and
                    result.get('id_don_vi') == dept_id):
                return result.get('id_bai_viet_binh_luan')
        return None

    def _create_result(self, post_id, comment_id, dept_id, similarity):
        data = {
            "id_bai_viet": post_id,
            "id_binh_luan": comment_id if comment_id else "0",
            "id_don_vi": dept_id,
            "phan_tram_lien_quan": str(round(similarity, 2)),
            "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        try:
            response = requests.post(self.result_api, json=data)
            if response.status_code not in [200, 201]:
                print(
                    f"[{uuid.uuid4()}] Error: Failed to create result, status code: {response.status_code}, response: {response.text}")
            else:
                print(
                    f"[{uuid.uuid4()}] Success: Created result, status code: {response.status_code}, response: {response.text}")
        except Exception as e:
            print(f"[{uuid.uuid4()}] Error creating result: {str(e)}")

    def _update_result(self, result_id, similarity):
        data = {
            "id_bai_viet_binh_luan": result_id,
            "phan_tram_lien_quan": str(round(similarity, 2)),
            "modified_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        try:
            response = requests.put(f"{self.result_api}/{result_id}", json=data)
            if response.status_code != 200:
                print(
                    f"[{uuid.uuid4()}] Error: Failed to update result, status code: {response.status_code}, response: {response.text}")
            else:
                print(f"[{uuid.uuid4()}] Success: Updated result {result_id} with new similarity {similarity}")
        except Exception as e:
            print(f"[{uuid.uuid4()}] Error updating result: {str(e)}")

    def update_keywords(self):
        self.departments = self._load_keywords()
        self.department_embeddings = self._calculate_department_embeddings()
        return self.process_content()


def main():
    try:
        print(f"[{uuid.uuid4()}] Initializing Department Classifier...")
        classifier = DepartmentClassifierAPI()

        while True:
            print(f"[{uuid.uuid4()}] Processing content...")
            has_new_data = classifier.process_content()

            if not has_new_data:
                print(f"[{uuid.uuid4()}] Info: No new data to process. Stopping.")
                break

            print(f"[{uuid.uuid4()}] Checking for keyword updates...")
            has_new_data = classifier.update_keywords()

            if not has_new_data:
                print(f"[{uuid.uuid4()}] Info: No new data after keyword update. Stopping.")
                break

            print(f"[{uuid.uuid4()}] Waiting for next cycle...")
            time.sleep(300)

    except Exception as e:
        print(f"[{uuid.uuid4()}] An error occurred: {str(e)}")


if __name__ == "__main__":
    main()