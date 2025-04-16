import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
from datetime import datetime
import time
import uuid
from tqdm import tqdm
import os


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
        self.processed_items = self._load_processed_items()
        self.similarity_threshold = 0.65  # Adjustable

    def _load_processed_items(self):
        """Load processed items from JSON file."""
        try:
            with open("processed_items.json", "r") as f:
                data = json.load(f)
                print(f"[{uuid.uuid4()}] Loaded {len(data)} processed items.")
                return {(item["post_id"], item["comment_id"]) for item in data}
        except FileNotFoundError:
            print(f"[{uuid.uuid4()}] No processed items file found, starting fresh.")
            return set()
        except Exception as e:
            print(f"[{uuid.uuid4()}] Error loading processed items: {str(e)}")
            return set()

    def _save_processed_items(self):
        """Save processed items to JSON file."""
        try:
            with open("processed_items.json", "w") as f:
                data = [{"post_id": p, "comment_id": c} for p, c in self.processed_items]
                json.dump(data, f)
            print(f"[{uuid.uuid4()}] Saved {len(self.processed_items)} processed items.")
        except Exception as e:
            print(f"[{uuid.uuid4()}] Error saving processed items: {str(e)}")

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
            print(f"[{uuid.uuid4()}] Loaded keywords for {len(departments)} departments.")
            return departments
        except Exception as e:
            print(f"[{uuid.uuid4()}] Error loading keywords: {str(e)}")
            return {}

    def _calculate_department_embeddings(self):
        department_embeddings = {}
        for dept_id, keywords in self.departments.items():
            dept_emb = self._get_embeddings(" ".join(keywords))
            department_embeddings[dept_id] = dept_emb.mean(axis=0)
        print(f"[{uuid.uuid4()}] Calculated embeddings for {len(department_embeddings)} departments.")
        return department_embeddings

    def _get_embeddings(self, text):
        if not text or not isinstance(text, str) or len(text.split()) < 3:
            print(f"[{uuid.uuid4()}] Warning: Invalid or too short text input for embeddings")
            return np.zeros((1, 768))

        try:
            encoded = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
            with torch.no_grad():
                outputs = self.model(**encoded)
                embeddings = outputs.last_hidden_state.numpy()
                if len(embeddings) == 0:
                    print(f"[{uuid.uuid4()}] Warning: Empty embeddings generated")
                    return np.zeros((1, 768))
                return embeddings[0]
        except Exception as e:
            print(f"[{uuid.uuid4()}] Error generating embeddings: {str(e)}")
            return np.zeros((1, 768))

    def classify_text(self, text):
        if not text or not isinstance(text, str) or len(text.split()) < 3:
            print(f"[{uuid.uuid4()}] Warning: Invalid or too short text input for classification")
            return {}

        try:
            text_emb = self._get_embeddings(text)
            if text_emb is None or len(text_emb) == 0:
                print(f"[{uuid.uuid4()}] Warning: Empty embeddings for classification")
                return {}

            text_emb_mean = text_emb.mean(axis=0)

            results = {}
            for dept_id, dept_emb in self.department_embeddings.items():
                if dept_emb is None or len(dept_emb) == 0:
                    continue
                similarity = cosine_similarity([text_emb_mean], [dept_emb])[0][0]
                if similarity >= self.similarity_threshold:
                    results[dept_id] = float(similarity * 100)
            return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            print(f"[{uuid.uuid4()}] Error in classification: {str(e)}")
            return {}

    def process_content(self, page=1, limit=1000):
        existing_results = self._get_existing_results()
        if not isinstance(existing_results, list):
            print(f"[{uuid.uuid4()}] Error: Existing results is not a list, skipping processing")
            return False

        # Fetch posts with pagination
        posts_response = requests.get(self.post_api, params={"page": page, "limit": limit})
        if posts_response.status_code != 200:
            print(f"[{uuid.uuid4()}] Error: Failed to fetch posts, status code: {posts_response.status_code}")
            return False
        posts = posts_response.json() if posts_response.status_code == 200 else []
        print(f"[{uuid.uuid4()}] Retrieved {len(posts)} posts (page {page})")

        # Fetch comments with pagination
        comments_response = requests.get(self.comment_api, params={"page": page, "limit": limit})
        if comments_response.status_code != 200:
            print(f"[{uuid.uuid4()}] Error: Failed to fetch comments, status code: {comments_response.status_code}")
            return False
        comments = comments_response.json() if comments_response.status_code == 200 else []
        print(f"[{uuid.uuid4()}] Retrieved {len(comments)} comments (page {page})")

        # Group comments by post
        post_comments = {post['id_bai_viet']: [] for post in posts if isinstance(post, dict) and 'id_bai_viet' in post}
        for comment in comments:
            if isinstance(comment, dict) and 'id_bai_viet' in comment:
                post_comments.setdefault(comment['id_bai_viet'], []).append(comment)

        # Check for duplicates
        post_ids = {post['id_bai_viet'] for post in posts if isinstance(post, dict) and 'id_bai_viet' in post}
        processed_post_ids = {p for p, c in self.processed_items}
        overlap = post_ids & processed_post_ids
        if len(overlap) / max(len(post_ids), 1) > 0.9:
            print(
                f"[{uuid.uuid4()}] Warning: {len(overlap)}/{len(post_ids)} duplicate post IDs in page {page}. Stopping.")
            return False

        has_new_data = False
        skipped_empty = 0
        skipped_duplicate = 0
        for post in tqdm(posts, desc=f"Processing posts (page {page})"):
            if not isinstance(post, dict) or 'id_bai_viet' not in post:
                continue
            post_id = post['id_bai_viet']
            if post_id in post_comments:
                if self._process_item(post, None, existing_results):
                    has_new_data = True
                for comment in tqdm(post_comments[post_id], desc=f"Processing comments for post {post_id}",
                                    leave=False):
                    if self._process_item(post, comment, existing_results):
                        has_new_data = True
                    else:
                        if (post_id, comment.get('id_binh_luan', '0')) in self.processed_items:
                            skipped_duplicate += 1
                        else:
                            skipped_empty += 1
            else:
                if self._process_item(post, None, existing_results):
                    has_new_data = True
                else:
                    if (post_id, '0') in self.processed_items:
                        skipped_duplicate += 1
                    else:
                        skipped_empty += 1

        print(
            f"[{uuid.uuid4()}] Finished page {page}. New items processed: {len(self.processed_items) - len(processed_post_ids)}, Skipped (duplicate): {skipped_duplicate}, Skipped (empty): {skipped_empty}")
        self._save_processed_items()
        return has_new_data

    def _process_item(self, post, comment, existing_results):
        if not isinstance(post, dict) or 'noi_dung_bai_viet' not in post or 'id_bai_viet' not in post:
            print(f"[{uuid.uuid4()}] Error: Invalid post data: {post}")
            return False

        try:
            post_id = post['id_bai_viet']
            comment_id = comment['id_binh_luan'] if comment and isinstance(comment,
                                                                           dict) and 'id_binh_luan' in comment else "0"
            item_key = (post_id, comment_id)

            if item_key in self.processed_items:
                return False

            if comment and (not isinstance(comment, dict) or 'noi_dung_binh_luan' not in comment):
                print(f"[{uuid.uuid4()}] Error: Invalid comment data: {comment}")
                return False

            content = comment['noi_dung_binh_luan'] if comment else post['noi_dung_bai_viet']
            if not content or len(content.split()) < 3:
                print(f"[{uuid.uuid4()}] Warning: Empty or too short content for post {post_id}, comment {comment_id}")
                return False

            classifications = self.classify_text(content)

            has_changes = False

            for dept_id, similarity in classifications.items():
                if similarity >= self.similarity_threshold * 100:
                    data = {
                        "id_bai_viet": post_id,
                        "id_binh_luan": comment_id,
                        "id_don_vi": dept_id,
                        "phan_tram_lien_quan": str(round(similarity, 2))
                    }
                    try:
                        response = requests.post(self.result_api, json=data)
                        if response.status_code in [200, 201]:
                            print(
                                f"[{uuid.uuid4()}] Success: Processed result for post {post_id}, comment {comment_id}, department {dept_id}")
                            has_changes = True
                        else:
                            print(
                                f"[{uuid.uuid4()}] Error: Failed to process result, status code: {response.status_code}, response: {response.text}")
                    except Exception as e:
                        print(f"[{uuid.uuid4()}] Error processing result: {str(e)}")

            if has_changes:
                self.processed_items.add(item_key)
            return has_changes
        except Exception as e:
            print(f"[{uuid.uuid4()}] Error processing item: {str(e)}")
            return False

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

    def update_keywords(self):
        self.departments = self._load_keywords()
        self.department_embeddings = self._calculate_department_embeddings()
        return self.process_content()


def main():
    try:
        print(f"[{uuid.uuid4()}] Initializing Department Classifier...")
        classifier = DepartmentClassifierAPI()

        max_pages = 10
        page = 1
        while page <= max_pages:
            print(f"[{uuid.uuid4()}] Processing content (page {page})...")
            has_new_data = classifier.process_content(page=page)

            if not has_new_data:
                print(f"[{uuid.uuid4()}] Info: No new data to process in page {page}. Moving to next page or stopping.")
                page += 1
                continue

            print(f"[{uuid.uuid4()}] Checking for keyword updates...")
            has_new_data = classifier.update_keywords()

            if not has_new_data:
                print(f"[{uuid.uuid4()}] Info: No new data after keyword update in page {page}. Moving to next page.")
                page += 1
                continue

            print(f"[{uuid.uuid4()}] Waiting for next cycle...")
            time.sleep(300)
            page += 1

        print(f"[{uuid.uuid4()}] Script completed.")
    except Exception as e:
        print(f"[{uuid.uuid4()}] An error occurred: {str(e)}")


if __name__ == "__main__":
    main()