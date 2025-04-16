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
import re


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
        self.embedding_cache = {}  # Cache for embeddings
        self.similarity_threshold = 0.65

    def _load_processed_items(self):
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
        try:
            with open("processed_items.json", "w") as f:
                data = [{"post_id": p, "comment_id": c} for p, c in self.processed_items]
                json.dump(data, f)
            print(f"[{uuid.uuid4()}] Saved {len(self.processed_items)} processed items.")
        except Exception as e:
            print(f"[{uuid.uuid4()}] Error saving processed items: {str(e)}")

    def _load_keywords(self):
        try:
            response = requests.get(self.dictionary_api, timeout=10)
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
                keywords = item['tu_khoa'].strip()
                if keywords and len(keywords.split()) >= 2:
                    departments[dept_id].append(keywords)
                else:
                    print(f"[{uuid.uuid4()}] Warning: Skipping short or empty keyword: {keywords} for dept {dept_id}")
            print(f"[{uuid.uuid4()}] Loaded keywords for {len(departments)} departments.")
            return departments
        except Exception as e:
            print(f"[{uuid.uuid4()}] Error loading keywords: {str(e)}")
            return {}

    def _calculate_department_embeddings(self):
        department_embeddings = {}
        for dept_id, keywords in self.departments.items():
            if not keywords:
                print(f"[{uuid.uuid4()}] Warning: No keywords for dept {dept_id}, skipping embedding.")
                continue
            text = " ".join(keywords)
            dept_emb = self._get_embeddings(text)
            if dept_emb is not None and dept_emb.shape[0] > 0:
                department_embeddings[dept_id] = dept_emb.mean(axis=0)
            else:
                print(f"[{uuid.uuid4()}] Warning: Failed to generate embedding for dept {dept_id}")
        print(f"[{uuid.uuid4()}] Calculated embeddings for {len(department_embeddings)} departments.")
        return department_embeddings

    def _get_embeddings(self, text):
        if not text or not isinstance(text, str) or len(text.split()) < 2:
            print(f"[{uuid.uuid4()}] Warning: Invalid or too short text input for embeddings: '{text}'")
            return np.zeros((1, 768))

        cache_key = hash(text)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        try:
            encoded = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
            with torch.no_grad():
                outputs = self.model(**encoded)
                embeddings = outputs.last_hidden_state.numpy()
                if len(embeddings) == 0:
                    print(f"[{uuid.uuid4()}] Warning: Empty embeddings generated for text: '{text}'")
                    return np.zeros((1, 768))
                self.embedding_cache[cache_key] = embeddings[0]
                return embeddings[0]
        except Exception as e:
            print(f"[{uuid.uuid4()}] Error generating embeddings: {str(e)}")
            return np.zeros((1, 768))

    def classify_text(self, text):
        if not text or not isinstance(text, str) or len(text.split()) < 2:
            print(f"[{uuid.uuid4()}] Warning: Invalid or too short text input for classification: '{text}'")
            return {}

        try:
            text_emb = self._get_embeddings(text)
            if text_emb is None or len(text_emb) == 0:
                print(f"[{uuid.uuid4()}] Warning: Empty embeddings for classification: '{text}'")
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

    def process_content(self, page=1, limit=500):
        existing_results = self._get_existing_results()
        if not isinstance(existing_results, list):
            print(f"[{uuid.uuid4()}] Error: Existing results is not a list, skipping processing")
            return False

        # Fetch posts
        try:
            posts_response = requests.get(self.post_api, params={"page": page, "limit": limit}, timeout=15)
            if posts_response.status_code != 200:
                print(f"[{uuid.uuid4()}] Error: Failed to fetch posts, status code: {posts_response.status_code}")
                return False
            posts = posts_response.json()
            if not isinstance(posts, list):
                print(f"[{uuid.uuid4()}] Error: Posts data is not a list, got: {posts}")
                return False
            print(f"[{uuid.uuid4()}] Retrieved {len(posts)} posts (page {page})")
        except Exception as e:
            print(f"[{uuid.uuid4()}] Error fetching posts: {str(e)}")
            return False

        # Fetch comments (try multiple pages if empty)
        comments = []
        max_comment_pages = 3  # Try up to 3 comment pages
        for comment_page in range(page, page + max_comment_pages):
            try:
                comments_response = requests.get(self.comment_api, params={"page": comment_page, "limit": limit},
                                                 timeout=15)
                if comments_response.status_code != 200:
                    print(
                        f"[{uuid.uuid4()}] Error: Failed to fetch comments (page {comment_page}), status code: {comments_response.status_code}")
                    continue
                page_comments = comments_response.json()
                if not isinstance(page_comments, list):
                    print(
                        f"[{uuid.uuid4()}] Error: Comments data is not a list (page {comment_page}), got: {page_comments}")
                    continue
                comments.extend(page_comments)
                print(f"[{uuid.uuid4()}] Retrieved {len(page_comments)} comments (comment page {comment_page})")
                if len(page_comments) > 0:
                    break  # Stop if we get comments
            except Exception as e:
                print(f"[{uuid.uuid4()}] Error fetching comments (page {comment_page}): {str(e)}")
                continue
        print(f"[{uuid.uuid4()}] Total comments retrieved: {len(comments)}")

        # Group comments by post
        post_comments = {post['id_bai_viet']: [] for post in posts if isinstance(post, dict) and 'id_bai_viet' in post}
        invalid_comments = 0
        for comment in comments:
            if isinstance(comment,
                          dict) and 'id_bai_viet' in comment and 'id_binh_luan' in comment and 'noi_dung_binh_luan' in comment:
                post_comments.setdefault(comment['id_bai_viet'], []).append(comment)
            else:
                invalid_comments += 1
                print(f"[{uuid.uuid4()}] Warning: Invalid comment data: {comment}")
        if invalid_comments > 0:
            print(f"[{uuid.uuid4()}] Warning: Found {invalid_comments} invalid comments.")

        # Log comment distribution
        for post_id, comments in post_comments.items():
            print(f"[{uuid.uuid4()}] Post {post_id} has {len(comments)} comments.")

        # Check for duplicates
        post_ids = {post['id_bai_viet'] for post in posts if isinstance(post, dict) and 'id_bai_viet' in post}
        processed_post_ids = {p for p, c in self.processed_items}
        overlap = post_ids & processed_post_ids
        duplicate_ratio = len(overlap) / max(len(post_ids), 1)
        print(
            f"[{uuid.uuid4()}] Duplicate check: {len(overlap)}/{len(post_ids)} duplicates (ratio: {duplicate_ratio:.2f})")
        if duplicate_ratio > 0.9:
            print(f"[{uuid.uuid4()}] Warning: High duplicate ratio ({duplicate_ratio:.2f}) in page {page}. Stopping.")
            return False

        has_new_data = False
        skipped_empty = 0
        skipped_duplicate = 0
        processed_count = 0
        for post in tqdm(posts, desc=f"Processing posts (page {page})"):
            if not isinstance(post, dict) or 'id_bai_viet' not in post:
                print(f"[{uuid.uuid4()}] Warning: Invalid post data: {post}")
                continue
            post_id = post['id_bai_viet']
            if post_id in post_comments:
                if self._process_item(post, None, existing_results):
                    processed_count += 1
                    has_new_data = True
                else:
                    if (post_id, '0') in self.processed_items:
                        skipped_duplicate += 1
                    else:
                        skipped_empty += 1
                for comment in tqdm(post_comments[post_id], desc=f"Processing comments for post {post_id}",
                                    leave=False):
                    if self._process_item(post, comment, existing_results):
                        processed_count += 1
                        has_new_data = True
                    else:
                        if (post_id, comment.get('id_binh_luan', '0')) in self.processed_items:
                            skipped_duplicate += 1
                        else:
                            skipped_empty += 1
            else:
                if self._process_item(post, None, existing_results):
                    processed_count += 1
                    has_new_data = True
                else:
                    if (post_id, '0') in self.processed_items:
                        skipped_duplicate += 1
                    else:
                        skipped_empty += 1

        print(
            f"[{uuid.uuid4()}] Finished page {page}. New items processed: {processed_count}, Skipped (duplicate): {skipped_duplicate}, Skipped (empty): {skipped_empty}")
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
                print(f"[{uuid.uuid4()}] Skipping processed item: post {post_id}, comment {comment_id}")
                return False

            if comment and (not isinstance(comment, dict) or 'noi_dung_binh_luan' not in comment):
                print(f"[{uuid.uuid4()}] Error: Invalid comment data: {comment}")
                return False

            content = comment['noi_dung_binh_luan'] if comment else post['noi_dung_bai_viet']
            content = re.sub(r'<[^>]+>', '', content).strip()
            if not content or len(content.split()) < 2:
                print(
                    f"[{uuid.uuid4()}] Warning: Empty or too short content for post {post_id}, comment {comment_id}, content: '{content}'")
                with open("skipped_content.log", "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now()}] Post {post_id}, Comment {comment_id}: '{content}'\n")
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
                        response = requests.post(self.result_api, json=data, timeout=15)
                        if response.status_code in [200, 201]:
                            print(
                                f"[{uuid.uuid4()}] Success: Processed result for post {post_id}, comment {comment_id}, department {dept_id}, similarity {round(similarity, 2)}%")
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
            response = requests.get(self.result_api, timeout=15)
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
            start_time = time.time()
            has_new_data = classifier.process_content(page=page, limit=500)
            elapsed_time = time.time() - start_time

            if not has_new_data:
                print(
                    f"[{uuid.uuid4()}] Info: No new data to process in page {page}. Moving to next page or stopping. Elapsed time: {elapsed_time:.2f}s")
                page += 1
                continue

            print(f"[{uuid.uuid4()}] Checking for keyword updates...")
            has_new_data = classifier.update_keywords()

            if not has_new_data:
                print(
                    f"[{uuid.uuid4()}] Info: No new data after keyword update in page {page}. Moving to next page. Elapsed time: {elapsed_time:.2f}s")
                page += 1
                continue

            print(f"[{uuid.uuid4()}] Waiting for next cycle (300s)...")
            time.sleep(300)
            page += 1

        print(f"[{uuid.uuid4()}] Script completed.")
    except Exception as e:
        print(f"[{uuid.uuid4()}] An error occurred: {str(e)}")


if __name__ == "__main__":
    main()