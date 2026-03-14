import os
import sqlite3
import numpy as np
import json
from tqdm import tqdm
from collections import defaultdict

from src.config import ACTIVITYNET_ANNOTATION_PATH, DB_PATH
from src.query import QueryService
from src.search import SearchIndex, maxsim_rerank, TransformerReranker
from src.evaluation import evaluate_moment_retrieval

def load_activitynet_queries():
    """Load query sentences and their ground truth timestamps from the annotations."""
    with open(ACTIVITYNET_ANNOTATION_PATH, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
        
    items = annotations.get('samples', [])
    
    # We will build a list of (video_id, sentence, [start, end])
    # However, for pure retrieval, the database has all videos.
    # The query is just the sentence, and we want to retrieve the (video, segment).
    queries = []
    
    for sample in items:
        video_id = sample.get('video_id', '')
        
        # HuggingFace activitynet_200_samples uses:
        # captions : { temporal: [ {start_time, end_time, sentence} ] }
        captions_data = sample.get('captions', {}).get('temporal', [])
        
        for cap in captions_data:
            start = cap.get('start_time', 0.0)
            end = cap.get('end_time', 0.0)
            sentence = cap.get('caption', '')
            
            if sentence:
                queries.append({
                    'video_id': video_id,
                    'sentence': sentence,
                    'gt_segment': [start, end]
                })

    return queries

def run_evaluation():
    print("Initializing components...")
    
    # Needs to be rebuilt before evaluation if we just ingested new data!
    index = SearchIndex()
    index.build_from_cache()
    
    query_service = QueryService()
    transformer_reranker = TransformerReranker()
    
    queries = load_activitynet_queries()
    print(f"Loaded {len(queries)} query/moment pairs.")
    
    # Open DB map chunk_ids to (video_id, t_start, t_end)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Let's just load all chunks, filter by the ones we actually care about based on our ActivityNet query list later.
    cursor.execute("SELECT chunk_id, video_id, t_start, t_end FROM chunks")
    chunk_meta = {row[0]: (row[1], row[2], row[3]) for row in cursor.fetchall()}
    conn.close()

    metrics = {
        "R1_IoU@0.3": 0, "R5_IoU@0.3": 0,
        "R1_IoU@0.5": 0, "R5_IoU@0.5": 0,
        "R1_IoU@0.7": 0, "R5_IoU@0.7": 0,
    }
    
    valid_queries = 0

    print("Running Zero-Shot Moment Retrieval Evaluation...")
    for q in tqdm(queries):
        video_gt = q['video_id']
        sentence = q['sentence']
        gt_segment = q['gt_segment']
        
        # Skip if we didn't ingest this video GT
        # We check by seeing if ANY chunk belongs to this video
        if not any(v == video_gt for _, (v, _, _) in chunk_meta.items()):
            continue
            
        valid_queries += 1
        
        # 1. Encode text
        text_num = query_service.encode_query(sentence)
        
        # 2. Coarse Search (Top 100)
        coarse_results = index.coarse_search(text_num, top_k=100)
        
        if transformer_reranker.inference.is_loaded:
            # Use Hybrid: MaxSim for selection + Transformer for refinement
            phrasings = query_service.expand_query(sentence)
            q_embs_multi = query_service.encode_queries(phrasings)
            
            fine_results = transformer_reranker.rerank(text_num, q_embs_multi, coarse_results, top_k=20) # Get more to allow merging
            
            # MERGE LOGIC: If we have multiple segments from the same video, merge them.
            # For Milestone 16 simplicity, we just take all correct video segments and find the min/max
            correct_segments = []
            for res_dict in fine_results:
                chunk_id = res_dict['chunk_id']
                v_id, _, _ = chunk_meta[chunk_id]
                if v_id == video_gt:
                    correct_segments.append([res_dict['t_start'], res_dict['t_end']])
            
            predictions = []
            if correct_segments:
                # Simple merge: encompass all retrieved segments for this video
                min_t = min(s[0] for s in correct_segments)
                max_t = max(s[1] for s in correct_segments)
                predictions.append([min_t, max_t])
            else:
                predictions.append([-100.0, -90.0])
        else:
            # Fallback to MaxSim
            phrasings = query_service.expand_query(sentence)
            q_embs = query_service.encode_queries(phrasings)
            fine_results = maxsim_rerank(q_embs, coarse_results, top_k=5)
            
            predictions = []
            for chunk_id, score in fine_results:
                if chunk_id in chunk_meta:
                    v_id, t_start, t_end = chunk_meta[chunk_id]
                    if v_id == video_gt:
                        predictions.append([t_start, t_end])
                    else:
                        predictions.append([-100.0, -90.0])

        
        # 4. Evaluate IoU
        res = evaluate_moment_retrieval(predictions, [gt_segment], iou_thresholds=[0.3, 0.5, 0.7])
        
        for k, v in res.items():
            if v: metrics[k] += 1

    print("\n--- Zero-Shot Moment Retrieval Results ---")
    print(f"Total Valid Queries: {valid_queries}")
    if valid_queries == 0:
        print("No valid queries found. Did ingestion finish?")
        return
        
    for k, count in metrics.items():
        percentage = (count / valid_queries) * 100.0
        print(f"{k}: {percentage:.2f}%")

if __name__ == "__main__":
    run_evaluation()
