# federated_learning/sharding.py
"""
Non-IID Categorical Sharding for Federated Learning.
Partitions the MSR-VTT training set (excl. 1K-A test) into 4 client shards.
"""
import sqlite3, random
from collections import Counter
import numpy as np

def shard_data():
    # 1. Load test IDs to exclude
    test_ids = set(l.strip() for l in open('MSRVTT/MSRVTT/structured-symlinks/test_list_miech.txt'))
    print(f"Loaded {len(test_ids)} test IDs to exclude from training shards.")

    # 2. Connect to DB and fetch training candidates with categories
    conn = sqlite3.connect('fedvcmr.db')
    # Use JOIN for performance! Subqueries in select are slow on large tables.
    query = """
    SELECT v.video_id, c.caption 
    FROM videos v
    LEFT JOIN captions c ON v.video_id = c.video_id
    GROUP BY v.video_id
    """
    rows = conn.execute(query).fetchall()
    
    train_data = []
    for vid, cap in rows:
        if vid not in test_ids:
            # Simple keyword-based 'category' proxy for non-IID skew
            cat = 'General'
            if any(w in cap.lower() for w in ['minecraft', 'game', 'gaming', 'play', 'xbox', 'nintendo']):
                cat = 'Gaming'
            elif any(w in cap.lower() for w in ['sports', 'soccer', 'football', 'basketball', 'baseball']):
                cat = 'Sports'
            elif any(w in cap.lower() for w in ['cooking', 'food', 'chef', 'kitchen', 'recipe']):
                cat = 'Cooking'
            elif any(w in cap.lower() for w in ['talk', 'news', 'interview', 'discuss', 'speak']):
                cat = 'News'
            
            train_data.append((vid, cat))
    
    print(f"Found {len(train_data)} training videos.")
    
    # 3. Create Shards (Non-IID)
    # We want Specific clients to have specific categories
    clients = {0: [], 1: [], 2: [], 3: []}
    
    # Strong Skew:
    # Client 0: 80% Gaming
    # Client 1: 80% Sports
    # Client 2: 80% Cooking
    # Client 3: 80% News
    # Others are distributed randomly.
    
    category_map = {}
    for vid, cat in train_data:
        category_map.setdefault(cat, []).append(vid)
        
    for cat, vids in category_map.items():
        random.shuffle(vids)
        if cat == 'Gaming':
            p = [0.8, 0.06, 0.07, 0.07]
        elif cat == 'Sports':
            p = [0.06, 0.8, 0.07, 0.07]
        elif cat == 'Cooking':
            p = [0.06, 0.07, 0.8, 0.07]
        elif cat == 'News':
            p = [0.06, 0.07, 0.07, 0.8]
        else:
            p = [0.25, 0.25, 0.25, 0.25]
            
        # Assign based on probabilities
        for vid in vids:
            cid = np.random.choice(4, p=p)
            clients[cid].append(vid)
            
    # 4. Save to DB
    conn.execute('DROP TABLE IF EXISTS fl_shards')
    conn.execute('CREATE TABLE fl_shards (video_id TEXT PRIMARY KEY, client_id INTEGER)')
    
    insert_data = []
    for cid, vids in clients.items():
        for vid in vids:
            insert_data.append((vid, cid))
    
    conn.executemany('INSERT INTO fl_shards VALUES (?, ?)', insert_data)
    conn.commit()
    conn.close()
    
    # 5. Summary
    print("\n--- Federated Sharding Summary ---")
    for cid, vids in clients.items():
        print(f"Client {cid}: {len(vids)} videos")
        # Sample cats
        sample_cats = [c for v, c in train_data if v in vids]
        counts = Counter(sample_cats)
        print(f"  Distribution: {dict(counts.most_common(5))}")

if __name__ == '__main__':
    shard_data()
