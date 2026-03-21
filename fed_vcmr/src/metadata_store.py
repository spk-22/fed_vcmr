import sqlite3
import os

class MetadataStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Videos table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                path TEXT,
                duration REAL,
                split TEXT
            )
        ''')
        
        # Chunks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                video_id TEXT,
                t_start REAL,
                t_end REAL,
                scale REAL,
                cache_path TEXT,
                FOREIGN KEY (video_id) REFERENCES videos (video_id)
            )
        ''')
        
        # Captions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS captions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT,
                caption TEXT,
                FOREIGN KEY (video_id) REFERENCES videos (video_id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def add_video(self, video_id, path, duration, split):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('INSERT OR REPLACE INTO videos VALUES (?, ?, ?, ?)', (video_id, path, duration, split))
        conn.commit()
        conn.close()

    def add_chunk(self, chunk_id, video_id, t_start, t_end, scale, cache_path):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?, ?)', 
                      (chunk_id, video_id, t_start, t_end, scale, cache_path))
        conn.commit()
        conn.close()

    def add_caption(self, video_id, caption):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO captions (video_id, caption) VALUES (?, ?)', (video_id, caption))
        conn.commit()
        conn.close()
