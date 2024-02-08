import sqlite3


class DatabaseManager:
    def __init__(self, db_filename):
        self.conn = sqlite3.connect(db_filename)

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master WHERE type='table' AND name='students';
        """)
        result = cursor.fetchone()
        if result:
            # Table already exists
            print("Table 'students' already exists.")
        else:
            # Table does not exist, create it
            cursor.execute("""
                CREATE TABLE students (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    grade INTEGER NOT NULL
                )
            """)
            self.conn.commit()

        cursor.execute("""
            SELECT name FROM sqlite_master WHERE type='table' AND name='predicts';
        """)
        result = cursor.fetchone()
        if result:
            # Table already exists
            print("Table 'predicts' already exists.")
        else:
            # Table does not exist, create it
            cursor.execute("""
                CREATE TABLE predicts (
                    id INTEGER PRIMARY KEY,
                    student_video_path TEXT NOT NULL,
                    teacher_video_path TEXT NOT NULL,
                    model_name TEXT NOT NULL
                )
            """)
            self.conn.commit()

        cursor.execute("""
            SELECT name FROM sqlite_master WHERE type='table' AND name='outputs';
        """)
        result = cursor.fetchone()
        if result:
            # Table already exists
            print("Table 'outputs' already exists.")
        else:
            # Table does not exist, create it
            cursor.execute("""
                CREATE TABLE outputs (
                    id INTEGER PRIMARY KEY,
                    predict_id INTEGER NOT NULL,
                    output_path TEXT NOT NULL                    
                )
            """)
            self.conn.commit()

        cursor.execute("""
            SELECT name FROM sqlite_master WHERE type='table' AND name='logs';
        """)
        result = cursor.fetchone()
        if result:
            # Table already exists
            print("Table 'logs' already exists.")
        else:
            # Table does not exist, create it
            cursor.execute("""
                CREATE TABLE logs (
                    id INTEGER PRIMARY KEY,
                    predict_id INTEGER NOT NULL,
                    output_id INTEGER NOT NULL,
                    log_index INTEGER NOT NULL,
                    log TEXT NOT NULL                 
                )
            """)
            self.conn.commit()

        cursor.execute("""
            SELECT name FROM sqlite_master WHERE type='table' AND name='scores';
        """)
        result = cursor.fetchone()
        if result:
            # Table already exists
            print("Table 'scores' already exists.")
        else:
            # Table does not exist, create it
            cursor.execute("""
                CREATE TABLE scores (
                    id INTEGER PRIMARY KEY,
                    predict_id INTEGER NOT NULL,
                    output_id INTEGER NOT NULL,
                    ai_score DOUBLE NOT NULL,
                    timing_score DOUBLE NOT NULL,
                    velocity_score DOUBLE NOT NULL,
                    accuracy_score DOUBLE NOT NULL             
                )
            """)
            self.conn.commit()
            
        cursor.execute("""
            SELECT name FROM sqlite_master WHERE type='table' AND name='section_scores';
        """)
        result = cursor.fetchone()
        if result:
            # Table already exists
            print("Table 'section_scores' already exists.")
        else:
            # Table does not exist, create it
            cursor.execute("""
                CREATE TABLE section_scores (
                    id INTEGER PRIMARY KEY,
                    score_id INTEGER NOT NULL,
                    section_name TEXT NOT NULL,
                    ai_score DOUBLE NOT NULL,
                    timing_score DOUBLE NOT NULL,
                    velocity_score DOUBLE NOT NULL,
                    accuracy_score DOUBLE NOT NULL                    
                )
            """)
            self.conn.commit()

    def insert_predict(self, student_video_path, teacher_video_path, model_name):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO predicts (student_video_path, teacher_video_path, model_name) VALUES (?, ?, ?)
        """, (student_video_path, teacher_video_path, model_name))
        self.conn.commit()
        # Return the id of the inserted row
        return cursor.lastrowid

    def get_predicts(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM predicts
        """)
        return cursor.fetchall()

    def get_predicts_by_id(self, id):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM predicts WHERE id = ?
        """, (id,))
        return cursor.fetchone()

    def get_predicts_by_model_name(self, model_name):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM predicts WHERE model_name = ?
        """, (model_name,))
        return cursor.fetchall()
    
    def delete_predicts_by_id(self, id):
        cursor = self.conn.cursor()
        # Find all scores associated with this predict
        scores = self.get_scores_by_predict_id(id)
        for score in scores:
            self.delete_score_by_id(score[0])        
        
        cursor.execute("""
            DELETE FROM predicts WHERE id = ?
        """, (id,))
        self.conn.commit()
        return cursor.lastrowid

    def insert_output(self, predict_id, output_path):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO outputs (predict_id, output_path) VALUES (?, ?)
        """, (predict_id, output_path))
        self.conn.commit()
        return cursor.lastrowid

    def get_outputs(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM outputs
        """)
        return cursor.fetchall()

    def get_outputs_by_id(self, id):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM outputs WHERE id = ?
        """, (id,))
        return cursor.fetchone()

    def get_outputs_by_predict_id(self, predict_id):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM outputs WHERE predict_id = ?
        """, (predict_id,))
        return cursor.fetchall()

    def get_outputs_by_output_path(self, output_path):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM outputs WHERE output_path = ?
        """, (output_path,))
        return cursor.fetchall()

    def insert_log(self, predict_id, output_id, log_index, log):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO logs (predict_id, output_id, log_index, log) VALUES (?, ?, ?, ?)
        """, (predict_id, output_id, log_index, log))
        self.conn.commit()
        return cursor.lastrowid

    def get_logs(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM logs
        """)
        return cursor.fetchall()

    def get_logs_by_id(self, id):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM logs WHERE id = ?
        """, (id,))
        return cursor.fetchone()

    def get_logs_by_predict_id(self, predict_id):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM logs WHERE predict_id = ?
        """, (predict_id,))
        return cursor.fetchall()

    def get_logs_by_output_id(self, output_id):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM logs WHERE output_id = ?
        """, (output_id,))
        return cursor.fetchall()
    
    def get_logs_by_log_index(self, log_index):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM logs WHERE log_index = ?
        """, (log_index,))
        return cursor.fetchall()
    
    def insert_score(self, predict_id, output_id, ai_score, timing_score, velocity_score, accuracy_score):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO scores (predict_id, output_id, ai_score, timing_score, velocity_score, accuracy_score) VALUES (?, ?, ?, ?, ?, ?)
        """, (predict_id, output_id, ai_score, timing_score, velocity_score, accuracy_score))                    
        self.conn.commit()
        return cursor.lastrowid
    
    def get_scores(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM scores
        """)
        return cursor.fetchall()
    
    def get_scores_by_id(self, id):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM scores WHERE id = ?
        """, (id,))
        return cursor.fetchone()

    def update_score(self, id, ai_score, timing_score, velocity_score, accuracy_score):
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE scores SET ai_score = ?, timing_score = ?, velocity_score = ?, accuracy_score = ? WHERE id = ?
        """, (ai_score, timing_score, velocity_score, accuracy_score, id))        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_scores_by_predict_id(self, predict_id):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM scores WHERE predict_id = ?
        """, (predict_id,))
        return cursor.fetchall()
    
    def delete_score_by_id(self, id):
        cursor = self.conn.cursor()
        # Find all section scores associated with this score
        section_scores = self.get_section_scores_by_score_id(id)
        for section_score in section_scores:
            self.delete_section_score_by_id(section_score[0])
            
        cursor.execute("""
            DELETE FROM scores WHERE id = ?
        """, (id,))
        self.conn.commit()
        return cursor.lastrowid
    
    def delete_section_score_by_id(self, id):
        cursor = self.conn.cursor()
        cursor.execute("""
            DELETE FROM section_scores WHERE id = ?
        """, (id,))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_scores_by_output_id(self, output_id):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM scores WHERE output_id = ?
        """, (output_id,))
        return cursor.fetchall()

    def insert_student(self, name, grade):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO students (name, grade) VALUES (?, ?)
        """, (name, grade))
        self.conn.commit()
        return cursor.lastrowid

    def get_students(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM students
        """)
        return cursor.fetchall()

    def close(self):
        self.conn.close()
        
    def insert_section_score(self, score_id, section_name, ai_score, timing_score, velocity_score, accuracy_score):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO section_scores (score_id, section_name, ai_score, timing_score, velocity_score, accuracy_score) VALUES (?, ?, ?, ?, ?, ?)
        """, (score_id, section_name, ai_score, timing_score, velocity_score, accuracy_score))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_section_scores(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM section_scores
        """)
        return cursor.fetchall()
    
    def get_section_scores_by_id(self, id):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM section_scores WHERE id = ?
        """, (id,))
        return cursor.fetchone()
    
    def get_section_scores_by_score_id(self, score_id):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM section_scores WHERE score_id = ?
        """, (score_id,))
        return cursor.fetchall()   
    
    def update_section_score(self, id, ai_score, timing_score, velocity_score, accuracy_score):
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE section_scores SET ai_score = ?, timing_score = ?, velocity_score = ?, accuracy_score = ? WHERE id = ?
        """, (ai_score, timing_score, velocity_score, accuracy_score, id))        
        self.conn.commit()
        return cursor.lastrowid 
