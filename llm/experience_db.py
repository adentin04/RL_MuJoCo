"""SQLite experience database — logs every episode for analysis."""

import sqlite3
from datetime import datetime


class ExperienceDB:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                instruction     TEXT,
                target          TEXT,
                total_reward    REAL,
                steps           INTEGER,
                success         INTEGER,
                final_distance  REAL
            )
        """)
        self.conn.commit()

    def log_episode(
        self, instruction, target, total_reward, steps, success, final_distance
    ):
        self.conn.execute(
            """INSERT INTO episodes
               (timestamp, instruction, target, total_reward, steps, success, final_distance)
               VALUES (?,?,?,?,?,?,?)""",
            (
                datetime.now().isoformat(),
                instruction,
                target,
                float(total_reward),
                int(steps),
                int(bool(success)),
                float(final_distance),
            ),
        )
        self.conn.commit()

    def stats(self, instruction=None):
        """Return aggregate stats, optionally filtered by instruction."""
        if instruction:
            row = self.conn.execute(
                "SELECT COUNT(*), AVG(total_reward), AVG(success), AVG(final_distance) "
                "FROM episodes WHERE instruction = ?",
                (instruction,),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*), AVG(total_reward), AVG(success), AVG(final_distance) "
                "FROM episodes",
            ).fetchone()
        return {
            "episodes": row[0],
            "avg_reward": row[1],
            "success_rate": row[2],
            "avg_distance": row[3],
        }

    def recent_failures(self, limit=10):
        """Return the most recent failed episodes."""
        return self.conn.execute(
            "SELECT id, instruction, total_reward, final_distance, timestamp "
            "FROM episodes WHERE success = 0 ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def close(self):
        self.conn.close()
