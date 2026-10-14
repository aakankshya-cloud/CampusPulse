"""SQL demo analytics. Only generated fictional rows may be persisted here."""
import sqlite3
from contextlib import closing
from .data import synthetic_data

def demo_summary():
    with closing(sqlite3.connect(':memory:')) as connection:
        synthetic_data().to_sql('demo_weeks', connection, index=False)
        rows = connection.execute('''
            SELECT week, COUNT(DISTINCT student_id),
                   ROUND(AVG(sleep_hours), 2), ROUND(AVG(social_minutes), 1),
                   ROUND(AVG(loneliness_score), 1)
            FROM demo_weeks GROUP BY week ORDER BY week
        ''').fetchall()
    return [{'week': r[0], 'students': r[1], 'sleep': r[2], 'social': r[3],
             'score': r[4]} for r in rows]
