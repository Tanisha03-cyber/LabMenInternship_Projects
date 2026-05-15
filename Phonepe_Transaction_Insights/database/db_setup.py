import mysql.connector
from mysql.connector import Error

# ── DB CONFIG ────────────────────────────────────────────────────────
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Rajkumargoyal787..!!"   # 
}

DB_NAME = "phonepe_pulse"

def get_connection(with_db=True):
    config = DB_CONFIG.copy()
    if with_db:
        config["database"] = DB_NAME
    return mysql.connector.connect(**config)

def create_database():
    conn = get_connection(with_db=False)
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
    print(f"Database '{DB_NAME}' created successfully.")
    cursor.close()
    conn.close()

def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    tables = {
        "agg_transaction": """
            CREATE TABLE IF NOT EXISTS agg_transaction (
                id INT AUTO_INCREMENT PRIMARY KEY,
                state VARCHAR(100),
                year INT,
                quarter INT,
                transaction_type VARCHAR(100),
                transaction_count BIGINT,
                transaction_amount DOUBLE
            )""",

        "agg_user": """
            CREATE TABLE IF NOT EXISTS agg_user (
                id INT AUTO_INCREMENT PRIMARY KEY,
                state VARCHAR(100),
                year INT,
                quarter INT,
                brand VARCHAR(100),
                registered_users BIGINT,
                percentage DOUBLE
            )""",

        "agg_insurance": """
            CREATE TABLE IF NOT EXISTS agg_insurance (
                id INT AUTO_INCREMENT PRIMARY KEY,
                state VARCHAR(100),
                year INT,
                quarter INT,
                insurance_type VARCHAR(100),
                insurance_count BIGINT,
                insurance_amount DOUBLE
            )""",

        "map_transaction": """
            CREATE TABLE IF NOT EXISTS map_transaction (
                id INT AUTO_INCREMENT PRIMARY KEY,
                state VARCHAR(100),
                year INT,
                quarter INT,
                district VARCHAR(100),
                transaction_count BIGINT,
                transaction_amount DOUBLE
            )""",

        "map_user": """
            CREATE TABLE IF NOT EXISTS map_user (
                id INT AUTO_INCREMENT PRIMARY KEY,
                state VARCHAR(100),
                year INT,
                quarter INT,
                district VARCHAR(100),
                registered_users BIGINT,
                app_opens BIGINT
            )""",

        "map_insurance": """
            CREATE TABLE IF NOT EXISTS map_insurance (
                id INT AUTO_INCREMENT PRIMARY KEY,
                state VARCHAR(100),
                year INT,
                quarter INT,
                district VARCHAR(100),
                insurance_count BIGINT,
                insurance_amount DOUBLE
            )""",

        "top_transaction": """
            CREATE TABLE IF NOT EXISTS top_transaction (
                id INT AUTO_INCREMENT PRIMARY KEY,
                state VARCHAR(100),
                year INT,
                quarter INT,
                entity_level VARCHAR(50),
                entity_name VARCHAR(100),
                transaction_count BIGINT,
                transaction_amount DOUBLE
            )""",

        "top_user": """
            CREATE TABLE IF NOT EXISTS top_user (
                id INT AUTO_INCREMENT PRIMARY KEY,
                state VARCHAR(100),
                year INT,
                quarter INT,
                entity_level VARCHAR(50),
                entity_name VARCHAR(100),
                registered_users BIGINT
            )""",

        "top_insurance": """
            CREATE TABLE IF NOT EXISTS top_insurance (
                id INT AUTO_INCREMENT PRIMARY KEY,
                state VARCHAR(100),
                year INT,
                quarter INT,
                entity_level VARCHAR(50),
                entity_name VARCHAR(100),
                insurance_count BIGINT,
                insurance_amount DOUBLE
            )"""
    }

    for table_name, ddl in tables.items():
        cursor.execute(ddl)
        print(f"Table '{table_name}' created successfully.")

    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    create_database()
    create_tables()
    print("\n All 9 tables ready in MySQL!")
