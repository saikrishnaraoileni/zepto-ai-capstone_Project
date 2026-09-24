# build_db.py
# Step 3 - take the cleaned csv and put it into an actual SQLite database
# with 2 tables so we get practice with a normalized schema (PK/FK)

import sqlite3
import pandas as pd

df = pd.read_csv("books_clean.csv")

# connect (this creates the file books.db if it doesn't exist yet)
conn = sqlite3.connect("books.db")
cur = conn.cursor()

# drop tables first in case we run this script more than once, so we dont
# get duplicate data every time
cur.execute("DROP TABLE IF EXISTS books")
cur.execute("DROP TABLE IF EXISTS categories")

# categories table - just an id and a name
cur.execute("""
    CREATE TABLE categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE
    )
""")

# books table - has a category_id column that points back to categories table
# this is the "foreign key" part of the normalized schema
cur.execute("""
    CREATE TABLE books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        price_gbp REAL,
        price_inr REAL,
        rating INTEGER,
        in_stock INTEGER,
        category_id INTEGER,
        FOREIGN KEY (category_id) REFERENCES categories(category_id)
    )
""")

conn.commit()

# first insert all the unique category names
unique_categories = df["category"].unique()
for cat_name in unique_categories:
    cur.execute("INSERT INTO categories (category_name) VALUES (?)", (cat_name,))

conn.commit()

# now build a little lookup so we know which category_id belongs to which name
cur.execute("SELECT category_id, category_name FROM categories")
rows = cur.fetchall()
category_id_lookup = {}
for cid, cname in rows:
    category_id_lookup[cname] = cid

# now insert every book, looking up its category_id as we go
for index, row in df.iterrows():
    cat_id = category_id_lookup[row["category"]]
    cur.execute(
        """INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (row["title"], row["price_gbp"], row["price_inr"], int(row["rating"]), int(row["in_stock"]), cat_id)
    )

conn.commit()

print("done! categories table has", len(unique_categories), "rows")
print("books table has", len(df), "rows")

conn.close()
