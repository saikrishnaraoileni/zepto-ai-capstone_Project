# queries.py
# Step 4 - run some SQL queries against books.db to practice SELECT/WHERE,
# ORDER BY, LIMIT, DISTINCT, BETWEEN and a JOIN. Then double check the JOIN
# result also works if I do it in pandas instead of SQL.

import sqlite3
import pandas as pd

conn = sqlite3.connect("books.db")

# --- Query 1: SELECT + WHERE ---
print("\n--- Query 1: in-stock books over 1000 INR ---")
q1 = "SELECT title, price_inr FROM books WHERE in_stock = 1 AND price_inr > 1000"
result1 = conn.execute(q1).fetchall()
for row in result1[:10]:  # just print first 10 so it doesn't flood the terminal
    print(row)
print("total rows:", len(result1))

# --- Query 2: ORDER BY + LIMIT ---
print("\n--- Query 2: 10 cheapest books ---")
q2 = "SELECT title, price_inr FROM books ORDER BY price_inr ASC LIMIT 10"
result2 = conn.execute(q2).fetchall()
for row in result2:
    print(row)

# --- Query 3: DISTINCT ---
print("\n--- Query 3: what rating values exist ---")
q3 = "SELECT DISTINCT rating FROM books ORDER BY rating"
result3 = conn.execute(q3).fetchall()
print(result3)

# --- Query 4: BETWEEN ---
print("\n--- Query 4: books priced between 500 and 1500 INR ---")
q4 = "SELECT title, price_inr FROM books WHERE price_inr BETWEEN 500 AND 1500"
result4 = conn.execute(q4).fetchall()
for row in result4:
    print(row)
print("total rows:", len(result4))

# --- Query 5: JOIN ---
print("\n--- Query 5: 4 and 5 star books, joined with category name ---")
q5 = """
    SELECT categories.category_name, books.title, books.rating
    FROM books
    JOIN categories ON books.category_id = categories.category_id
    WHERE books.rating >= 4
    ORDER BY categories.category_name, books.rating DESC
"""
result5 = conn.execute(q5).fetchall()
for row in result5[:10]:
    print(row)
print("total rows:", len(result5))


# --- now do 2 of these again but with pandas instead of raw sqlite3 ---
print("\n--- same as Query 1 but using pd.read_sql ---")
df_q1 = pd.read_sql(q1, conn)
print(df_q1.head())

print("\n--- same as Query 5 but using pd.read_sql ---")
df_q5_sql = pd.read_sql(q5, conn)
print(df_q5_sql.head())


# --- and now try to get the SAME result as query 5 but doing the join
# ourselves in pandas with merge() instead of asking sqlite to do it ---
print("\n--- doing the join manually with pandas merge() ---")
books_df = pd.read_sql("SELECT * FROM books", conn)
categories_df = pd.read_sql("SELECT * FROM categories", conn)

merged = pd.merge(books_df, categories_df, on="category_id")
merged = merged[merged["rating"] >= 4]
merged = merged[["category_name", "title", "rating"]]
merged = merged.sort_values(["category_name", "rating"], ascending=[True, False])

print(merged.head())

# quick check that both approaches give the same result
# (sorting both the same way first so the comparison is fair)
sql_version = df_q5_sql.sort_values(["category_name", "title"]).reset_index(drop=True)
merge_version = merged.sort_values(["category_name", "title"]).reset_index(drop=True)

print("\ndo the SQL join and the pandas merge match?", sql_version.equals(merge_version))

conn.close()
