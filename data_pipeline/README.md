# Module 1 - Data Pipeline

What this does: scrapes books from books.toscrape.com, cleans the messy
text fields into real numbers, converts price to INR using a fixed rate,
puts everything into a small SQLite database with 2 linked tables, and
runs some SQL queries on it.

## My files
- `scrape.py` - grabs books from 5 categories, saves raw data to books_raw.csv
- `clean.py` - turns the price/rating/availability text into real
  float/int/bool columns, adds price_inr, drops any row that doesn't parse
- `build_db.py` - builds books.db with a categories table and a books table
  (books table has a category_id that points back to categories)
- `queries.py` - runs the 5 required SQL queries + checks pandas merge
  gives the same answer as the SQL join
- `run_all.py` - just runs all 4 scripts one after another so I don't have
  to keep typing commands

## Notes to self / decisions I made
- conversion rate: 1 GBP = 105.50 INR (this is fixed like the assignment
  says, not looked up from anywhere)
- when a row doesn't parse right (bad price etc) I just drop it instead of
  trying to guess a value for it - felt safer than making something up
- went with 5 categories (Travel, Mystery, Historical Fiction, Classics,
  Poetry) just to be safely over the 60 book minimum

## Real run results
- scraped 107 books total across the 5 categories (Travel 11, Mystery 32,
  Historical Fiction 26, Classics 19, Poetry 19)
- cleaning kept all 107 rows this time, 0 got dropped
- books.db ended up with 5 categories and 107 books
- ran all 5 queries fine, and the pandas-merge-vs-sql-join check printed
  True

## How to run it
```
pip install requests beautifulsoup4 lxml pandas
python run_all.py
```
(scrape.py needs internet, the rest don't)

## random thing I noticed
some book titles have weird looking characters when I print them in
powershell (like an apostrophe showing up as gibberish) - pretty sure this
is just a terminal display thing and not an actual problem with the data,
looked fine when I opened the csv in vscode
