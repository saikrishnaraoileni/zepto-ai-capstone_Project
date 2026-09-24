# scrape.py
# Step 1 of the data pipeline assignment
# Goal: go to books.toscrape.com and pull out title, price, rating,
# availability and category for a bunch of books, then save it as a csv.

# I picked 5 categories just so I get way more than the required 60 books.
# You can change this list if you want to try different categories.

import requests
from bs4 import BeautifulSoup
import csv
import time

homepage_url = "http://books.toscrape.com/"

categories_i_want = ["Travel", "Mystery", "Historical Fiction", "Classics", "Poetry"]

# this will hold every book we scrape, as a list of dictionaries
all_books = []


def get_category_links():
    # first go to the homepage and find the sidebar links for each category
    r = requests.get(homepage_url)
    soup = BeautifulSoup(r.text, "html.parser")

    # the category links are inside div.side_categories -> ul -> li -> ul -> li -> a
    links = soup.select("div.side_categories ul li ul li a")

    category_dict = {}
    for link in links:
        cat_name = link.text.strip()
        cat_url = homepage_url + link["href"]
        category_dict[cat_name] = cat_url

    return category_dict


def scrape_one_category(name, url):
    print("scraping category:", name)
    books_from_this_category = []

    current_url = url

    # keep going while there is a "next" page button
    while True:
        r = requests.get(current_url)
        soup = BeautifulSoup(r.text, "html.parser")

        # every book on the page is inside article.product_pod
        book_boxes = soup.select("article.product_pod")

        for box in book_boxes:
            title = box.h3.a["title"]
            price = box.select_one("p.price_color").text
            # the rating is stored weirdly - its in the class name like "star-rating Three"
            rating_class = box.select_one("p.star-rating")["class"]
            rating_word = rating_class[1]  # class[0] is just "star-rating"
            availability = box.select_one("p.instock.availability").text.strip()

            # save this book as a dictionary, add category too since we know it
            book_info = {
                "title": title,
                "price": price,
                "star_rating": rating_word,
                "availability": availability,
                "category": name,
            }
            books_from_this_category.append(book_info)

        # check if theres a next page button
        next_button = soup.select_one("li.next a")
        if next_button:
            # the next page url is relative to current page, this took me a while to figure out
            next_href = next_button["href"]
            current_url = current_url.rsplit("/", 1)[0] + "/" + next_href
            time.sleep(0.3)  # dont want to hammer the site too fast
        else:
            break  # no more pages for this category

    print("  got", len(books_from_this_category), "books from", name)
    return books_from_this_category


# ---- main part of the script ----

category_links = get_category_links()

for cat in categories_i_want:
    if cat in category_links:
        result = scrape_one_category(cat, category_links[cat])
        all_books.extend(result)
    else:
        print("hmm, couldn't find category:", cat)

print("TOTAL BOOKS SCRAPED:", len(all_books))

# now save everything to a csv so clean.py can use it next
with open("books_raw.csv", "w", newline="", encoding="utf-8") as f:
    fieldnames = ["title", "price", "star_rating", "availability", "category"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for b in all_books:
        writer.writerow(b)

print("saved to books_raw.csv, moving on to clean.py next")
