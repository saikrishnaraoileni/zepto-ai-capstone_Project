# clean.py
# Step 2 - take the raw scraped data and turn the messy text columns into
# actual numbers/booleans we can use in the database and later in queries.

import pandas as pd

# this is the fixed conversion rate the assignment tells us to use
# 1 GBP = 105.50 INR (not looking this up live, just a constant like the assignment says)
GBP_TO_INR = 105.50

# the rating on the website is shown as a word not a number, so need a way
# to turn "Three" into 3 etc
rating_word_to_number = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


def clean_price(price_text):
    # price looks like "£51.77" so just remove the pound sign and convert to float
    # sometimes the £ symbol gets weirdly encoded so I strip that mangled version too
    try:
        cleaned = price_text.replace("£", "")
        cleaned = cleaned.replace("Â", "")
        cleaned = cleaned.strip()
        return float(cleaned)
    except:
        # if something goes wrong just return None, we'll drop the row later
        return None


def clean_rating(rating_text):
    # rating_text is something like "Three", look it up in our dictionary
    if rating_text in rating_word_to_number:
        return rating_word_to_number[rating_text]
    else:
        return None


def clean_availability(availability_text):
    # availability_text looks like "In stock (22 available)"
    # we just want True/False for whether its in stock
    if availability_text is None:
        return None
    if "in stock" in availability_text.lower():
        return True
    else:
        return False


# ---- main part ----

df = pd.read_csv("books_raw.csv")
print("rows before cleaning:", len(df))

# apply our cleaning functions to make new columns
df["price_gbp"] = df["price"].apply(clean_price)
df["rating"] = df["star_rating"].apply(clean_rating)
df["in_stock"] = df["availability"].apply(clean_availability)

# if any of these came out as None/NaN, that row failed to parse properly.
# I decided to just DROP those rows instead of making up a value for them,
# because a weird price/rating usually means something was different about
# that specific book listing, not a "normal" missing value we should guess at.
bad_rows = df["price_gbp"].isna() | df["rating"].isna() | df["in_stock"].isna()
print("rows that failed to parse (dropping these):", bad_rows.sum())
df = df[~bad_rows]

# now that we only have good rows, these columns are safe to force to int/bool
df["rating"] = df["rating"].astype(int)
df["in_stock"] = df["in_stock"].astype(bool)

# finally add the INR price column using the fixed rate
df["price_inr"] = df["price_gbp"] * GBP_TO_INR
df["price_inr"] = df["price_inr"].round(2)  # just to make it look nicer

print("rows after cleaning:", len(df))
print("conversion rate used: 1 GBP =", GBP_TO_INR, "INR")

# only keep the columns we actually need going forward
final_columns = ["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]
df_final = df[final_columns]

df_final.to_csv("books_clean.csv", index=False)
print("saved books_clean.csv")
