import requests
import pandas as pd
from bs4 import BeautifulSoup
from tabulate import tabulate

# URL of the Wikipedia page listing highest-grossing films
url = 'https://books.toscrape.com/catalogue/page-1.html'

# # Send a GET request to the URL
response = requests.get(url)
# print(response)
#
# # Parse the HTML content of the webpage
soup = BeautifulSoup(response.text, 'html.parser')
# print(soup)
#
body = soup.body
# print(body)

table = body.find_all('div', class_='col-sm-8 col-md-9')
# print(table)

img_Tag = []
for i in table:
    y = i.find_all('div', class_='image_container')
    print(y)

for i in y:
    img_Tag.append(i.find_all('img', class_="thumbnail"))


