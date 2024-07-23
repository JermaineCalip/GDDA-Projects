import requests
from bs4 import BeautifulSoup
from tabulate import tabulate

url = 'https://www.bbc.com/sport/football/premier-league/table'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
table = soup.find('table', {'class': 'ssrcss-14j0ip6-Table e3bga5w5'})

rows = table.find_all('tr')

sport = []
for row in rows[1:]:
    cells = row.find_all(['td'])
    if cells:
        Position = cells[0].text.strip()
        Team = cells[1].text.strip()
        Played = cells[2].text.strip()
        Won = cells[3].text.strip()
        Drawn = cells[4].text.strip()
        Lost = cells[5].text.strip()
        For = cells[6].text.strip()
        Against = cells[7].text.strip()
        GD = cells[8].text.strip()
        Points = cells[9].text.strip()
        Form = cells[10].text.strip()
        Forms = Form.replace('LResult', '').replace('WResult', '').replace('DResult', '')

        sport.append([Position, Team, Played, Won, Drawn, Lost, For, Against, GD, Points, Forms])

print(tabulate(sport, headers=['Position', 'Team', 'Played', 'Won', 'Drawn', 'Lost', 'For', 'Against', 'GD', 'Points', 'Form']))

# df = pd.DataFrame(hockey, columns=['Position', 'Team', 'Played', 'Won', 'Drawn', 'Lost', 'For', 'Against', 'GD', 'Points', 'Form'])
# df.to_csv('Hockey.csv', index=False)

