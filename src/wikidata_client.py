import requests


class WikidataClient:
    def __init__(self):
        self.base_url = "https://query.wikidata.org/sparql"

    def query_wikidata(self, query):
        headers = {"Accept": "application/sparql+xml"}
        response = requests.post(self.base_url, json={
                                 "query": query}, headers=headers)
        return response.json()
