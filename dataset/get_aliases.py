# pip install sparqlwrapper
# https://rdflib.github.io/sparqlwrapper/

import sys
from SPARQLWrapper import SPARQLWrapper, JSON
import json
from tqdm import tqdm
import os
from pararel_utils import MPARAREL_FOLDER, TUPLES_FOLDER, OBJECT_KEY, OBJECT_URI

SPARQL_URL = "https://query.wikidata.org/sparql"

query_format = """SELECT ?altLabel
{{
 VALUES (?wd) {{(wd:{qcode})}}
 ?wd skos:altLabel ?altLabel .
 FILTER (lang(?altLabel) = "{lang}")
}}"""


def get_aliases(qcode, lang):
    query = query_format.format(qcode=qcode, lang=lang)
    user_agent = "WDQS-example Python/%s.%s" % (
        sys.version_info[0],
        sys.version_info[1],
    )
    sparql = SPARQLWrapper(SPARQL_URL, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return [result["altLabel"]["value"] for result in results["results"]["bindings"]]


tuples_folder = os.path.join(MPARAREL_FOLDER, TUPLES_FOLDER)
output_folder = os.path.join(MPARAREL_FOLDER, TUPLES_FOLDER + "_with_aliases")
os.makedirs(output_folder, exist_ok=True)
for lang in tqdm(os.listdir(tuples_folder), desc="Languages"):
    os.makedirs(os.path.join(output_folder, lang), exist_ok=True)
    lines = []
    for relation_filename in tqdm(
        os.listdir(os.path.join(tuples_folder, lang)), desc="Relations"
    ):
        path_to_file = os.path.join(tuples_folder, lang, relation_filename)
        with open(path_to_file) as f:
            for line in f:
                data = json.loads(line)
                data.pop(OBJECT_KEY)
                data[OBJECT_KEY] = get_aliases(data[OBJECT_URI], lang)
                lines.append(json.dumps(data))
        with open(os.path.join(output_folder, lang, relation_filename), "w") as outfile:
            outfile.write("\n".join(lines))
