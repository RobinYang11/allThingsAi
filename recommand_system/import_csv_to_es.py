

import pandas as pd
from elasticsearch import Elasticsearch, helpers

es = Elasticsearch("http://localhost:9200/")

def import_csv_to_es(csv_file_path, es_index):
    # Initialize Elasticsearch client

    # es.indices.create(index=es_index)  # Create the index if it doesn't exist

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    actions = []
    for _,row in df.iterrows():
        print(row.to_dict())
        actions.append({
           "_index" :es_index,
           "_source":row.to_json()
        })
        
    helpers.bulk(es, actions)


import_csv_to_es("./users_interactions.csv","shared_inter")

