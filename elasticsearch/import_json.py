



from elasticsearch import Elasticsearch, helpers
import json
from elasticsearch import Elasticsearch, helpers
import json

# 数据集在 百度飞桨 AI STUDIO中下载
# https://aistudio.baidu.com/datasetdetail/177185

# 初始化 Elasticsearch 客户端
es = Elasticsearch("http://localhost:9200/",
                #  api_key="wkYFKO1lSMu-Ino7OIKMhw", 可以在kibana中生成API_KEY
)

def combine_documents(json_data):
    # Load the JSON data
    data = json.loads(json_data)
    
    # Initialize an empty string to hold the combined text
    combined_text = ""
    
    # Iterate through each document
    for document in data.get("documents", []):
        # Combine the title
        combined_text += document.get("title", "") + "\n\n"
        
        # Combine the segmented title
        combined_text += "Segmented Title: " + " ".join(document.get("segmented_title", [])) + "\n\n"
        
        # Combine the paragraphs
        combined_text += "\n".join(document.get("paragraphs", [])) + "\n\n"
        
        # Combine the segmented paragraphs
        combined_text += "Segmented Paragraphs:\n"
        for segmented_paragraph in document.get("segmented_paragraphs", []):
            combined_text += " ".join(segmented_paragraph) + "\n"
        combined_text += "\n"  # Add extra newline for separation
    
    return combined_text.strip()  # Remove any trailing whitespace



def bulk_import_to_elasticsearch(json_file_path,es_index_name):
    # Open the JSON file for reading
    with open(json_file_path, 'r', encoding='utf-8') as f:
        # Use the helpers.bulk method to import documents
        actions = []
        for line in f:
            doc = json.loads(line)  # Parse each line as a JSON document
            print("================\n")
            
            # print(doc)
            abc = combine_documents(json.dumps(doc))
            abc = abc.replace(" ", "")
            doc["content"] = abc
            # doc["documnt"]
            
            if 'documents' in doc:
                del doc['documents']
            print(doc)
            print("================\n")
            action = {
                "_index": es_index_name,  # Replace with your index name
                "_source": doc
            }
            actions.append(action)
        
        # Bulk insert the documents into Elasticsearch
        helpers.bulk(es, actions)

# Call the function with the path to your JSON file
bulk_import_to_elasticsearch('./search.test.json','search_index_4')
