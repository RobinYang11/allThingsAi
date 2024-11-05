from elasticsearch import Elasticsearch


def create_ilm_index(index_name):
    """
    创建一个带有 ILM（Index Lifecycle Management）策略的 Elasticsearch 索引。

    参数:
    index_name (str): 要创建的索引的名称。索引将以 "{index_name}-000001" 的格式命名。

    该函数执行以下操作：
    1. 创建一个带有别名的索引。
    2. 创建 ILM 策略，定义索引的生命周期管理。
    3. 设置索引的 ILM 设置，包括生命周期名称和别名。
    4. 可选地检查 ILM 状态并打印结果。

    示例:
    >>> create_ilm_index("my_index")
    """
    
    
    ilm_policy_name  = f"{index_name}-ilm"

    es = Elasticsearch("http://localhost:9200/")

    # Create the index with an alias
    es.indices.create(
        index=f"{index_name}-000001",
        body={
            "aliases": {
                index_name: {
                    "is_write_index": True
                }
            }
        }
    )

    # Create the ILM policy
    es.ilm.put_lifecycle(
        name=ilm_policy_name,
        body={
            "policy": {
                "phases": {
                    "hot": {
                        "actions": {
                            "rollover": {
                                "max_age": "1d",
                                "max_docs": 1000,
                                "max_size": "50mb"
                            }
                        }
                    },
                    "delete": {
                        "min_age": "30d",
                        "actions": {
                            "delete": {}
                        }
                    }
                }
            }
        }
    )

    # Set the index settings for ILM
    es.indices.put_settings(
        index=f"{index_name}-000001",
        body={
            "index": {
                "lifecycle": {
                    "name": ilm_policy_name,
                    "rollover_alias": index_name
                }
            }
        }
    )

    # Optionally, you can check the ILM status
    ilm_explain = es.indices.get_settings(index=index_name)
    print(ilm_explain)

    # Rollover the index
    # es.indices.rollover(index=index_name)

create_ilm_index("search-index")