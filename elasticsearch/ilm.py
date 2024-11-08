



from elasticsearch import Elasticsearch


    

def initialize_index_with_ilm(alias_name):
    """
    根据别名创建初始化索引，并设置索引生命周期管理策略。

    参数:
    alias_name (str): 要创建的索引的别名。

    该函数执行以下操作：
    1. 创建一个带有别名的索引。
    2. 创建 ILM 策略，定义索引的生命周期管理。
    3. 设置索引的 ILM 设置，包括生命周期名称和别名。
    4. 输出索引策略详情。
    """
    index_name = f"{alias_name}-000001"
    ilm_policy_name = f"{alias_name}-ilm"

    es = Elasticsearch("http://localhost:9200/")

    # 创建带有别名的索引
    es.indices.create(
        index=index_name,
        body={
            "aliases": {
                alias_name: {
                    "is_write_index": True
                }
            }
        }
    )

    # 创建 ILM 策略
    es.ilm.put_lifecycle(
        name=ilm_policy_name,
        body={
            "policy": {
                "phases": {
                    "hot": {
                        "actions": {
                            "rollover": {
                                "max_docs": 1000,
                                "max_age": "1m",
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

    # 设置索引的 ILM 设置
    es.indices.put_settings(
        index=index_name,
        body={
            "index": {
                "lifecycle": {
                    "name": ilm_policy_name,
                    "rollover_alias": alias_name
                }
            }
        }
    )

    # 确保新索引也应用 ILM 策略
    es.indices.put_settings(
        index=f"{alias_name}-*",
        body={
            "index": {
                "lifecycle": {
                    "name": ilm_policy_name
                }
            }
        }
    )

    # 输出索引策略详情
    ilm_explain = es.indices.get_settings(index=alias_name)
    print(ilm_explain)

if __name__ == "__main__":
    alias_name = "search-index"  # You can change this to your desired alias name
    initialize_index_with_ilm(alias_name)

