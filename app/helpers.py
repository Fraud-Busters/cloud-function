from google.cloud import storage

fbxdana_client = storage.Client()
fbxdana = fbxdana_client.get_bucket("fbxdana")
