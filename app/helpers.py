from google.cloud import storage
from pymongo import MongoClient
import os
from bson.objectid import ObjectId

mongodb_uri = os.environ['MONGO_URI']

fbxdana_client = storage.Client()
fbxdana = fbxdana_client.get_bucket("fbxdana")
client = MongoClient(mongodb_uri)
db = client['fb']['predictions']

def set_status(pred_id, status):
    db.update_one({"_id": ObjectId(pred_id)}, {"$set": {"status": status}})

def set_status_err(pred_id, status, err):
    db.update_one({"_id": ObjectId(pred_id)}, {"$set": {"status": status, "errorMsg": err}})

def set_status_out_key_preview(pred_id, status, out_key, preview_result):
    db.update_one({"_id": ObjectId(pred_id)}, {"$set": {"status": status, "outKey": out_key, "previewResult": preview_result}})
