from cloudevents.http import CloudEvent
from app.processing import predict
import functions_framework
from app.helpers import set_status, set_status_out_key_preview, set_status_err

# Triggered by a change in a storage bucket
@functions_framework.cloud_event
def hello_gcs(cloud_event: CloudEvent) -> tuple:
    """This function is triggered by a change in a storage bucket.

    Args:
        cloud_event: The CloudEvent that triggered this function.
    Returns:
        The event ID, event type, bucket, name, metageneration, and timeCreated.
    """
    data = cloud_event.data

    event_id = cloud_event["id"]
    event_type = cloud_event["type"]

    bucket = data["bucket"]
    name = data["name"]
    metageneration = data["metageneration"]
    timeCreated = data["timeCreated"]
    updated = data["updated"]

    print(f"Event ID: {event_id}")
    print(f"Event type: {event_type}")
    print(f"Bucket: {bucket}")
    print(f"File: {name}")
    print(f"Metageneration: {metageneration}")
    print(f"Created: {timeCreated}")
    print(f"Updated: {updated}")

    if ("_out.csv" in name) or (not "_in.csv" in name):
        print("Skipping file")
        return event_id, event_type, bucket, name, metageneration, timeCreated, updated

    url = f"gs://{bucket}/{name}"
    key = name.replace("_in", "_out")
    pred_id = name.split("/")[1]
    pred_id = pred_id.split("_")[0]

    print(f"Key: {key}")
    print(f"URL: {url}")
    try:
        set_status(pred_id, "PROCESSING")
        preview = predict(url, key)
        set_status_out_key_preview(pred_id, "COMPLETED", key, preview)
    except Exception as e:
        print("ERROR: ", e)
        set_status_err(pred_id, "FAILED_TO_PROCESS", str(e))


    return event_id, event_type, bucket, name, metageneration, timeCreated, updated


# [END functions_cloudevent_storage]
