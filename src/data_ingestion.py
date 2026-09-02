import os
from databricks.sdk import WorkspaceClient


def download_data_from_volume(volume_path: str, local_dest: str = "raw_telco.csv") -> str:
    print(f"📥 Fetching raw dataset from Databricks Volume: {volume_path}")
    w = WorkspaceClient()
    
    w.files.download_to(file_path=volume_path, destination=local_dest)
            
    return local_dest

    