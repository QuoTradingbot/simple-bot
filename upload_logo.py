"""Upload logo to Azure Blob Storage"""
from azure.storage.blob import BlobServiceClient
import sys

try:
    # Get connection string from Azure
    import subprocess
    result = subprocess.run(
        ['az', 'storage', 'account', 'show-connection-string', '--name', 'quotradingfiles', '-o', 'tsv'],
        capture_output=True,
        text=True
    )
    connection_string = result.stdout.strip()
    
    if not connection_string:
        print("❌ Could not get connection string")
        sys.exit(1)
    
    # Create blob service client
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    
    # Create container if it doesn't exist
    container_name = "images"
    try:
        container_client = blob_service_client.create_container(container_name, public_access='blob')
        print(f"✅ Created container: {container_name}")
    except Exception as e:
        if "ContainerAlreadyExists" in str(e) or "already exists" in str(e).lower():
            print(f"✅ Container {container_name} already exists")
        else:
            raise
    
    # Upload the logo
    logo_path = r"C:\Users\kevin\Downloads\simple-bot\logo for Quo Trading.png"
    blob_name = "quotrading-logo.png"
    
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    
    with open(logo_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    
    logo_url = f"https://quotradingfiles.blob.core.windows.net/{container_name}/{blob_name}"
    print(f"\n✅ Logo uploaded successfully!")
    print(f"📎 URL: {logo_url}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
