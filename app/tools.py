import os
from dotenv import load_dotenv
import boto3

load_dotenv()

s3 = boto3.client(
   "s3",
   aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
   aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)


def upload_file_to_s3(file, bucket_name, acl="public-read"):

    try:

        s3.upload_fileobj(
            file,
            bucket_name,
            file.filename,
            ExtraArgs={
                "ACL": acl,
                "ContentType": file.content_type
            }
        )

    except Exception as e:
        print("ERROR: ", e)
        return e

    return f"{os.getenv('AWS_REGION')}{file.filename}"
