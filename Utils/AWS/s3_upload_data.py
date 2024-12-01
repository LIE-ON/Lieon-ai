import os
import boto3
import logging
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

# 디버깅용 로깅 활성화 (S3 관련 모든 요청의 로그 출력)
boto3.set_stream_logger('boto3.resources', logging.DEBUG)


# AWS S3 클라이언트 생성
def create_s3_client(aws_access_key_id=None, aws_secret_access_key=None, region_name=None):
    try:
        if aws_access_key_id and aws_secret_access_key:
            # 명시적 자격 증명 설정
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name
            )
        else:
            # 환경 변수 또는 AWS CLI 설정에 따라 자격 증명을 자동으로 불러옴
            s3_client = boto3.client('s3', region_name=region_name)

        print("S3 client created successfully.")
        return s3_client

    except (NoCredentialsError, PartialCredentialsError) as e:
        print("Credentials error:", e)
        return None


# 폴더 내 모든 파일을 S3에 업로드하는 함수
def upload_folder_to_s3(folder_path, bucket_name, dir_dst, s3_client):
    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        return False

    success_count = 0
    failure_count = 0

    for root, _, files in os.walk(folder_path):
        for file_name in files:
            local_path = os.path.join(root, file_name)
            # S3 키를 올바르게 생성
            relative_path = os.path.relpath(local_path, folder_path)
            s3_key = os.path.normpath(os.path.join(dir_dst, relative_path)).replace('\\', '/')

            try:
                s3_client.upload_file(local_path, bucket_name, s3_key)
                print(f"Uploaded {local_path} to s3://{bucket_name}/{s3_key}")
                success_count += 1
            except NoCredentialsError:
                print("Error: No AWS credentials found.")
                failure_count += 1
            except Exception as e:
                print(f"Failed to upload {local_path}. Error: {e}")
                failure_count += 1

    print(f"업로드 성공: {success_count}개 파일")
    print(f"업로드 실패: {failure_count}개 파일")

    return True  # 또는 return success_count, failure_count


# 실행 예시
aws_access_key_id = 'AKIAQXGV7F44U47OOML5'
aws_secret_access_key = 'your_secret_access_key'
region_name = 'ap-northeast-2'  # 리전 설정 (필요에 따라 변경 가능)
bucket_name = 'lieon-data'
folder_path = 'Local_audio_folder_path'  # 로컬 폴더 경로
dir_dst = 'Local_audio_upper_folder_path'  # 's3://lieon-data/' 제거


# S3 클라이언트 생성
s3_client = create_s3_client(aws_access_key_id, aws_secret_access_key, region_name)

# 폴더 업로드 실행
if s3_client:
    upload_folder_to_s3(folder_path, bucket_name, dir_dst, s3_client)

