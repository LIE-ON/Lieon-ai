import os
import boto3
import logging
from collections import Counter
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

# 디버깅용 로깅 설정 (필요에 따라 로그 레벨 조정)
boto3.set_stream_logger('boto3.resources', logging.CRITICAL)

# AWS S3 클라이언트 생성 함수 (이전 코드와 동일)
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

        print("S3 클라이언트가 성공적으로 생성되었습니다.")
        return s3_client

    except (NoCredentialsError, PartialCredentialsError) as e:
        print("자격 증명 오류:", e)
        return None

# S3에서 특정 경로의 파일 리스트를 가져오는 함수
def list_s3_files(bucket_name, prefix, s3_client):
    s3_files = []
    paginator = s3_client.get_paginator('list_objects_v2')
    try:
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    # 파일명만 추출하여 리스트에 추가
                    file_key = obj['Key']
                    file_name = os.path.basename(file_key)
                    s3_files.append(file_name)
        return s3_files
    except Exception as e:
        print(f"S3에서 객체 목록을 가져오는 중 오류 발생: {e}")
        return []

# 라벨 파일에서 대응되는 오디오 파일명으로 변환하는 함수
def label_to_audio_filename(label_filename):
    return label_filename.replace('label', 'data').replace('.csv', '.wav')

# S3 오디오 파일과 로컬 라벨 파일을 비교하는 함수
def find_extra_audio_files_in_s3(label_dir, bucket_name, s3_prefix, s3_client):
    # 1. S3에 업로드된 오디오 파일 리스트 가져오기
    s3_audio_files = list_s3_files(bucket_name, s3_prefix, s3_client)
    s3_audio_filenames = s3_audio_files

    # S3에서 가져온 오디오 파일 개수 출력
    print(f"S3에서 가져온 오디오 파일 개수: {len(s3_audio_filenames)}개")

    # 2. 라벨 파일 리스트 가져오기
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.csv')]
    expected_audio_filenames = [label_to_audio_filename(f) for f in label_files]

    # 라벨 파일 개수 출력
    print(f"로컬에서 가져온 라벨 파일 개수: {len(label_files)}개")

    # 3. S3에 있지만 라벨에 없는 오디오 파일 찾기
    extra_audio_files = set(s3_audio_filenames) - set(expected_audio_filenames)

    # 4. 결과 출력
    if extra_audio_files:
        print(f"라벨에 대응되지 않는 S3 오디오 파일 수: {len(extra_audio_files)}개")
        print("라벨에 없는 S3 오디오 파일 목록:")
        for file_name in extra_audio_files:
            print(file_name)
    else:
        print("S3의 모든 오디오 파일이 로컬 라벨에 대응됩니다.")

    # 5. S3 오디오 파일에서 파일명 중복 검사
    duplicates = [item for item, count in Counter(s3_audio_filenames).items() if count > 1]

    if duplicates:
        print(f"S3에서 파일명이 중복된 오디오 파일 수: {len(duplicates)}개")
        print("중복된 파일명 목록:")
        for file_name in duplicates:
            print(file_name)
    else:
        print("S3에서 파일명이 중복된 오디오 파일이 없습니다.")

# 실행 예시
if __name__ == "__main__":
    aws_access_key_id = 'your_keyid'  # 실제 키로 대체하세요
    aws_secret_access_key = 'your_secrec_access_key'  # 실제 키로 대체하세요
    region_name = 'ap-northeast-2'  # 리전 설정
    bucket_name = 'lieon-data'
    s3_prefix = 'Dataset/Train/Audio_edit/'  # S3에서의 경로 (접두사)
    label_edit_directory = 'local_label_directory'

    # S3 클라이언트 생성
    s3_client = create_s3_client(aws_access_key_id, aws_secret_access_key, region_name)

    # 비교 및 중복 검사 실행
    if s3_client:
        find_extra_audio_files_in_s3(label_edit_directory, bucket_name, s3_prefix, s3_client)
