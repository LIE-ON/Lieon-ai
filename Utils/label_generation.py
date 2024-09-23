# 순차적으로 파일 처리
for idx, audio_file in enumerate(audio_files, start=1):
    print(f'Processing: {audio_file} ({idx}/{len(audio_files)})')  # 진행 상태 표시
    # PyAnnote pipeline 로드
    pipeline = load_pipeline(use_auth_token)

    try:
        # process_file(audio_file, output_csv_dir, pipeline)
        process_file_for_augment(audio_file, output_csv_dir, pipeline)
    except Exception as e:
        print(f"Error processing file {audio_file}: {e}")