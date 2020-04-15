bash download_IAM_data.sh
python extract_all_words_lines.py
bash process_raw_gts.sh
rm -f largeWriterIndependentTextLineRecognitionTask.zip
rm -f lines.tgz
rm -f words.tgz
rm -f xml.tgz
