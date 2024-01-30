import os
import soundfile as sf
import csv

path = "../dataset"

files = os.listdir(path)

csv_rows = [["audio_path", "caption", "duration"]]
for file_path in files:
    info = sf.info(path+"/"+file_path)
    duration = info.frames / info.samplerate # Calculate the duration in seconds
    csv_rows.append([f"/workspace/sound_test/dataset/{file_path}", "", duration])

# CSV 파일 열기
with open("../csv_files/dummy_test.csv", 'w', newline='') as file:
    # Write to CSV file
    writer = csv.writer(file)
    writer.writerows(csv_rows)
