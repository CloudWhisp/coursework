import os
import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf

folders = ['001', '002', '003', '004', '005']

target_sr = 16000

total_files = 0

for folder in folders:
    original_dir = os.path.join(folder, 'NLP+PD', 'wav files')

    if not os.path.exists(original_dir):
        print(f"Папка не найдена: {original_dir} — пропускаем")
        continue

    prepared_dir = f"{folder}_resample"
    os.makedirs(prepared_dir, exist_ok=True)

    ref_files = [f for f in os.listdir(original_dir) if f.lower().endswith(".wav")]

    if not ref_files:
        print(f"В папке {original_dir} нет .wav файлов — пропускаем")
        continue

    total_files += len(ref_files)
    print(f"\nОбрабатываем папку {folder}: {len(ref_files)} файлов → {prepared_dir}")

    for filename in tqdm(ref_files, desc=folder):
        input_path = os.path.join(original_dir, filename)
        output_path = os.path.join(prepared_dir, filename)

        try:
            y, sr = librosa.load(input_path, sr=None, mono=False)

            if y.ndim > 1 and y.shape[0] > 1:
                y = librosa.to_mono(y)

            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

            max_amp = np.max(np.abs(y))
            if max_amp > 0:
                y = y / max_amp * 0.9

            sf.write(output_path, y, target_sr, subtype='PCM_16') 

        except Exception as e:
            print(f"\nОшибка при обработке {input_path}: {e}")

print(f"\nГотово! Обработано файлов: {total_files}")
print("Подготовленные папки:")
for folder in folders:
    print(f"  {folder}_resample")