import os
import zipfile
import requests
import pandas as pd
from pathlib import Path

# Zenodo URL for RAVDESS Audio Speech (Speech actors 01-24)
RAVDESS_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"

# Emotion labels mapping in RAVDESS
EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def download_ravdess(dest_dir: str = "ml/data/ravdess", zip_name: str = "ravdess_speech.zip") -> str:
    """
    Downloads and extracts the RAVDESS Speech dataset from Zenodo if it doesn't already exist.
    """
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Check if we already have files in actor directories
    actors_exist = all((dest_path / f"Actor_{i:02d}").exists() for i in range(1, 25))
    if actors_exist:
        print("RAVDESS Actor directories already exist. Skipping download.")
        return str(dest_path)
        
    zip_path = Path("ml/data") / zip_name
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not zip_path.exists():
        print(f"Downloading RAVDESS Speech dataset from Zenodo to {zip_path}...")
        try:
            response = requests.get(RAVDESS_URL, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024 # 1MB chunks
            downloaded = 0
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"Progress: {percent:.1f}% ({downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB)", end='\r')
            print("\nDownload completed successfully.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise e
            
    print(f"Extracting dataset to {dest_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # We want to extract it in the dest_dir
            zip_ref.extractall(dest_path)
        print("Extraction complete.")
        
        # Optionally clean up the zip file to save disk space
        try:
            os.remove(zip_path)
            print("Cleaned up raw zip file.")
        except OSError:
            pass
            
    except Exception as e:
        print(f"Error extracting zip: {e}")
        raise e
        
    return str(dest_path)

def parse_ravdess_metadata(dataset_dir: str = "ml/data/ravdess") -> pd.DataFrame:
    """
    Scans the extracted RAVDESS folders and extracts label information from file names.
    Filenames are formatted as:
    modality(03)-vocal_channel(01)-emotion(01-08)-intensity(01-02)-statement(01-02)-repetition(01-02)-actor(01-24).wav
    """
    dataset_path = Path(dataset_dir)
    data = []
    
    # Search for all wav files recursively
    wav_files = list(dataset_path.glob("**/*.wav"))
    if len(wav_files) == 0:
        print(f"No WAV files found in {dataset_dir}. Make sure data is downloaded and extracted.")
        return pd.DataFrame()
        
    for wav_file in wav_files:
        filename = wav_file.stem
        parts = filename.split('-')
        
        if len(parts) != 7:
            # Not a RAVDESS file format, skip it
            continue
            
        modality, vocal_channel, emotion_code, intensity_code, statement_code, repetition_code, actor_code = parts
        
        # Verify it's speech
        if modality != "03" or vocal_channel != "01":
            continue
            
        emotion = EMOTIONS.get(emotion_code, "unknown")
        actor_id = int(actor_code)
        
        # Gender determined by actor ID (odd = male, even = female)
        gender = "female" if actor_id % 2 == 0 else "male"
        
        # Intensity (01 = normal, 02 = strong)
        intensity = "normal" if intensity_code == "01" else "strong"
        
        # Statement
        statement = "kids" if statement_code == "01" else "dogs"
        
        data.append({
            "file_path": str(wav_file.resolve()),
            "filename": wav_file.name,
            "emotion": emotion,
            "intensity": intensity,
            "statement": statement,
            "actor_id": actor_id,
            "gender": gender,
            "repetition": int(repetition_code)
        })
        
    df = pd.DataFrame(data)
    print(f"Parsed {len(df)} files from RAVDESS.")
    return df

def get_speaker_independent_split(df: pd.DataFrame):
    """
    Splits the RAVDESS dataframe into Training, Validation, and Testing speaker-independently.
    Training: Actors 1-18
    Validation: Actors 19-20
    Testing: Actors 21-24
    """
    if df.empty:
        raise ValueError("Dataframe is empty. Cannot split.")
        
    train_df = df[df["actor_id"].between(1, 18)].reset_index(drop=True)
    val_df = df[df["actor_id"].between(1, 19) & (df["actor_id"] >= 19) & (df["actor_id"] <= 20)].reset_index(drop=True)
    test_df = df[df["actor_id"].between(21, 24)].reset_index(drop=True)
    
    print(f"Speaker-independent split summary:")
    print(f"  Train: {len(train_df)} files (Actors 1-18)")
    print(f"  Val:   {len(val_df)} files (Actors 19-20)")
    print(f"  Test:  {len(test_df)} files (Actors 21-24)")
    
    # Check for speaker overlap
    train_actors = set(train_df["actor_id"])
    val_actors = set(val_df["actor_id"])
    test_actors = set(test_df["actor_id"])
    
    assert train_actors.isdisjoint(val_actors), "Data Leakage: Overlap between training and validation speakers!"
    assert train_actors.isdisjoint(test_actors), "Data Leakage: Overlap between training and testing speakers!"
    assert val_actors.isdisjoint(test_actors), "Data Leakage: Overlap between validation and testing speakers!"
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    # Test script execution
    try:
        path = download_ravdess()
        df = parse_ravdess_metadata(path)
        if not df.empty:
            train, val, test = get_speaker_independent_split(df)
            print("Data pipeline setup verified successfully!")
    except Exception as e:
        print(f"Verification failed: {e}")
