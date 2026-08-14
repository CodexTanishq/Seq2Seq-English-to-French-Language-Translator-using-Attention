import pandas as pd
import requests
import zipfile
import io

url = "http://www.manythings.org/anki/fra-eng.zip"

# Tell the server we are a standard web browser to bypass bot-blocking
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

print("Downloading dataset...")
response = requests.get(url, headers=headers)

# Ensure the server actually sent a 200 OK success status before trying to unzip
if response.status_code == 200:
   
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        with z.open('fra.txt') as f:
            df = pd.read_csv(f, sep='\t', header=None, names=['English', 'French', 'Attribution'])
            df = df[['English', 'French']] # Drop the attribution column
            print(f"Success! Loaded {len(df)} sentence pairs.")
            df.to_csv('cleaned_fra_eng.csv', index=False)
            print("Saved data to cleaned_fra_eng.csv")
            
            # Print the first 5 rows to verify
            print(df.head())
else:
    print(f"Failed to download. The server returned status code: {response.status_code}")