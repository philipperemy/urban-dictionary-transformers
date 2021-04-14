## Urban Dictionary Transformers
Transformers applied to Urban Dictionary for fun.


### Download the data

```bash
wget https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/8010727/UT_raw_plus_lowercase.7z
7z x UT_raw_plus_lowercase.7z # sudo apt-get install p7zip-full
md5sum words.json # 5aca6e9bb2c8b9eb7fc5ebc9f947ec33  words.json
```

### Run the code

```bash
pip install -r requirements.txt
python urban.py
```
