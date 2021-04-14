## Urban Dictionary Transformers
Transformers applied to Urban Dictionary for fun.

### Goal

From the description we try to guess the word.

- Input: **One who has a mania for music.**
- Target: **melomaniac**

![image](https://user-images.githubusercontent.com/4516927/114693303-6c04ce00-9d54-11eb-8ce8-28499512605b.png)

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
