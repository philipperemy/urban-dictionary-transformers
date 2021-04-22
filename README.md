## Urban Dictionary Transformers
Transformers applied to Urban Dictionary for fun.


 *Model name* | *Dataset* | *Num samples (train)* | *Num samples (test)* | *Cross Entropy (test)* | *Num Epochs* | Download model | Results (test)
 | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- 
Facebook BART Base          | Urban Dictionary | 1,190,865 | 62,677 | 1.5180 | 10 |  [Click (zip)](https://drive.google.com/drive/folders/1dI3o4yTBWHv5s15LxowCY3FDtFyGWkgO?usp=sharing) |  [Click (json)](https://drive.google.com/drive/folders/1dI3o4yTBWHv5s15LxowCY3FDtFyGWkgO?usp=sharing) |

The checkpoints are compatible with [simpletransformers](https://github.com/ThilinaRajapakse/simpletransformers).

### Goal

From the description we try to guess the word.

- Input: **One who has a mania for music.**
- Target: **melomaniac**

<hr>
<img src="https://user-images.githubusercontent.com/4516927/114693303-6c04ce00-9d54-11eb-8ce8-28499512605b.png"></img>
<hr>

### Some Results

Isn't that good?

```json
{
    "target": "Oh, my back!",
    "input": "Used when something unpleasant or distasteful is about to happen.",
    "predicted": "Shit is goin down"
},
{
    "target": "ho",
    "input": "A bitch.",
    "predicted": "Kanye West"
},
{
  "target": "tequila suicide",
  "input": "An alternative to the [tequila slammer]. Snort the salt, down the shot, squeeze lime in the eye.\r\n\r\n*WARNING* May cause nasal bleeding, bloodshot eyes and being totally [muntered].",
  "predicted": "tequila sniper"
},
{
  "target": "A.F.I.D.",
  "input": "asian female immigrant driver",
  "predicted": "a.f.i.d."
},
{
  "target": "ecuador",
  "input": "The best country in the world.  It is home to a superior race of men who is beginning world domination in Rochester, Minnesota and in Rogers, (the whole) Arkansas.",
  "predicted": "America"
},
{
  "target": "rich",
  "input": "Someone who has a lot of money",
  "predicted": "rich"
},
{
  "target": "high-altitude fuck",
  "input": "1. An instance of sexual intercourse performed in a mountainous region that is usually accompanied by an intense over-bearing shame inflicted on the participants by the conservative views of the society in which they live.  It is also usually directly preceded by stream fishing.  ",
  "predicted": "mountain fuck"
},
```


### How to?

#### Download the data

```bash
wget https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/8010727/UT_raw_plus_lowercase.7z
7z x UT_raw_plus_lowercase.7z # sudo apt-get install p7zip-full
md5sum words.json # 5aca6e9bb2c8b9eb7fc5ebc9f947ec33  words.json
```

#### Run the code

```bash
# Download the ZIP https://drive.google.com/drive/folders/1dI3o4yTBWHv5s15LxowCY3FDtFyGWkgO.
unzip urban-checkpoints-20210421.zip
pip install -r requirements.txt
export CUDA_VISIBLE_DEVICES=1; python urban.py --resume_from urban-checkpoints-20210421 --eval_only
```
