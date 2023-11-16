# Code and Data for Dialogue Topic Segmenter
This repository maintains:
* **[PART I]** The source code of "[Improving Unsupervised Dialogue Topic Segmentation with Utterance-Pair Coherence Scoring](https://aclanthology.org/2021.sigdial-1.18.pdf)", SIGDIAL-21.
* **[PART II - Continously Update]** A [dataset collection](https://github.com/lxing532/Dialogue-Topic-Segmenter/tree/main/data) of Dialogue Topic Segmentation cleaned to a standarized format.

<br/>

## PART I:
This list of python scripts are together as the source codebase of our paper:
- For training:
  - __*train.py*__ : contains the main code (Sec 3.2 in the paper) for the training process of the utterance-pair coherence scoring model grounded on BERT (Next Sentence Prediction). 
  - __*data_utils.py*__ : contains the main code (Sec 3.1 in the paper) for pseudo training data generation, which will be loaded to train the coherence scoring model.
  - __*model_utils.py*__ : contains the class of coherence scoring model.
- For evaluation:
  - __*segment.py*__ : contains the main code to conduct evluation procedure based on the TextTiling segmentation framework.
  - __*neural_texttiling.py*__ : contains the detailed implementation of TextTiling with different settings of text encoder (e.g., Bi-encoder, cross-encoder, coherence scoring etc)

### Training/Testing Steps:
**0. Instaill env requirements**
```
pip install -r requirements.txt
```
**1. Download DailyDial from this [link](http://yanran.li/dailydialog) and add the following three files to __*./data/train/dailydialog/*__ :**
``` diff
+ dialogues_text.txt
+ dialogues_topic.txt
+ dialogue_act.txt
```
**2. Execute the training command, for example:**
```
python train.py -t ./data/train/dailydialog/ -e bert-base-uncased -s ./checkpoints -m 1 -r 10 -b 32
```
>💡Notes:
>  * The current data loading/generation code (data_utils.py) is specifically implemented for DailyDialog, please adjust it accordingly to load your datasets if need.
>  * The training code only support bert-based language model supporting mode of Next Sentence Prediction, otherwise it will run into errors (e.g., loading roberta, sbert as text encoder).
>  * Checkpoint of each epoch will be saved to your specified directory.

**3. For evaluation, you can run command like:**
```
python segment.py -t ./data/eval/dialseg_711.json -e ./checkpoints/cpt_0.pth -m CM
```
>💡Notes:
>  * The current data loading/generation code (data_utils.py) is specifically implemented for DailyDialog, please adjust it accordingly to load your datasets if need.
>  * The training code only support bert-based language model supporting mode of Next Sentence Prediction, otherwise it will run into errors (e.g., loading roberta, sbert as text encoder).
>  * Checkpoint of each epoch will be saved to your specified directory.


## 2. Data Generation:
Once the source of training data is ready, we run *data_process.py* to generate the postive and negative utterance pair samples for the training of BERT-based coherence scoring model. Please note that the code will generate three files:
* dialogues_text.txt
* dialogues_topic.txt
* dialogues_act.txt

These three files will be required to work together to manage the data loading of model training.

## 3. Coherence Scoring Model Training:
Please modify the paths in the *model.py* file to the paths you save your data files in. This code will save the utterance-pair coherence scoring model, which will be further utilized in *test.py* for topic segmentation inference.

## 4. Topic Segmentation Inference:
In the evaluation phase, three datasets are used for model testing, they are:
* [DialSeg 711](https://github.com/xyease/TADAM)
* [Doc2Dial](https://paperswithcode.com/paper/doc2dial-a-goal-oriented-document-grounded)
* [ZYS](https://github.com/xyease/TADAM)

