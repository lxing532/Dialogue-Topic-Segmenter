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
>💡Instruction:
>  * The evaluation script supports the standarized data form defined in our [Dialogue Topic Segmentation Data Hub](https://github.com/lxing532/Dialogue-Topic-Segmenter/tree/main/data). You can easily conduct test by loading any corpus under this directory (./data/eval/).
>  * The evaluation script supports three sentence pair scoring paradigm:
>    * **_Sequence Classification (SC)_** : Encode each sentence individually and compute consine similarty as sentence-pair score.
>      ```
>      python segment.py -t ./data/eval/dialseg_711.json -e bert-base-uncased -m SC
>      ```
>    * **_Next Sentence Prediction (NSP)_** : Encode a pair of sentences together and use the next sentence probability as sentence-pair score.
>      ```
>      python segment.py -t ./data/eval/dialseg_711.json -e bert-base-uncased -m NSP
>      ```
>    * **_Coherence Modeing (CM)_** : Encode a pair of sentences together with trained coherence scoring model and use the coherence score as sentence-pair score.
>      ```
>      python segment.py -t ./data/eval/dialseg_711.json -e ./checkpoints/cpt_1.pth -m CM
>      ```
>  * The default setting to obtain sentence representation is mean-pooling over all token hidden states, if you want to explore other options (e.g., CLS representation), replace line 17 -> line 18 in neural_texttiling.py .
>  * The default included evaluation metrics are: **P_k, Windiff, F1**. To add your own metrics, please adjust the code in ```neural_texttiling.TextTiling```.
>  * Sometimes you may want to investigate actual segmentation prediction by case more than metric values, ```neural_texttiling.TextTiling``` returns segment_prediction as well.


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

