# Code and Data for Dialogue Topic Segmenter
This repository maintains:
* **[PART I]** The source code of "[Improving Unsupervised Dialogue Topic Segmentation with Utterance-Pair Coherence Scoring](https://aclanthology.org/2021.sigdial-1.18.pdf)", SIGDIAL-21.
* **[PART II - Continously Update]** A [dataset collection](https://github.com/lxing532/Dialogue-Topic-Segmenter/tree/main/data) of Dialogue Topic Segmentation cleaned to a standarized format.

<br/>

## PART I:
This list of python scripts are together as the source codebase of our paper:
- For training:
  - *train.py*: contains the main code (Sec 3.2 in the paper) for the training process of the utterance-pair coherence scoring model grounded on BERT (Next Sentence Prediction). 
  - *data_utils.py*: contains the main code (Sec 3.1 in the paper) for pseudo training data generation, which will be used to train the coherence scoring model.
  - *model_utils.py*: contains the class of coherence scoring model.
- For evaluation:
  - *segment.py*: contains the main code to conduct evluation procedure based on the TextTiling segmentation framework.
  - *neural_texttiling.py*: contains the detailed implementation of TextTiling with different settings of text encoder (e.g., Bi-encoder, cross-encoder, coherence scoring etc)

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

