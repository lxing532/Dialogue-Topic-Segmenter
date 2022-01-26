# Codes and data for Dialogue Topic Segmenter
This repository maintains the source code for "[Improving Unsupervised Dialogue Topic Segmentation with
Utterance-Pair Coherence Scoring](https://aclanthology.org/2021.sigdial-1.18.pdf)", SIGDIAL 2021.

## 1. Datasets:
In the paper, we mention two training datasets for Utterance-Pair Coherence Scoring model training. They are:
* [DailyDialog](http://yanran.li/dailydialog) (for English)
* [NaturalConv](https://ai.tencent.com/ailab/nlp/dialogue/#datasets) (for Chinese)
You can access and download these two datasets online!

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
