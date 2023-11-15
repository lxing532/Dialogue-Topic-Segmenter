# Data Hub for Dialogue Topic Segmentation

Welcome to the data hub dedicated to maintaining and organizing open-sourced data resources for Dialogue Topic Segmentation! Hopefully, it will serve as a comprehensive collection point for evaluation datasets related to research and publications specifically for Dialogue Topic Segmentation.

## Why Maintain This Data Hub?
While conducting research on dialogue topic segmentation, I noticed that corpora released by various research groups often present segment labels in different formats. This variability makes it time-consuming to write scripts for unifying these datasets into a format that is easily accessible and usable by my code. Therefore, I believe it would be beneficial to establish a data hub containing all available corpora standardized into a more clear and understandable format. This would save future researchers the effort of having to code for each specific corpus individually.


## Folder Structure

This Folder is organized into two primary directories:

- `./train`:
  - **Purpose**: For replicating experiments detailed in our research.
  - **Content**: Please add the [DailyDial dataset](https://arxiv.org/abs/1710.03957) here. You can download DailyDial from this [link](http://yanran.li/dailydialog).

- `./eval`:
  - **Purpose**: Contains a compilation of all open-source datasets for dialogue topic segmentation.
  - **Content**: Each dataset is formatted identically for ease of use.

### Evaluation Data Format

Each data sample follows a consistent format, as illustrated in the following example:

```json
{
  "dial_id": 0,
  "utterances": [
    "Example utterance 1",
    "Example utterance 2",
    ...
  ],
  "segments": [4, 6, 6, 4, 4],
  "set": "test"
}
```
