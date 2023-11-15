# Data Hub for Dialogue Topic Segmentation
Welcome to the data hub dedicated to maintaining and organizing open-sourced data resources for Dialogue Topic Segmentation! Hopefully, it will serve as a comprehensive collection point for evaluation datasets related to research and publications specifically for Dialogue Topic Segmentation.


## 💡 Why Maintain This Data Hub?
> While conducting research on dialogue topic segmentation, I noticed that corpora released by various research groups often present segment labels in different formats. This variability makes it time-consuming to write scripts for unifying these datasets into a format that is easily accessible and usable by my code. Therefore, I believe it would be beneficial to establish a data hub containing all available corpora standardized into a more clear and understandable format. This would save future researchers the effort of having to code for each specific corpus individually.


## 💡 Folder Structure
This Folder is organized into two primary directories:

- `./train`:
  - **Purpose**: For replicating experiments detailed in [SIGDIAL-21 paper](https://www.lz-xing.com/assets/publications/2021_sigdial/paper.pdf).
  - **Content**: Please add the [DailyDial dataset](https://arxiv.org/abs/1710.03957) here. You can download DailyDial from this [link](http://yanran.li/dailydialog).

- `./eval`:
  - **Purpose**: Contains a compilation of all open-source datasets I found for dialogue topic segmentation.
  - **Content**: Each dataset is formatted identically for ease of use as illustrated in the following example:

```json
{
    "dial_id": 0,
    "utterances": [
        "check the weather for the 7 day forecast",
        "What city are you interested in?",
        "Los Angeles, please. Will it be hot?",
        "It will be hot today in Los Angeles.",
        "Yes, can you give me the information on the Huntingdon Marriott Hotel?",
        "Absolutely. It is an expensive hotel located in the west part of town. It has 4 starts and includes free wifi and parking. Would you like help booking a room?",
        "Yes please, I need a reservation for 6 people for 5 nights starting on Saturday.",
        "Sorry, there are not enough rooms available for that time period. Perhaps a different day or a shorter stay might yield better results.",
        "How about for 1 night? If that works, I'll need a reference number of course.",
        "Booking was successful. Your reference number is : OO8QDA62.",
        "I need a train from London Kings Cross to Cambridge.",
        "I have 70 trains travelling that route. To narrow it down, what day would you like to leave and what time would you like to depart/arrive?",
        "sure, I would like to go on Saturday, and arrive by 20:20. As close to that time as I can arrive.",
        "I have a 19:17 from London that arrives at 20:08. Would that work for you?",
        "Yes, let me have 7 tickets, please.",
        "Booking was successful, the total fee is 132.16 GBP payable at the station. Reference number is : QMD5P3EG. Is there anything else I can help you with?",
        "I need to find a shopping center.",
        "The Stanford Shopping Center at 773 Alger Dr is 3 miles away in no traffic. Would you like directions there?",
        "Yes please.",
        "I sent all the info on the screen, please drive carefully!",
        "Schedule meeting.",
        "What details shall I add to your meeting reminder?",
        "Please set it for the 13th at 11am. It is going to be with management and to discuss our company picnic. Thank you.",
        "Reminder set for your meeting at 11am on the 13th with management to discuss your company picnic. Is there anything else?"
    ],
    "segments": [
        4,
        6,
        6,
        4,
        4
    ],
    "set": "test"
}
```
For this instance, the label of segments is presented in the way of numbers of conserctive sentences. Specifically, [4,6,6,4,4] means this dialogue should be segmented into Seg1 (4 utterances), Seg2 (6 utterances), Seg3 (6 utterances), Seg4 (4 utterances), Seg5 (4 utterances).


## 💡 Train or Validation or Test?
If a corpus comes with an original division into train, validation (dev), and test sets, I will adhere to this structure by assigning the value of 'set' as 'train', 'dev', or 'test' accordingly. In cases where such a division is not provided, I will randomly allocate 1% of the data as the validation (dev) set and designate the remaining samples as test samples.


## 💡 Contributing and Future Plans
We plan to continually expand and update this dataset collection. If you have developed or discovered new datasets relevant to dialogue topic segmentation, we welcome your contributions by pull request:
1. **Format Syncing:** Format your dataset to match our standard structure and submit it.
2. **Original Data Submission:** If format conversion is challenging and you don't have time for it, feel free to submit the original dataset. We will assist in aligning it with our standard format.
