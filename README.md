# TransCA

## 1. Instructions

TransCA is designed to accurately predict the traffic flow on each link in a mass ride-sharing traffic network. The user inputs include the traffic demand of car and non-car travellers (i.e., the travellers owing cars and the opposite) at each origin-destination pair and the structure of the network. The output is a prediction of traffic flow for each link of the entire ride-sharing traffic network.\
TransCA is programmed in **python3.8**.

## 2. Installation

*   **Necessary packages**\
    `numpy` `pandas` `openyxl` in order to calculate the ridesharing traffic flow.\
    `matplotlib` `tkinter` in order to build the graphical user interface.

*   **Download the code**\
    You can download the code from this repository including `app_TransCAv1.1.py` and `TransCA.py`.\
    `app_TransCAv1.1.py` can calculate the ridesharing traffic flow with a graphical user interface.\
    `TransCA.py` have the the same functionality as the former code but without GUI.

    The operating instruction of the app is showed in   [OI-of-TransCA.pdf](./OI-of-TransCA.pdf).

## 3. Test data

We use an urban transportation network of Sioux-Falls City, South Dakota, which has 24 nodes, 76 links, and 528 OD pairs, as an example to further demonstrate the software of TransCA.

Test data is showed in [SiouxFalls_Network.xlsx](./data/SiouxFalls_Network.xlsx).


## 4. Contact us 
If you find problems or bugs, you can leave your comments and send us emails. Here are the emails.
* majie@seu.edu.cn
* wangjiacheng@seu.edu.cn

