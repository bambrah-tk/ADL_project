# ADL_project
Applied Deep Learing Project WS2023 <br>
Taranpreet Kaur Bambrah 11717209

## Assignment 2: Hacking
### Clarification on Assignment Misunderstanding
In the early stages of the project, there was a misinterpretation of the initial assignment's requirements. The assignment called for the identification of a paper that addressed our Optical Character Recognition (OCR) problem—specifically, recognizing words—using a traditional machine learning algorithm. The subsequent objective was to surpass the performance of the traditional approach by implementing a deep learning algorithm and then conduct a comparative analysis. <br>
Regrettably, the misunderstanding led to a divergence in the planned approach. Rather than identifying a specific paper as a benchmark, the focus initially shifted towards surpassing a generic traditional OCR method. Consequently, the project currently lacks a predetermined traditional algorithm for direct comparison. <br>
I am engaged in an ongoing search to identify a suitable traditional OCR paper that aligns with the assignmnet requirements.

### Evaluation of the Project Scope and Shifiting Focus to Word Recognition
Initially, the project was conceived with the intention of implementing Optical Character Recognition (OCR) specifically tailored for recognizing digits. <br>
In the preliminary stages, experiments were conducted using the MNIST dataset, a well-known benchmark for digit recognition. Convolutional Neural Networks (CNNs) were employed to capture spatial hierarchies and patterns within the digit images. <br>
As the project unfolded, a pivotal decision was made to expand the scope beyond digit recognition. The transition from recognizing individual digits to complete words introduced a richer and more complex dimension to the OCR task. This shift was inspired by the recognition of the broader applications of OCR in real-world scenarios, where interpreting entire words is often more relevant than isolated digits. <br>
To explore word recognition, the IAM Handwritten Dataset was introduced to the project. This dataset contains handwritten samples of words, each already segmented into individual units. The decision to leverage this dataset was driven by its compatibility with the revised project objective, allowing for a seamless transition to word recognition. <br>
The IAM Handwritten Dataset is a collection of handwritten text samples that have been transcribed and segmented into individual words, providing a well-curated dataset for the task of word recognition. <br>

In the scope of this project, the focus is explicitly on the text recognition component of Optical Character Recognition (OCR), rather than addressing the text detection step. <br>

### References
[Handwritten-word-recognition-OCR----IAM-dataset---CNN-and-BiRNN](https://github.com/naveen-marthala/Handwritten-word-recognition-OCR----IAM-dataset---CNN-and-BiRNN) <br>
[handwritten_text_detection_and_recognition](https://github.com/furqan4545/handwritten_text_detection_and_recognition/tree/master) <br>
[Word-recognition](https://www.kaggle.com/code/prasadmshaivas/word-recognition) <br>

## Topic: Optical Character Recognition (OCR) (Assignment 1 Initiate)
**Optical Character Recognition (OCR)** is a classic problem in computer vision where the goal is to recognize and interpret text from images or scanned documents. Traditional OCR methods often involve complex pre-processing techniques, feature engineereing and classical machine learning algortihms such as SVM. <br>
In this project, the goal would be to leverage deep learning techniques to beat the performance of traditional OCR methods. So the type of this project would be **"Beat the classics"**.  <br>
The datasets that I would be using are the famous MNIST (handwritten digits) and EMNIST (handwrittten letters). <br>
The idea would be to develop a robust OCR system capable of recognizing both handwritten digits and handwritten letters accurately. Therefore I would be using CNNs and LSTM.

## Work-breakdown structure:
- Data Preprocessing: 5 days
- Model Design and Architecture Selection: 5 days
- Training and Fine-Tuning:  10 days
- Evaluation and Comparison: 2 days
- Improvement Iterations: 5 days
- Building Application : 10 days
- Preparing Presentation: 2 days 
- Documentation and Writing Final Report: 3 days

(days doesn't mean full days, it means spending some hours at a day)

## References to scientific papers
[Optical Character Recognition via Deep Learning](https://cs230.stanford.edu/files_winter_2018/projects/6910235.pdf) <br>
[Optical character recognition using deep learning](https://dspace5.zcu.cz/bitstream/11025/48953/1/Thesis___Pavel_Andrlik.pdf) <br>
[A Survey of Deep Learning Approaches for OCR and Document Understanding](https://ml-retrospectives.github.io/neurips2020/camera_ready/29.pdf) <br>
[Optical Character Recognition using Deep learning – A Technical Review](https://www.researchgate.net/publication/326009476_Optical_Character_Recognition_using_Deep_learning_-_A_Technical_Review) <br>

## Ideas for me
- experiment with CNN and RNN or maybe also transformer based models for sequence recogntiton
- use appropriate loss functions CTC
- evaluation- depends on the paper
- data augmentation with rotations, scaling and noise  <br>
.... to be continued
