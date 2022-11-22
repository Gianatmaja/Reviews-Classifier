# Reviews Classifier in Python
This project involves feature representation of text data, implementation of perceptron algorithm, as well as gradient descent in Python.

### Project Structure
This repository has the following structure

    .
    ├── data                                        # Contains reviews dataset
    │  ├── truecased_reviews_dev.jsonl
    │  ├── truecased_reviews_dev_one.jsonl
    │  ├── truecased_reviews_test.jsonl
    │  ├── truecased_reviews_train.jsonl
    ├── models                            
    │  ├── _init_.py
    │  ├── gd.py                                    # Gradient descent algorithm
    │  ├── perceptron.py                            # Perceptron algorithm
    ├── readers                 
    │  ├── _init_.py
    │  ├── reviews_dataset.py                       # Load reviews dataset function
    ├── _init_.py
    ├── sentiment_classifier.py                     # Performs sentiment analysis
    ├── utils.py
    ├── images
    └── README.md

The main python file is sentiment_classifier.py, where the bulk of the code is implemented. There, we can load the dataset (function implemented in 
reviews_dataset.py), generate feature representations, create an instance of our perceptron algorithm implementation (with functions implemented in the 
perceptron.py file), as well as a gradient descent algorithm (with functions implemented in the gd.py file), using the squared loss and L2-norm 
regulariser.

### Snapshots of Training Process
![Perceptron](https://github.com/Gianatmaja/Reviews-Classifier/blob/main/images/Screenshot%202022-10-10%20at%2010.05.56%20PM.png)

![gd](https://github.com/Gianatmaja/Reviews-Classifier/blob/main/images/Screenshot%202022-10-10%20at%2010.06.05%20PM.png)
