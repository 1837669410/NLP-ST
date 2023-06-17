# NLP-ST

A simple and easy to understand NLP teaching

# algorithm

- statistical method: [inverted-index](#para1) | [tfidf](#para2) | [tfidf-sklearn](#para3) | [hmm](#para4)
- word vector: [cbow](#para5) | [skip-gram](#para6)
- sentence vector: [textcnn](#para7) | [seq2seq](#para8) | [seq2seq-cnn](#para9)
- attention: [seq2seq-BahdanauAttention](#para10) | [seq2seq-LuongAttention](#para11) | [transformer](#para12)
- large Language Model: [elmo](#para13) | [gpt](#para14) | [bert](#para15)

## <a id="para1"/>inverted-index(倒排索引)

[code](https://github.com/1837669410/NLP-ST/blob/main/inverted-index.py)

[paper1](https://github.com/1837669410/NLP-ST/blob/main/paper/Keyword-in-Context%20index%20for%20Technical%20Literature.pdf)：Keyword-in-Context Index for Technical Literature

[paper2](https://github.com/1837669410/NLP-ST/blob/main/paper/The%20Inverted%20Multi-Index.pdf)：The Inverted Multi-Index

## <a id="para2"/>tfidf

[code](https://github.com/1837669410/NLP-ST/blob/main/tf-idf.py)

## <a id="para3"/>tfidf-sklearn

[code](https://github.com/1837669410/NLP-ST/blob/main/tf-idf-sklearn.py)

## <a id="para4"/>hmm

[code](https://github.com/1837669410/NLP-ST/blob/main/hmm.py)

## <a id="para5"/>cbow

[code](https://github.com/1837669410/NLP-ST/blob/main/cbow.py)

[paper1](https://github.com/1837669410/NLP-ST/blob/main/paper/Efficient%20Estimation%20of%20Word%20Representations%20in%20Vector%20Space.pdf)：Efficient Estimation of Word Representations in Vector Space

## <a id="para6"/>skip-gram

[code](https://github.com/1837669410/NLP-ST/blob/main/skip-gram.py)

[paper1](https://github.com/1837669410/NLP-ST/blob/main/paper/Efficient%20Estimation%20of%20Word%20Representations%20in%20Vector%20Space.pdf)：Efficient Estimation of Word Representations in Vector Space

## <a id="para7">textcnn

[code](https://github.com/1837669410/NLP-ST/blob/main/textcnn.py)

[paper1](https://github.com/1837669410/NLP-ST/blob/main/paper/Convolutional%20Neural%20Networks%20for%20Sentence%20Classification.pdf)：Convolutional Neural Networks for Sentence Classification

[paper2](https://github.com/1837669410/NLP-ST/blob/main/paper/A%20Sensitivity%20Analysis%20of%20(and%20Practitioners%E2%80%99%20Guide%20to)%20Convolutional.pdf)：A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification

## <a id="para8">seq2seq

[code](https://github.com/1837669410/NLP-ST/blob/main/seq2seq.py)

[paper1](https://github.com/1837669410/NLP-ST/blob/main/paper/Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.pdf)：Sequence to Sequence Learning with Neural Networks

## <a id="para9">seq2seq-cnn

[code](https://github.com/1837669410/NLP-ST/blob/main/seq2seq-cnn.py)

[paper1](https://github.com/1837669410/NLP-ST/blob/main/paper/Convolutional%20Neural%20Networks%20for%20Sentence%20Classification.pdf)：Convolutional Neural Networks for Sentence Classification

## <a id="para10">seq2seq-BahdanauAttention

[code](https://github.com/1837669410/NLP-ST/blob/main/seq2seq-BahdanauAttention.py)

[paper1](https://github.com/1837669410/NLP-ST/blob/main/paper/Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.pdf)：Neural Machine Translation By Jointly Learning To Align And Translate

## <a id="para11">seq2seq-LuongAttention

[code](https://github.com/1837669410/NLP-ST/blob/main/seq2seq-LuongAttention.py)

[paper1](https://github.com/1837669410/NLP-ST/blob/main/paper/Effective%20Approaches%20to%20Attention-based%20Neural%20Machine%20Translation.pdf)：Effective Approaches to Attention-based Neural Machine Translation

## <a id="para12">transformer

[code](https://github.com/1837669410/NLP-ST/blob/main/transformer.py)

[paper1](https://github.com/1837669410/NLP-ST/blob/main/paper/Attention%20Is%20All%20You%20Need.pdf)：Attention Is All You Need

## <a id="para13">elmo

[code](https://github.com/1837669410/NLP-ST/blob/main/elmo.py)

[paper1](https://github.com/1837669410/NLP-ST/blob/main/paper/Deep%20contextualized%20word%20representation.pdf)：Deep contextualized word representation

## <a id="para14">gpt

[code](https://github.com/1837669410/NLP-ST/blob/main/gpt.py)

[paper1](https://github.com/1837669410/NLP-ST/blob/main/paper/Improving%20Language%20Understanding%20by%20Generative%20Pre-Training.pdf)：Improving Language Understanding by Generative Pre-Training

## <a id="para15">bert

[code](https://github.com/1837669410/NLP-ST/blob/main/bert_nextmask.py)

[paper1](https://github.com/1837669410/NLP-ST/blob/main/paper/BERT%20Pre-training%20of%20Deep%20Bidirectional%20Transformers%20for%20Language%20Understanding.pdf)：BERT：Pre-training of Deep Bidirectional Transformers for Language Understanding

Note: There is a slight difference between the model of this Bert and the original paper. I have used an improved training method, which is already written at the beginning of the code. You can check it yourself

# Todo

- Translate a corresponding Pytorch code
- Supporting video explanation (this may take a long time)
- Recruit interested students to complete together

# reference

1、莫烦python [github](https://github.com/MorvanZhou/NLP-Tutorials) | [web](https://mofanpy.com/tutorials/machine-learning/nlp/)

2、动手学深度学习 [web](https://zh.d2l.ai/)

3、李沐老师 [bilibili](https://space.bilibili.com/1567748478/video)
