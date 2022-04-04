# Reading Comprehension Question Answering System

Training an LSTM/Attention based neural network on the [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) dataset. Give the model a passage and ask it questions for which the answer lies in the passage. The model is able to find the answer in the passage. 

## References

The data preprocessing part (i.e. the files setup.py and args.py), has been borrowed from https://github.com/chrischute/squad.

Model architecture used is from the paper https://arxiv.org/pdf/1704.00051.pdf -

```
@inproceedings{chen2017reading,
  title={Reading {Wikipedia} to Answer Open-Domain Questions},
  author={Chen, Danqi and Fisch, Adam and Weston, Jason and Bordes, Antoine},
  booktitle={Association for Computational Linguistics (ACL)},
  year={2017}
} 
```
