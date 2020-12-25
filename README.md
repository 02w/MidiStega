## Steganography course project

Idea: hide information by generating music.

-  **Step1** Train a "language model" for music generation.
-  **Step2** Build a tree using the probability distribution learned by the model, then find the leaf node as prediction for next time step according to the bitstream.

## Reference

- Seq2Seq model: [link](https://teddykoker.com/2020/02/nlp-from-scratch-annotated-attention/)
- midi to words: [colab link](https://colab.research.google.com/github/cpmpercussion/creative-prediction/blob/master/notebooks/3-zeldic-musical-RNN.ipynb)
- hide bitstream: [YangzlTHU/RNN-Stega](https://github.com/YangzlTHU/RNN-Stega)
- Other
  - [yinoue93/CS224N_proj](https://github.com/yinoue93/CS224N_proj)
  - pytorch lightning [docs](https://pytorch-lightning.readthedocs.io)
  - [mcleavey/musical-neural-net](https://github.com/mcleavey/musical-neural-net)

