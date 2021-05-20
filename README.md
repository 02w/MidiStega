## Steganography course project

Idea: hide information by generating music.

-  **Step1** Train a "language model" for music generation.
-  **Step2** Build a "tree" using the probability distribution learned by the model, then find the leaf node as prediction for next time step according to the bitstream. That's to say, information is hidden in a generated sequence based on the rule by which a "word" is chosen at each time step.

## Reference

- Seq2Seq model: [link](https://teddykoker.com/2020/02/nlp-from-scratch-annotated-attention/)
- midi to words: [Pitch-based representation](https://salu133445.github.io/muspy/representations/index.html)
- hide bitstream: [YangzlTHU/RNN-Stega](https://github.com/YangzlTHU/RNN-Stega)
- data: [maestro](https://magenta.tensorflow.org/datasets/maestro)
- Other
  - [yinoue93/CS224N_proj](https://github.com/yinoue93/CS224N_proj)
  - [mcleavey/musical-neural-net](https://github.com/mcleavey/musical-neural-net)
  - pytorch lightning [docs](https://pytorch-lightning.readthedocs.io)

## See Also
For a pipeline of compress-encrypt-hide, and a demo web application built with [Streamlit](https://streamlit.io/), check out the [`app`](https://github.com/02w/MidiStega/tree/app) branch. Run it with `streamlit run app.py`.

Actually, it's quite simple to train a classifier to determine whether there is secret information hidden in a midi file. ResNet can obtain a pretty good result.

See [ResNet-for-TSC](https://github.com/02w/ResNet-for-TSC) as an example.