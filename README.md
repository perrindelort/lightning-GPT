# Lightning GPT

This repo is an implemention of GPT from _Attention is all you need_ following [Andrej Karpathy's tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY), that I can only encourage you to watch :)

# So it's just copy-pasted from his Github repo ?

Obviously not ! My goal here was to deepen my knowledge and learn new things ! Thus, I wrote the code while following the video and added some changes :
- Notably, this implementation is made using [Pytorch Lightning](https://github.com/Lightning-AI/pytorch-lightning/), a powerful and ergonomic library, made to better organize Pytorch code !
- A tensorboard can be used to better monitor the training
- I also added some [Einops'](https://github.com/arogozhnikov/einops) uses. This library is well known and widely used by [experts](https://x.com/karpathy/status/1290826075916779520) to have a more readable code regarding Pytorch's tensors operations.

# Showcase

As GPT first of its name isn't very efficient, its output is gibberish — Shakespearean-style gibberishin, verily, due to our data ! — though it's still oddly entertaining to watch.

## Infinite generation
The infinite generation streamed to the console

https://github.com/user-attachments/assets/d70bb25c-e569-48c9-b131-994fbfbab376

## Output evolution
<p align="center">
  <img src="assets/output_evolution.gif" alt="Demo GIF" />
</p>

# Getting started
## Installing the package
Create a venv and run

 ```pip install .```

## Training a model

To train a model, simply run:

 ```python -m lightning_gpt fit --config configs/{model}.yaml```

where model is either 'blm' or 'gpt'.

## Infer a model

To generate our shakepsearean-gibberish with a model, simply run:

 ```python -m lightning_gpt fit --config configs/{model}.yaml```

where model is either 'blm' or 'gpt'.
