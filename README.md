# Lightning GPT

This repo is an implemention of GPT from _Attention is all you need_ following [Andrej Karpathy's tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY), that I can only encourage you to watch :)

# So it's juste copy-pasted from his Github repo ?

Obviously not ! My goal here was to deepen my knowledge and learn new things ! Thus, I wrote the code while following the video and added some changes :
- Notably, this implementation is made using [Pytorch Lightning](https://github.com/Lightning-AI/pytorch-lightning/), a powerful and ergonomic library, made to better organize Pytorch code !
- A tensorboard can be used to better monitor the training
- I also added some [Einops'](https://github.com/arogozhnikov/einops) uses. This library is well known and widely used by [experts](https://x.com/karpathy/status/1290826075916779520) to have a more readable code regarding Pytorch's tensors operations.

# TODO
- [ ] Finish Tensorboard
- [ ] Save weights
- [ ] Load existing weights
- [ ] LRScheduler
- [ ] Another dataset
- [ ] Test using [Tiktoken](https://github.com/openai/tiktoken) and [SentencePiece](https://github.com/google/sentencepiece)
- [ ] Add env files
- [ ] Add Docker file
