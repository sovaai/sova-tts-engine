# Tacotron2

Here you can find modified version of the [Tacotron2 repository from NVIDIA](https://github.com/NVIDIA/tacotron2).

Key differences:  
1. [GST](https://arxiv.org/abs/1803.09017) module is added;
2. Mutual Information Estimator is added (based on the following [article](https://arxiv.org/pdf/1909.01145.pdf) and [repo](https://github.com/bfs18/tacotron2));
3. Added the possibility to include attention loss in the train process (using diagonal or [prealigned](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8703406) guidance);
4. Some work has been done to improve the usability of the code;
5. Other minor changes and additions.