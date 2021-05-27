# Tacotron2

The Tacotron2 network is used as the main synthesis engine in the SOVA-TTS project. We took its [implementation from NVIDIA](https://github.com/NVIDIA/tacotron2), added various improvements that might be found in articles, and made the code more user-friendly. 

Key differences:  
1. [GST](https://arxiv.org/abs/1803.09017) module is added;
2. Mutual Information Estimator is added (based on the following [article](https://arxiv.org/pdf/1909.01145.pdf) and [repo](https://github.com/bfs18/tacotron2));
3. Added the possibility to include attention loss in the train process (using diagonal or [prealigned](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8703406) guidance);
4. Some work has been done to improve the usability of the code;
5. Other minor changes and additions.

# How to train a new model

First of all you need to install all dependencies (which can be found in the reuqirements.txt) and convert the dataset to the LJ Speech format, where each line contains relative path to the audio file and its text, separated by "|" sign, e.g.:

> wavs/000000.wav|С трев+ожным ч+увством бер+усь я з+а пер+о.

Then divide it into two files: the training list (90% of the data) and the validation list (10% of the data).

After that configure the config file as needed ([here](https://github.com/sovaai/sova-tts-engine/blob/master/data/README.md) you can find an explanation of the main fields of the config file), or just use the default one, filling in the values of parameters `output_dir` (where to save checkpoints), `training_files` (path to the training list), `validation_files` (path to the validation list) and `audios_path` (path to the audio folder, so that together with the relative path to the audio, the full path is obtained).

When everything is ready, launch the training process: 
* in case if you changed hparams.yaml inside the 'data' folder: `python train.py`
* in case if you have some other config file: `python train.py -p path/to/hparams.yaml`
