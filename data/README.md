# Config explanation
To train a new model, you need to understand what the fields in the config file mean. Many of them are taken from the NVIDIA repository, and many have been added by us. Below are the parameters of the configuration file that you most likely want to change.

## Experiment Parameters
`device` - device on which calculations will be performed in pytorch format, e.g. 'cuda:0'  

`epochs` - number of training epochs  
`iters_per_checkpoint` - number of iterations before validation and saving the checkpoint   

`output_dir` - folder where the checkpoints and logs will be saved  
`log_dir` - subfolder inside the `output_dir`  

`checkpoint` - path to the pretrained model (optional)  
`warm_start` - if True, the parameters of the optimizer, the lr scheduler, and the layers from the parameter `ignore_laysers` will not be loaded from the checkpoint  

`seed` - just seed  
`fp16_run` - if True, calculations with fp16 accuracy will be performed where possible   

`ignore_laysers` - layers that should not be loaded from the checkpoint (only when `warm_start==True`)  
`ignore_mismatched_layers` - if True, layers from the checkpoint whose dimension differs from the layers of the same name in the model will not be loaded  

## Data Parameters  
`load_mel_from_disk` - if True, mel spectrograms will be downloaded from disk and not calculated on the go  
`audios_path` - path to the audio files folder  
`training_files` - path to the list of files for training   
`validation_files` - path to the list of files for validation  

`charset` - character set, using for text encoding (for more information, see the nlp-preprocessor documentation)   
`use_basic_handler`  - whether to use basic or full functionality for chosen charset (language)

`mask_stress` -  Union[float, bool]. If float, the number must be in {0, 1} - the probability of masking stressed words into learning process (passing them unstressed)  
`mask_phonemes` -  Union[float, bool]. If float, the number must be in {0, 1} - the probability of masking phoneme representation of words into learning process (passing them in grapheme representation; works only if phonetization module is presented in the text preprocessing)  
`word_level_prob` - if True, the stress and phoneme masking will be applied to individual words, otherwise to the entire sentence.

`shuffle` - whether to shuffle the data for training  
`optimize` - if true, then the training batches will be formed from sentences that are close in length  
`len_diff` - what difference in length is acceptable for sentences in a single batch  

## Audio Parameters
`add_silence` - whether to add a small amount of silence to the end of the audio (helps more stable work of the gate layer)  
`trim_silence` - whether to cut out sections of silence (including inside audio)  
`trim_top_db` - the level at which silence is cut off (too low level can lead to the loss of useful information)  

## Model Parameters
`activation` - used activation function (prenet, encoder); possible entries: 'linear', 'relu', 'leaky_relu', 'selu', 'tanh'  

`use_gst` - whether to use the [GST](https://github.com/sovaai/sova-tts-engine/blob/master/modules/gst.py) module  
`reference_encoder_activation` used activation function (gst reference encoder); possible entries: 'linear', 'relu', 'leaky_relu', 'selu', 'tanh'  

`stl_token_num` - number of style tokens in the GST module  
`stl_num_heads` - number of heads of the multi head attention layer in the GST module  

## Optimization Hyperparameters
`guided_attention_type` - way of regulating the mechanism of attention; possible entries: 'none', 'diagonal', 'prealigned'  
`attention_weight` - coefficient of participation loss of the attention in calculating the total loss  
`diagonal_factor` - 0.15 (the greater the `diagonal factor`, the less the penalty for non-diagonality)  

`optimizer` - possible entries: 'sgd', 'adam', 'radam', 'diffgrad', 'novograd', 'yogi', 'adabound'; used optimizers from [pytorch](https://pytorch.org/docs/stable/optim.html#algorithms) and [pytorch-optimizer](https://github.com/jettify/pytorch-optimizer) repo  
`learning_rate` - learning rate  
`weight_decay` - weight regularization  
`optim_options` - additional options for different types of optimizers ([possible parameters](https://github.com/sovaai/sova-tts-engine/blob/master/modules/optimizers.py#L28))  

`with_lookahead` - whether to activate lookahead wrapper around the optimizer, which stabilizes exploration of the loss surface and improves convergence; examples: Ranger = RAdam + LookAhead  

`lr_scheduler` - learning rate schedulers; possible entries: 'none', 'multi_step', 'exp', 'plateau', 'cyclic'; used schedulers from [pytroch](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)  
`lr_scheduler_options` - additional options for different types of learning rate schedulers ([possible parameters](https://github.com/sovaai/sova-tts-engine/blob/master/modules/optimizers.py#L103))  
`restore_scheduler_state` -   
`batch_size` - batch size; with `fp16_run==True` is is possible to set larger values for this parameter  

`initscheme` - weight initialization scheme  

## MMI options 
`use_mmi` - whether to use the [MMI](https://github.com/sovaai/sova-tts-engine/blob/master/modules/mmi.py) module  
`use_gaf` - whether to use gradient adaptive factor (working with `use_mmi==True` only)  
`max_gaf` - minimal value of the gradient adaptive factor  

## Teacher forcing control
`tf_replacement` - type of the mechanism that limits usage of the teacher forcing; possible entries:
  * 'none' - teacher forcing is always used
  * 'global_mean' - some frames will be replaced by dataset global mean value
  * 'decoder_output' - some frames will be replaced by decoder outputs from previous step

`p_tf_train` - probability with which frames will be treated in conventional teacher forcing mode during training  
`p_tf_val` - probability with which frames will be treated in conventional teacher forcing mode during validation 

`global_mean_npy` - path to the numpy file, containing global mean value for the dataset
