"""
BSD 3-Clause License

Copyright (c) 2018, NVIDIA Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from enum import Enum

import numpy as np
import torch
from torch import nn

from utils.utils import to_gpu


class AttentionTypes(str, Enum):
    none = "none"
    diagonal = "diagonal"
    prealigned = "prealigned"


    @staticmethod
    def guided_types():
        return [AttentionTypes.diagonal, AttentionTypes.prealigned]


class LossesType(str, Enum):
    MSE = "MSE"
    L1 = "L1"


mel_loss_func = {
    LossesType.MSE: nn.MSELoss,
    LossesType.L1: nn.L1Loss
}


def diagonal_guide(text_len, mel_len, g=0.2):
    gridN = np.tile(np.arange(text_len).reshape((1, -1)), (mel_len, 1))
    gridT = np.tile(np.arange(mel_len).reshape((-1, 1)), (1, text_len))

    W = 1 - np.exp(-(gridN / text_len - gridT / mel_len) ** 2 / (2 * g ** 2))

    return torch.Tensor(W)


def diagonal_loss(predicted, text_len, mel_len, g=0.2):
    guide = to_gpu(torch.Tensor(np.zeros(predicted.shape, dtype=np.float32)))
    guide[:mel_len, :text_len] = diagonal_guide(text_len, mel_len, g)
    return predicted * guide


def prealigned_loss(target, predicted):
    target = target if target.max != 0 else predicted # если матрица выравнивания недоступна - ошибка=0
    return (predicted - target) ** 2


class Tacotron2Loss(nn.Module):
    def __init__(self, hparams):
        self.guided_attention_type = AttentionTypes(hparams.guided_attention_type)
        self.diagonal_factor = None
        if self.guided_attention_type == AttentionTypes.diagonal:
            self.diagonal_factor = hparams.diagonal_factor
        self.attention_weight = hparams.attention_weight

        self.include_padding = hparams.include_padding
        self.mel_loss_type = LossesType(hparams.mel_loss_type)

        self.gate_positive_weight = torch.tensor(hparams.gate_positive_weight)

        super().__init__()


    def forward(self, model_output, targets):
        mels_target = targets.mels
        mels_target.requires_grad = False

        mels = model_output.mels
        mels_postnet = model_output.mels_postnet

        mel_loss = mel_loss_func[self.mel_loss_type]()(mels, mels_target) + \
                   mel_loss_func[self.mel_loss_type]()(mels_postnet, mels_target)

        gate_target = targets.gate
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        gate_out = model_output.gate.view(-1, 1)

        gate_loss = nn.BCEWithLogitsLoss(pos_weight=self.gate_positive_weight)(gate_out, gate_target)

        return mel_loss, gate_loss


class AttentionLoss(nn.Module):
    def __init__(self, hparams):
        self.guided_attention_type = AttentionTypes(hparams.guided_attention_type)
        self.diagonal_factor = None
        if self.guided_attention_type == AttentionTypes.diagonal:
            self.diagonal_factor = hparams.diagonal_factor
        self.attention_weight = hparams.attention_weight

        self.include_padding = hparams.include_padding

        super().__init__()


    def forward(self, targets, predictions, text_lengths, mel_lengths):
        batchsize = len(text_lengths)

        attention_loss = 0

        if self.guided_attention_type == AttentionTypes.none:
            return
        else:
            text_lengths = text_lengths.cpu().numpy() if isinstance(text_lengths, torch.Tensor) else text_lengths
            mel_lengths = mel_lengths.cpu().numpy() if isinstance(mel_lengths, torch.Tensor) else mel_lengths

            lengths = list(zip(text_lengths, mel_lengths))

            if self.guided_attention_type == AttentionTypes.diagonal:
                for i, text_mel in enumerate(lengths):
                    text_length, mel_length = text_mel

                    attention_loss += torch.sum(
                        diagonal_loss(
                            predicted=predictions[i],
                            text_len=text_length,
                            mel_len=mel_length,
                            g=self.diagonal_factor
                        )
                    )

            elif self.guided_attention_type == AttentionTypes.prealigned:
                for i in range(batchsize):
                    if targets[i].max == 0:
                        targets[i] = predictions[i]  # если матрица выравнивания недоступна - ошибка=0

                attention_loss += nn.MSELoss(reduction="sum")(predictions, targets)

            else:
                raise TypeError

            if not self.include_padding:
                active_elements = sum([text_len * mel_len for text_len, mel_len in lengths])
            else:
                active_elements = batchsize * max(text_lengths) * max(mel_lengths)

            attention_loss = attention_loss / active_elements * self.attention_weight

            return attention_loss