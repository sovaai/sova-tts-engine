from enum import Enum

from torch.optim import SGD, Adam
import torch_optimizer as optim


class OptimizersTypes(str, Enum):
    sgd = "sgd"
    yogi = "yogi"
    adam = "adam"
    radam = "radam"
    diffgrad = "diffgrad"
    novograd = "novograd"
    adabound = "adabound"


optimizers = {
    OptimizersTypes.sgd: SGD,
    OptimizersTypes.yogi: optim.Yogi,
    OptimizersTypes.adam: Adam,
    OptimizersTypes.radam: optim.RAdam,
    OptimizersTypes.diffgrad: optim.DiffGrad,
    OptimizersTypes.novograd: optim.NovoGrad,
    OptimizersTypes.adabound: optim.AdaBound
}

optimizers_options = {
    OptimizersTypes.sgd: ["momentum", "dampening", "nesterov"],
    OptimizersTypes.yogi: ["betas", "eps", "initial_accumulator"],
    OptimizersTypes.adam: ["betas", "eps", "amsgrad"],
    OptimizersTypes.radam: ["betas", "eps"],
    OptimizersTypes.diffgrad: ["betas", "eps"],
    OptimizersTypes.novograd: ["betas", "eps", "grad_averaging", "amsgrad"],
    OptimizersTypes.adabound: ["betas", "eps", "final_lr", "gamma", "amsbound"]
}


def build_optimizer(model, hparams):
    optimizer_type = OptimizersTypes[hparams.optimizer]
    optim_options = {} if hparams.optim_options is None else hparams.optim_options

    if optimizer_type in OptimizersTypes:
        if not all(arg in optimizers_options[optimizer_type] for arg in optim_options):
            raise ValueError("You tried to pass options incompatible with {} optimizer. "
                             "Check your parameters according to the description of the optimizer:\n\n{}".
                             format(optimizer_type, optimizers[optimizer_type].__doc__))

        optimizer = optimizers[optimizer_type](
            model.parameters(),
            lr=hparams.learning_rate,
            weight_decay=hparams.weight_decay,
            **optim_options
        )
    else:
        raise ValueError(f"`{optimizer_type}` is not a valid optimizer type")

    if hparams.with_lookahead:
        optimizer = optim.Lookahead(optimizer, k=5, alpha=0.5)

    return optimizer