from torch.optim.lr_scheduler import LambdaLR


def get_transformer_scheduler(
        optimizer,
        num_warmup_steps,
        last_epoch=-1
):
    def lr_lambda(current_step: int):
        current_step += 1
        return min(current_step ** (-0.5),
                   current_step * num_warmup_steps ** (-1.5))

    return LambdaLR(optimizer, lr_lambda, last_epoch)



def get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        init_eps: float = 0.1,
        last_epoch=-1
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            if current_step == 0:
                return init_eps * (1 / num_warmup_steps)
            else:
                return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

