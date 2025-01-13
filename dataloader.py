import torch
from optimum.habana import GaudiTrainer

class CustomTrainer(GaudiTrainer):
    '''
    example of a custom trainer that inherits from GaudiTrainer.
    This class is used to define the loss function and the prediction step.
    However, you can also define other custom Huggingface Trainer functions.
    '''
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids, labels, attention_mask = inputs
        outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)
    
