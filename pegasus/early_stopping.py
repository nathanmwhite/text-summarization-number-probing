# Text summarization number probing
# Original code Copyright © 2022 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

class Early_Stopping():
  
    def __init__(self, min_delta=0.0, patience=10):
        self.num_iterations_elapsed = 0
        self.min_delta = min_delta
        self.patience = patience
        self.early_stopping = False
        self.last_best = None
        
    def __call__(self, current_loss):
        if self.last_best == None:
            self.last_best = current_loss
        elif self.last_best - current_loss > self.min_delta:
            self.last_best = current_loss
            self.num_iterations_elapsed = 0
        elif self.last_best - current_loss < self.min_delta:
            self.num_iterations_elapsed += 1
            if self.num_iterations_elapsed >= self.patience:
                self.early_stopping = True
