import os
import logging
import torch
import csv

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def initialize_logger(log_dir):
    """Initialize logger for training"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create file handler
    fh = logging.FileHandler(log_dir)
    fh.setLevel(logging.INFO)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def save_checkpoint(outf, epoch, iteration, model, optimizer):
    """Save model checkpoint"""
    checkpoint_path = os.path.join(outf, f'checkpoint_epoch_{epoch}.pth')
    best_model_path = os.path.join(outf, 'best_rasnet_model.pth')
    
    checkpoint = {
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    torch.save(checkpoint, checkpoint_path)
    torch.save(model.state_dict(), best_model_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def record_loss1(loss_csv, epoch, iteration, epoch_time, lr, train_loss, val_loss):
    """Record loss to CSV file"""
    writer = csv.writer(loss_csv)
    writer.writerow([epoch, iteration, epoch_time, lr, train_loss, val_loss])
    loss_csv.flush()