import torch
import torch.optim as optim

def save_checkpoint(neural_network, optimizer, epoch, checkpoint_path):
    '''Saves checkpoint to a given directory, it if it is a folder it must exist'''
    torch.save({'epoch': epoch, 'model_state_dict': neural_network.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),}, checkpoint_path)

def save_checkpoint_2(neural_network, optimizer, epoch, checkpoint_path, scheduler):
    '''Saves checkpoint to a given directory, it if it is a folder it must exist'''
    torch.save({'epoch': epoch, 'model_state_dict': neural_network.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),}, checkpoint_path)

def load_optimizer_checkpoint_cuda(optimizer, checkpoint):
    '''Loads optimizer checkpoint using cuda'''
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

def load_checkpoint(neural_network, optimizer, checkpoint_path, optimizer_device='cuda'):
    '''Loads checkpoint on neuralnetwork and optimizer'''
    checkpoint = torch.load(checkpoint_path)
    neural_network.load_state_dict(checkpoint['model_state_dict'])
    if optimizer_device=='cuda': load_optimizer_checkpoint_cuda(optimizer, checkpoint)