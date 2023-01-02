from tqdm import tqdm
import torch
from torch import nn
from utils.label_dicts import label2class
from utils.path_utils import join_path

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


def save_predictions(predictions, save_path):
    predictions_path = join_path(save_path, 'best_test_predictions.txt')
    start_index = 8001
    with open(predictions_path, 'w') as file:
        for i, prediction in enumerate(predictions):
            file.write(f'{start_index+i}\t{label2class[prediction]}\n')


def train(config, model, optimizer, logger, scorer, train_loader, test_loader, save_path, lr_scheduler=None):
    max_f1_score = 0

    model.to(device)
    
    for epoch in range(config.epochs):

        data_iterator = tqdm(train_loader, desc='Train')
        train_loss = 0

        for _, (batch_data, batch_labels) in enumerate(data_iterator):
            
            model.train()
            data = batch_data.to(device)
            label = batch_labels.to(device)

            optimizer.zero_grad()
            batch_loss, _ = model(data, label)
            train_loss += batch_loss.item()
            batch_loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.gradient_clip
            )
            optimizer.step()
            lr_scheduler.step()

        train_loss = train_loss / len(train_loader)
        logger.logging_train(epoch, train_loss, 0)

        test_loss, test_predictions, f1_score = test(model, scorer, test_loader)
        logger.logging_test(epoch, test_loss, 0, f1_score)

        if f1_score > max_f1_score:
            max_f1_score = f1_score
            model.save_model(save_path)
            save_predictions(test_predictions, save_path)



def test(model, scorer, test_loader):
    test_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():

        model.eval()
        data_iterator = tqdm(test_loader)
        
        for _, (batch_data, batch_labels) in enumerate(data_iterator):

            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            true_labels.extend(batch_labels.cpu().detach().numpy().tolist())

            loss, logits = model(batch_data, batch_labels)
            test_loss += loss.item()

            predictions_i = torch.argmax(logits, dim=1)
            predictions_i = predictions_i.cpu().detach().numpy()
            predictions.extend(predictions_i.tolist())
            
    test_loss = test_loss / len(test_loader)
    f1_score = scorer.compute_f1_score(predictions, true_labels)

    return test_loss, predictions, f1_score
