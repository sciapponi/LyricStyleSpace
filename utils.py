
def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    # Train your model
    for batch_idx,batch in enumerate(tqdm(train_loader)):
        # Extract the input ids and attention masks from the batch
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Encode the inputs using the pre-trained model
        embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
        # print(embeddings)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(
                "\tEpoch {} Iteration {}:  Number of mined triplets = {}".format(
                    epoch, batch_idx, mining_func.num_triplets
                )
            )

    # Print the loss every epoch
    # print('\tEpoch [{}/{}], Loss: {}'.format(epoch, epochs, loss.item()))

def get_all_embeddings(dataloader, model):
  model.eval()
  embeddings, labels = [], []
  with torch.no_grad():
    for idx, batch in enumerate(tqdm(dataloader)):
      input_ids, attention_mask, label = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
      embeddings.append(model(input_ids=input_ids, attention_mask=attention_mask))
      labels.append(label)

  return torch.vstack(embeddings), torch.cat(labels)

def test(train_loader, test_loader, model, accuracy_calculator):
  train_embeddings, train_labels = get_all_embeddings(train_loader, model)
  test_embeddings, test_labels = get_all_embeddings(test_loader, model)
  accuracies = accuracy_calculator.get_accuracy(test_embeddings, test_labels, train_embeddings, train_labels)

  print(f"Test set accuracy (Precision@1) = {accuracies['precision_at_1']}")
