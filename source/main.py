# %%
from data_loading import *
from timbre_CNN import *
from evaluation import *
from torch.utils.data import DataLoader, sampler

saved_model_path = "model_test.pth"

# Hyperparameters
batch_size = 128
epochs = 20
learning_rate = 0.001
loss_function = nn.BCELoss()


if __name__ == '__main__':
    # %%
    print("\n----------------LOADING DATA----------------")
    loader = InstrumentLoader(data_dir)

    data = loader.preprocess()

    dataset = TimbreDataset(data)
    train_indices, test_indices = generate_set_indices(len(data), partition_ratio=0.8)
    train_sampler = sampler.SubsetRandomSampler(train_indices)
    test_sampler = sampler.SubsetRandomSampler(test_indices)

    loader_train = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler)
    loader_test = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler)

    # Calculate training and test set statistics
    print(len(train_sampler), "training samples")
    print(len(test_sampler), "test samples")
    train_class_balance = dataset[train_sampler.indices][1].sum(axis=0)/len(train_sampler)
    test_class_balance = dataset[test_sampler.indices][1].sum(axis=0)/len(test_sampler)
    print("Train set contains", np.round(train_class_balance * 100), "% Upright pianos")
    print("Test set contains", np.round(test_class_balance * 100), "% Upright pianos")
    # %%
    print("\n----------TRAINING/LOADING MODEL----------")
    if not os.path.isfile(saved_model_path):
        print("Creating and training new model")
        model = TimbreCNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        with torch.enable_grad():
            model.train(True)
            loss_train_log = []
            for epoch in range(epochs):
                running_loss = 0.0
                for i, batch in enumerate(loader_train):
                    x = batch[0].float()
                    label = batch[1].float()

                    optimizer.zero_grad()
                    y = model(x)
                    loss = loss_function(y, label)

                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                # Record training loss
                print("+Training - Epoch", epoch+1, "loss:", running_loss/(batch_size*(i+1)))
                loss_train_log.append(running_loss/(batch_size*(i+1)))
        # Plot training curves
        plt.plot(range(1, epochs + 1), loss_train_log, c='r', label='train')
        #plt.plot(epoch_val_log, loss_val_log, c='b', label='val')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid()
        plt.show()
        # %%
        # Save model
        torch.save(model, saved_model_path)
        print("Saved trained model to", saved_model_path)
    else:
        print("Loading model from", saved_model_path)
        model = torch.load(saved_model_path)
    print(model)
    model.count_parameters()

    print("\n-------------EVALUATING MODEL-------------")
    labels_acc = np.empty(0, dtype=int)
    preds_acc = np.empty(0, dtype=int)
    with torch.no_grad():
        # Inference mode
        model.train(False)

        for batch in loader_test:
            x = batch[0].float()
            label = batch[1].float()
            y = model(x)
            print("+Evaluating - Batch loss:", loss_function(y, label).item())
            pred = torch.round(y)
            # Accumulate per-batch ground truths and outputs
            labels_acc = np.append(labels_acc, label)
            preds_acc = np.append(preds_acc, pred)

    evaluate(labels_acc, preds_acc)

