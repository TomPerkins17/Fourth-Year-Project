# %%
from data_loading import *
from timbre_CNN import *
from evaluation import *
from torch.utils.data import DataLoader, sampler

saved_model_path = "model.pth"
val_interval = 4

# Hyperparameters
batch_size = 128
epochs = 20
learning_rate = 0.001
loss_function = nn.BCELoss()


def generate_split_indices(data, partition_ratios=None, mode="mixed", seed=None):
    # Make a random set of shuffled indices for sampling training/test sets randomly w/o overlap
    if partition_ratios is None:
        partition_ratios = [0.8, 0.1]
    rng = np.random.default_rng(seed=seed)
    if mode == "segment_instruments":

        instruments = data.instrument.unique()
        rng.shuffle(instruments)

        i = 0
        indices_train = []
        indices_val = []
        indices_test = []
        no_more_instruments = False
        # Iterate through instruments and add them to the training/validation set indices until ratios are reached
        next_instrument_indices = np.asarray(data.instrument == instruments[i]).nonzero()[0]
        while (len(indices_train) + len(next_instrument_indices))/len(data) <= partition_ratios[0]:
            indices_train = np.append(indices_train, next_instrument_indices)
            i += 1
            if i >= len(instruments):
                no_more_instruments = True
                break
            next_instrument_indices = np.asarray(data.instrument == instruments[i]).nonzero()[0]
        while (len(indices_train) + len(indices_val) + len(next_instrument_indices))/len(data) \
                <= partition_ratios[0] + partition_ratios[1] \
                and not no_more_instruments:
            indices_val = np.append(indices_val, next_instrument_indices)
            i += 1
            if i >= len(instruments):
                break
            next_instrument_indices = np.asarray(data.instrument == instruments[i]).nonzero()[0]
        for j in range(i, len(instruments)):
            indices_test = np.append(indices_test, np.asarray(data.instrument == instruments[j]).nonzero())
        print("")
    elif mode == "mixed":
        # Reproducible random shuffle of indices, using a fixed seed
        indices = np.arange(len(data))
        rng.shuffle(indices)

        split_point_train = int(len(data) * partition_ratios[0])
        split_point_val = split_point_train + int(len(data) * partition_ratios[1])
        indices_train = indices[:split_point_train]
        indices_val = indices[split_point_train:split_point_val]
        indices_test = indices[split_point_val:]
    else:
        raise Exception("Mode not recognised")

    indices_train = indices_train.astype(int)
    indices_val = indices_val.astype(int)

    # Compute training, validation and test set statistics
    print(len(indices_train), "training samples")
    print(len(indices_val), "validation samples")
    print(len(indices_test), "test samples")
    train_class_balance = data.iloc[indices_train].label.sum(axis=0)/len(indices_train)
    print("Train set contains", np.round(train_class_balance * 100), "% Upright pianos")
    if mode == "segment_instruments":
        print("\t", pd.unique(data.iloc[indices_train].instrument))
    val_class_balance = data.iloc[indices_val].label.sum(axis=0)/len(indices_val)
    print("Validation set contains", np.round(val_class_balance * 100), "% Upright pianos")
    if mode == "segment_instruments":
        print("\t", pd.unique(data.iloc[indices_val].instrument))
    if len(indices_test) == 0:
        print("Warning: Test set is empty")
        indices_test = np.array([])
        indices_test = indices_test.astype(int)
    else:
        indices_test = indices_test.astype(int)
        test_class_balance = data.iloc[indices_test].label.sum(axis=0)/len(indices_test)
        print("Test set contains", np.round(test_class_balance * 100), "% Upright pianos")
        if mode == "segment_instruments":
            print("\t", pd.unique(data.iloc[indices_test].instrument))
    return indices_train, indices_val, indices_test


def train_model(train_set, val_set):
    print("\n--------------TRAINING MODEL--------------")
    model = TimbreCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    with torch.enable_grad():
        model.train(True)
        loss_train_log = []
        loss_val_log = []
        epoch_val_log = []
        for epoch in range(epochs):
            running_loss = 0.0
            for i, batch in enumerate(train_set):
                x = batch[0].float().to(device)
                label = batch[1].float().to(device)

                optimizer.zero_grad()
                y = model(x)
                loss = loss_function(y, label)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            # Record training loss
            print("+Training - Epoch", epoch+1, "loss:", running_loss/(batch_size*(i+1)))
            loss_train_log.append(running_loss/(batch_size*(i+1)))

            # Calculate loss on validation set
            if (epoch == 1 or epoch == 19 or epoch % val_interval == 0) and val_set is not None:
                loss_val = 0
                with torch.no_grad():
                    for i, batch in enumerate(val_set):
                        x = batch[0].float().to(device)
                        label = batch[1].float().to(device)
                        y = model(x)
                        loss_val += loss_function(y, label).item()
                print("\t+Validation - Epoch", epoch + 1, "loss:", loss_val / (batch_size * (i + 1)))
                loss_val_log.append(loss_val / (batch_size * (i + 1)))
                epoch_val_log.append(epoch+1)

    # Plot training curves
    plt.plot(range(1, epochs + 1), loss_train_log, c='r', label='train')
    plt.plot(epoch_val_log, loss_val_log, c='b', label='val')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.title("Loss curves over "+str(epochs)+" epochs of training")
    plt.show()

    return model


def evaluate_CNN(evaluated_model, test_set):
    print("\n-------------EVALUATING MODEL-------------")
    labels_acc = np.empty(0, dtype=int)
    preds_acc = np.empty(0, dtype=int)
    with torch.no_grad():
        # Inference mode
        evaluated_model.train(False)
        evaluated_model = evaluated_model.to(device)

        for batch in test_set:
            x = batch[0].float().to(device)
            label = batch[1].float().to(device)
            y = evaluated_model(x)
            print("+Evaluating - Batch loss:", loss_function(y, label).item())
            pred = torch.round(y)
            # Accumulate per-batch ground truths and outputs
            labels_acc = np.append(labels_acc, label.cpu())
            preds_acc = np.append(preds_acc, pred.cpu())

    return evaluate_scores(labels_acc, preds_acc)


def cross_validate(total_train_set):
    print("\n-----------2-FOLD CROSS-VALIDATION-----------")
    set_1, set_2, _ = generate_split_indices(total_train_set, partition_ratios=[0.5, 0.5], mode="segment_instruments")

    train_cv1 = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler.SubsetRandomSampler(set_1))
    val_cv1 = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler.SubsetRandomSampler(set_2))
    model_cv1 = train_model(train_set=train_cv1, val_set=val_cv1)
    scores_cv1 = evaluate_CNN(model_cv1, val_cv1)
    print("Fold 1 validation set scores:")
    print("Confusion matrix:\n", scores_cv1["Confusion"])
    print("Accuracy:", np.round(scores_cv1["Accuracy"], 2))
    print("F1 score:", np.round(scores_cv1["F1"], 2))

    train_cv2 = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler.SubsetRandomSampler(set_2))
    val_cv2 = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler.SubsetRandomSampler(set_1))
    model_cv2 = train_model(train_set=train_cv2, val_set=val_cv2)
    scores_cv2 = evaluate_CNN(model_cv2, val_cv2)
    print("Fold 2 validation set scores:")
    print("Confusion matrix:\n", scores_cv2["Confusion"])
    print("Accuracy:", np.round(scores_cv2["Accuracy"], 2))
    print("F1 score:", np.round(scores_cv2["F1"], 2))

    scores_cv = {"Confusion": (scores_cv1["Confusion"] + scores_cv2["Confusion"])/2,
                 "Accuracy": (scores_cv1["Accuracy"] + scores_cv2["Accuracy"])/2,
                 "F1": (scores_cv1["F1"] + scores_cv2["F1"])/2}
    print("\n-------Overall cross-validation scores-------")
    print("Confusion matrix:\n", scores_cv["Confusion"])
    print("Accuracy:", np.round(scores_cv["Accuracy"], 2))
    print("F1 score:", np.round(scores_cv["F1"], 2))


if __name__ == '__main__':
    # Configure CPU or GPU using CUDA if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\n----------------LOADING DATA----------------")
    loader = InstrumentLoader(data_dir, note_range=[48, 72], set_velocity="M")

    data = loader.preprocess(fmin=20, fmax=20000, n_mels=300, normalisation="statistics")

    dataset = TimbreDataset(data)
    train_indices, val_indices, test_indices = generate_split_indices(data, partition_ratios=[0.8, 0.1],
                                                                      mode="segment_instruments")
    loader_train = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                              sampler=sampler.SubsetRandomSampler(train_indices))
    loader_val = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            sampler=sampler.SubsetRandomSampler(val_indices))
    loader_test = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             sampler=sampler.SubsetRandomSampler(test_indices))

    cross_validate(data.iloc[np.union1d(train_indices, val_indices)])

    if not os.path.isfile(saved_model_path):
        print("\nCreating and training new model")
        model = train_model(train_set=loader_train, val_set=loader_val)
        # Save model
        torch.save(model, saved_model_path)
        print("Saved trained model to", saved_model_path)
    else:
        print("\nLoading model from", saved_model_path)
        model = torch.load(saved_model_path)
    print(model)
    model.count_parameters()

    scores = evaluate_CNN(model, loader_test)
    print("Confusion matrix:\n", scores["Confusion"])
    print("Accuracy:", np.round(scores["Accuracy"], 2))
    print("F1 score:", np.round(scores["F1"], 2))
