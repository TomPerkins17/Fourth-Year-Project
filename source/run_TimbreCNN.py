# %%
import warnings
import sklearn
import pandas as pd

from data_loading import *
from timbre_CNN import *
from evaluation import *
from torch.utils.data import DataLoader, sampler
from melody_loading import *

model_dir = "models"
model_name = "diff-melodies_MIDIsampledseen"
val_interval = 5
perform_cross_val = False

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
    if mode == "segment-instruments-random":
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
            indices_test = np.append(indices_test, np.asarray(data.instrument == instruments[j]).nonzero()[0])
        np.random.shuffle(indices_train)
        np.random.shuffle(indices_val)
        np.random.shuffle(indices_test)

    elif mode == "segment-instruments-random-balanced":
        instruments_grand = data[data.label == 0].instrument.unique()
        instruments_upright = data[data.label == 1].instrument.unique()
        rng.shuffle(instruments_grand)
        rng.shuffle(instruments_upright)
        num_train_instruments = np.round(partition_ratios[0] * len(data.instrument.unique()))
        num_val_instruments = np.round(partition_ratios[1] * len(data.instrument.unique()))
        indices_train = []
        indices_val = []
        indices_test = []
        i_grand = 0
        i_upright = 0

        for i in range(0, len(data.instrument.unique())):
            if i % 2 and i_upright < len(instruments_upright):
                next_instrument_indices = np.asarray(data.instrument == instruments_upright[i_upright]).nonzero()[0]
                i_upright += 1
            elif i_grand < len(instruments_grand):
                next_instrument_indices = np.asarray(data.instrument == instruments_grand[i_grand]).nonzero()[0]
                i_grand += 1
            else:
                break
            if i < num_train_instruments:
                indices_train = np.append(indices_train, next_instrument_indices)
            elif i < num_train_instruments+num_val_instruments:
                indices_val = np.append(indices_val, next_instrument_indices)
            else:
                indices_test = np.append(indices_test, next_instrument_indices)

        np.random.shuffle(indices_train)
        np.random.shuffle(indices_val)
        np.random.shuffle(indices_test)

    elif mode == "segment-instruments-manual":
        # train_instruments = ["AkPnBcht", "AkPnBsdf", "grand-closed", "grand-removed", "grand-open",
        #                      "upright-open", "upright-semiopen", "upright-closed"]
        # val_instruments = ["StbgTGd2", "AkPnCGdD", "ENSTDkCl"]
        # test_instruments = ["AkPnStgb", "SptkBGAm", "ENSTDkAm"]
        train_instruments = ["Nord_BrightGrand-XL",         "Nord_AmberUpright-XL",
                             "Nord_ConcertGrand1Amb-Lrg",   "Nord_BabyUpright-XL",
                             "Nord_GrandImperial-XL",       "Nord_BlackUpright-Lrg",
                             "Nord_GrandLadyD-Lrg",         "Nord_BlueSwede-Lrg",
                             "Nord_RoyalGrand3D-XL",        "Nord_MellowUpright-XL",
                             "Nord_SilverGrand-XL",         "Nord_QueenUpright-Lrg",
                             "Nord_StudioGrand1-Lrg",       "Nord_RainPiano-Lrg"]
        val_instruments =   ["Nord_ItalianGrand-XL",        "Nord_GrandUpright-XL",
                             "Nord_StudioGrand2-Lrg"]
        test_instruments =  ["Nord_VelvetGrand-XL",         "Nord_RomanticUpright-Lrg",
                             "Nord_WhiteGrand-XL",          "Nord_SaloonUpright-Lrg",
                             "Nord_ConcertGrand1-Lrg",      "Nord_BambinoUpright-XL"]

        indices_train = np.asarray(data.instrument.isin(train_instruments)).nonzero()[0]
        indices_val = np.asarray(data.instrument.isin(val_instruments)).nonzero()[0]
        indices_test = np.asarray(data.instrument.isin(test_instruments)).nonzero()[0]
        np.random.shuffle(indices_train)
        np.random.shuffle(indices_val)
        np.random.shuffle(indices_test)

    elif mode == "segment-velocities":
        indices_train = np.asarray(data.velocity == "M").nonzero()[0]
        indices_val = np.asarray(data.velocity == "P").nonzero()[0]
        indices_test = np.asarray(data.velocity == "F").nonzero()[0]
        np.random.shuffle(indices_train)
        np.random.shuffle(indices_val)
        np.random.shuffle(indices_test)
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

    # Print training, validation and test set statistics
    indices_train = indices_train.astype(int)
    indices_val = indices_val.astype(int)
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
    print("Overall, dataset contains", np.round(100 * data.label.sum(axis=0)/len(data)), "% Upright pianos")
    return indices_train, indices_val, indices_test


def train_model(cnn_type, train_set, val_set, plot_title=""):
    print("\n--------------TRAINING MODEL--------------")
    model = cnn_type().to(device)
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
    plt.title("Loss curves over "+str(epochs)+" epochs of training - "+plot_title)
    plt.show()

    return model


def evaluate_CNN(evaluated_model, test_set):
    labels_total = np.empty(0, dtype=int)
    preds_total = np.empty(0, dtype=int)
    instruments_acc = np.empty(0, dtype=str)
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
            # Accumulate per-batch ground truths, outputs and instrument names
            labels_total = np.append(labels_total, label.cpu())
            preds_total = np.append(preds_total, pred.cpu())
            instruments_acc = np.append(instruments_acc, np.array(batch[2]))
    # Calculate scores per instrument
    per_inst_scores = pd.DataFrame()
    for instrument in np.unique(instruments_acc):
        instrument_mask = np.nonzero(instruments_acc == instrument)
        # Ignore Confusion matrix and F1 score which are irrelevant here
        instrument_scores = evaluate_scores(labels_total[instrument_mask], preds_total[instrument_mask])
        piano_class = "Upright" if labels_total[instrument_mask][0] else "Grand"
        per_inst_scores = per_inst_scores.append(pd.DataFrame([[np.round(instrument_scores["Accuracy"],2),piano_class]],
                                                              index=pd.Index([instrument], name="Instrument"),
                                                              columns=["Accuracy", "Class"]))
    print("---------Per-instrument scores---------")
    print(per_inst_scores)
    # Calculate overall scores
    overall_scores = evaluate_scores(labels_total, preds_total)
    return overall_scores


def cross_validate(cnn_type, total_seen_dataset, partition_mode):
    set_1, set_2, _ = generate_split_indices(total_seen_dataset.dataframe, partition_ratios=[0.5, 0.5], mode=partition_mode)

    # Perform fold 1 training and evaluation
    train_cv1 = DataLoader(total_seen_dataset, batch_size=batch_size, shuffle=False, sampler=sampler.SubsetRandomSampler(set_1))
    val_cv1 = DataLoader(total_seen_dataset, batch_size=batch_size, shuffle=False, sampler=sampler.SubsetRandomSampler(set_2))
    model_cv1 = train_model(cnn_type=cnn_type, train_set=train_cv1, val_set=val_cv1, plot_title="CV Fold 1")
    scores_cv1 = evaluate_CNN(model_cv1, val_cv1)
    print("\n------Fold 1 validation set scores--------")
    print("Confusion matrix:\n", scores_cv1["Confusion"])
    print("Accuracy:", np.round(scores_cv1["Accuracy"], 2))
    print("F1 score:", np.round(scores_cv1["F1"], 2))

    # Perform fold 2 training and evaluation
    train_cv2 = DataLoader(total_seen_dataset, batch_size=batch_size, shuffle=False, sampler=sampler.SubsetRandomSampler(set_2))
    val_cv2 = DataLoader(total_seen_dataset, batch_size=batch_size, shuffle=False, sampler=sampler.SubsetRandomSampler(set_1))
    model_cv2 = train_model(cnn_type=cnn_type, train_set=train_cv2, val_set=val_cv2, plot_title="CV Fold 2")
    scores_cv2 = evaluate_CNN(model_cv2, val_cv2)
    print("\n------Fold 2 validation set scores--------")
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

    print("\n\n----------------------LOADING DATA-----------------------")
    # timbre_CNN_type = SingleNoteTimbreCNN
    # loader = InstrumentLoader(data_dir, note_range=[48, 72], set_velocity=None, normalise_wavs=True, load_MIDIsampled=True)
    # total_data = loader.preprocess(fmin=20, fmax=20000, n_mels=300, normalisation="statistics")
    timbre_CNN_type = MelodyTimbreCNN
    loader = MelodyInstrumentLoader(data_dir, note_range=[48, 72], set_velocity=None, normalise_wavs=True, load_MIDIsampled=True)
    total_data = loader.preprocess_melodies(midi_dir, normalisation="statistics")

    # Split into seen and unseen subsets
    data_seen = total_data[total_data.dataset == "MIDIsampled"]
    data_unseen = total_data[total_data.dataset != "MIDIsampled"]

    dataset_seen = TimbreDataset(data_seen)

    partition_mode = "segment-instruments-random-balanced"
    train_indices, val_indices, _ = generate_split_indices(data_seen, partition_ratios=[0.8, 0.2],
                                                                      mode=partition_mode)
    loader_train = DataLoader(dataset_seen, batch_size=batch_size, shuffle=False,
                              sampler=sampler.SubsetRandomSampler(train_indices))
    loader_val = DataLoader(dataset_seen, batch_size=batch_size, shuffle=False,
                            sampler=sampler.SubsetRandomSampler(val_indices))
    # loader_test = DataLoader(dataset_seen, batch_size=batch_size, shuffle=False,
    #                          sampler=sampler.SubsetRandomSampler(test_indices))

    if perform_cross_val:
        print("\n\n-----------------2-FOLD CROSS-VALIDATION-----------------")
        cross_validate(cnn_type=timbre_CNN_type,
                       total_seen_dataset=dataset_seen,
                       partition_mode=partition_mode)

    print("\n\n-------------------RE-TRAINED MODEL-----------------------")
    model_filename = "model_"+partition_mode+"_"+model_name+".pth"
    saved_model_path = os.path.join(model_dir, timbre_CNN_type.__name__, model_filename)
    if not os.path.isfile(saved_model_path):
        print("\nCreating and training new model")
        model = train_model(cnn_type=timbre_CNN_type, train_set=loader_train, val_set=loader_val,
                            plot_title="Re-trained model: "+model_filename)
        # Save model
        torch.save(model, saved_model_path)
        print("Saved trained model to", saved_model_path)
    else:
        print("\nLoading pre-trained model from", saved_model_path)
        model = torch.load(saved_model_path)
    print(model)
    model.count_parameters()

    print("\n\n-------------Evaluation on the validation set-------------")
    scores_seen = evaluate_CNN(model, loader_val)
    print("---Overall validation set performance---")
    display_scores(scores_seen, "Validation set")

    print("\n\n--------------Evaluation on the unseen set---------------")
    dataset_unseen = TimbreDataset(data_unseen)
    loader_unseen = DataLoader(dataset_unseen, batch_size=batch_size, shuffle=False)
    scores_unseen = evaluate_CNN(model, loader_unseen)
    print("--------Overall unseen set performance--------")
    display_scores(scores_unseen, "Unseen test set")

