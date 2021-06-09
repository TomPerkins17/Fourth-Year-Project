# %%
import warnings
import sklearn
import pandas as pd
import gc

from data_loading import *
from timbre_CNN import *
from evaluation import *
from torch.utils.data import DataLoader, sampler
from melody_loading import *

result_dir = "results"
model_dir = "models"
model_name = "MIDIsampledseen_evalmode"
val_interval = 5
perform_cross_val = True
evaluation_bs = 512

# Hyperparameters
hyperparams_single = {"batch_size": 256,  # GPU memory limits us to <512
                      "epochs": 25,
                      "learning_rate": 0.002,
                      "loss_function": nn.BCELoss()}

hyperparams_melody = {"batch_size": 256,  # GPU memory limits us to <512
                      "epochs": 25,
                      "learning_rate": 0.002,
                      "loss_function": nn.BCELoss()}


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
        if np.sum(partition_ratios) == 1:   # Combine val and test sets if no test set required
            indices_val = np.append(indices_val, indices_test)
            indices_test = []
        np.random.shuffle(indices_train)
        np.random.shuffle(indices_val)
        np.random.shuffle(indices_test)

    elif mode == "segment-instruments-manual":
        # train_instruments = ["AkPnBcht", "AkPnBsdf", "grand-closed", "grand-removed", "grand-open",
        #                      "upright-open", "upright-semiopen", "upright-closed"]
        # val_instruments = ["StbgTGd2", "AkPnCGdD", "ENSTDkCl"]
        # test_instruments = ["AkPnStgb", "SptkBGAm", "ENSTDkAm"]
        # train_instruments = ["Nord_BrightGrand-XL",         "Nord_AmberUpright-XL",
        #                      "Nord_ConcertGrand1Amb-Lrg",   "Nord_BabyUpright-XL",
        #                      "Nord_GrandImperial-XL",       "Nord_BlackUpright-Lrg",
        #                      "Nord_GrandLadyD-Lrg",         "Nord_BlueSwede-Lrg",
        #                      "Nord_RoyalGrand3D-XL",        "Nord_MellowUpright-XL",
        #                      "Nord_SilverGrand-XL",         "Nord_QueenUpright-Lrg",
        #                      "Nord_StudioGrand1-Lrg",       "Nord_RainPiano-Lrg"]
        # val_instruments =   ["Nord_ItalianGrand-XL",        "Nord_GrandUpright-XL",
        #                      "Nord_StudioGrand2-Lrg"]
        # test_instruments =  ["Nord_VelvetGrand-XL",         "Nord_RomanticUpright-Lrg",
        #                      "Nord_WhiteGrand-XL",          "Nord_SaloonUpright-Lrg",
        #                      "Nord_ConcertGrand1-Lrg",      "Nord_BambinoUpright-XL"]
        train_instruments = ["Nord_BrightGrand-XL",         "Nord_AmberUpright-XL",
                             "Nord_ConcertGrand1-Lrg",      "Nord_BabyUpright-XL",
                             "Nord_GrandImperial-XL",       "Nord_BlackUpright-Lrg",
                             "Nord_RoyalGrand3D-XL",        "Nord_MellowUpright-XL",
                             "Nord_StudioGrand1-Lrg",       "Nord_RainPiano-Lrg",
                             "Nord_WhiteGrand-XL",          "Nord_RomanticUpright-Lrg",
                             "Nord_VelvetGrand-XL",         "Nord_GrandUpright-XL",
                             "Nord_StudioGrand2-Lrg",       "Nord_SaloonUpright-Lrg",
                             "Nord_ItalianGrand-XL",        "Nord_BlueSwede-Lrg"]
        val_instruments =   ["Nord_ConcertGrand1Amb-Lrg",   "Nord_BambinoUpright-XL",
                             "Nord_GrandLadyD-Lrg",         "Nord_QueenUpright-Lrg",
                             "Nord_SilverGrand-XL"]
        test_instruments = []

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


def generate_crossval_fold_indices(data, seed=None):
    rng = np.random.default_rng(seed=seed)
    instruments_grand = data[data.label == 0].instrument.unique()
    instruments_upright = data[data.label == 1].instrument.unique()
    rng.shuffle(instruments_grand)
    rng.shuffle(instruments_upright)
    num_instruments_fold1 = np.round(len(data.instrument.unique())/5)
    num_instruments_fold2 = np.round(len(data.instrument.unique())/5)
    num_instruments_fold3 = np.round(len(data.instrument.unique())/5)
    num_instruments_fold4 = np.round(len(data.instrument.unique())/5)
    indices_fold1 = []
    indices_fold2 = []
    indices_fold3 = []
    indices_fold4 = []
    indices_fold5 = []
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
        if i < num_instruments_fold1:
            indices_fold1 = np.append(indices_fold1, next_instrument_indices)
        elif i < num_instruments_fold1 + num_instruments_fold2:
            indices_fold2 = np.append(indices_fold2, next_instrument_indices)
        elif i < num_instruments_fold1 + num_instruments_fold2 + num_instruments_fold3:
            indices_fold3 = np.append(indices_fold3, next_instrument_indices)
        elif i < num_instruments_fold1 + num_instruments_fold2 + num_instruments_fold3 + num_instruments_fold4:
            indices_fold4 = np.append(indices_fold4, next_instrument_indices)
        else:
            indices_fold5 = np.append(indices_fold5, next_instrument_indices)
    np.random.shuffle(indices_fold1)
    np.random.shuffle(indices_fold2)
    np.random.shuffle(indices_fold3)
    np.random.shuffle(indices_fold4)
    np.random.shuffle(indices_fold5)

    print(len(indices_fold1), "samples in fold 1")
    print("\t", pd.unique(data.iloc[indices_fold1].instrument))
    print(len(indices_fold2), "samples in fold 2")
    print("\t", pd.unique(data.iloc[indices_fold2].instrument))
    print(len(indices_fold3), "samples in fold 3")
    print("\t", pd.unique(data.iloc[indices_fold3].instrument))
    print(len(indices_fold4), "samples in fold 4")
    print("\t", pd.unique(data.iloc[indices_fold4].instrument))
    print(len(indices_fold5), "samples in fold 5")
    print("\t", pd.unique(data.iloc[indices_fold5].instrument))

    return indices_fold1.astype(int), indices_fold2.astype(int), indices_fold3.astype(int), indices_fold4.astype(int), indices_fold5.astype(int)


def train_model(cnn_type, train_set, val_set, plot_title=""):
    print("\n--------------TRAINING MODEL--------------")
    model = cnn_type().to(device, non_blocking=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    with torch.enable_grad():
        loss_train_log = []
        loss_val_log = []
        epoch_val_log = []
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for i, batch in enumerate(train_set):
                x = batch[0].float().to(device, non_blocking=True)
                label = batch[1].float().to(device, non_blocking=True)

                optimizer.zero_grad()
                y = model(x)
                loss = loss_function(y, label)

                loss.backward()
                optimizer.step()
                running_loss += loss.detach()
                gc.collect()
            # Record training loss
            mean_epoch_loss = (running_loss/(batch_size*(i+1))).item()
            print("+Training - Epoch", epoch+1, "loss:", mean_epoch_loss)
            loss_train_log.append(mean_epoch_loss)

            # Calculate loss on validation set
            if (epoch == epochs-1 or epoch % val_interval == 0) and val_set is not None:
                loss_val = 0
                model.eval()
                with torch.no_grad():
                    for i, batch in enumerate(val_set):
                        x = batch[0].float().to(device, non_blocking=True)
                        label = batch[1].float().to(device, non_blocking=True)
                        y = model(x)
                        loss_val += loss_function(y, label).detach()
                        gc.collect()
                mean_epoch_val_loss = (loss_val / (batch_size * (i + 1))).item()
                print("\t+Validation - Epoch", epoch + 1, "loss:", mean_epoch_val_loss)
                loss_val_log.append(mean_epoch_val_loss)
                epoch_val_log.append(epoch+1)

    # Plot training curves
    fig = plt.figure()
    plt.plot(range(1, epochs + 1), loss_train_log, c='r', label='train')
    plt.plot(epoch_val_log, loss_val_log, c='b', label='val')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.title("Loss curves over "+str(epochs)+" epochs of training - "+plot_title)
    plt.show()

    return model, fig


def evaluate_CNN(evaluated_model, test_set):
    labels_total = np.empty(0, dtype=int)
    preds_total = np.empty(0, dtype=int)
    instruments_acc = np.empty(0, dtype=str)
    # Inference mode
    evaluated_model.eval()
    with torch.no_grad():
        evaluated_model = evaluated_model.to(device, non_blocking=True)

        for batch in test_set:
            x = batch[0].float().to(device, non_blocking=True)
            label = batch[1].float().to(device, non_blocking=True)
            y = evaluated_model(x)
            #print("+Evaluating - Batch loss:", loss_function(y, label).item())
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
    # Calculate overall scores
    overall_scores = evaluate_scores(labels_total, preds_total)
    return overall_scores, per_inst_scores


def cross_validate(cnn_type, cross_val_subset, cv_mode="2fold", partition_mode=None):

    cv_dataset = TimbreDataset(cross_val_subset)
    total_scores = pd.DataFrame()

    if cv_mode == "2fold":
        set_1, set_2, _ = generate_split_indices(cross_val_subset, partition_ratios=[0.5, 0.5], mode=partition_mode)
        training_sets = [set_1, set_2]
        validation_sets = [set_2, set_1]
    elif cv_mode == "5fold":
        fold1, fold2, fold3, fold4, fold5 = generate_crossval_fold_indices(cross_val_subset, seed=None)
        training_sets = [np.concatenate([fold2, fold3, fold4, fold5]),
                         np.concatenate([fold3, fold4, fold5, fold1]),
                         np.concatenate([fold4, fold5, fold1, fold2]),
                         np.concatenate([fold5, fold1, fold2, fold3]),
                         np.concatenate([fold1, fold2, fold3, fold4])]
        validation_sets = [fold1, fold2, fold3, fold4, fold5]
    else:
        raise Exception("CV mode "+cv_mode+" not implemented")

    for fold, (train_fold_indices, val_fold_indices) in enumerate(zip(training_sets, validation_sets)):
        train_fold = DataLoader(cv_dataset, batch_size=batch_size, shuffle=False,
                                sampler=sampler.SubsetRandomSampler(train_fold_indices), pin_memory=True)
        val_fold = DataLoader(cv_dataset, batch_size=evaluation_bs, shuffle=False,
                              sampler=sampler.SubsetRandomSampler(val_fold_indices), pin_memory=True)
        model_fold, _ = train_model(cnn_type=cnn_type, train_set=train_fold, val_set=val_fold,
                                    plot_title="CV Fold "+str(fold+1))
        scores_fold, per_inst_scores_fold = evaluate_CNN(model_fold, val_fold)
        print("\n------Fold "+str(fold+1)+" validation set scores--------")
        print(per_inst_scores_fold)
        print("Confusion matrix:\n", scores_fold["Confusion"])
        print("Accuracy:", np.round(scores_fold["Accuracy"], 2))
        print("F1 score:", np.round(scores_fold["F1"], 2))
        numeric_scores_fold = pd.DataFrame.from_dict({k: [v] for k, v in scores_fold.items() if k in ["Accuracy", "F1"]})
        numeric_scores_fold["no_samples"] = len(val_fold_indices)
        total_scores = total_scores.append(numeric_scores_fold)
    print("\n-------Overall cross-validation scores-------")

    mean_scores = {"Accuracy": (total_scores.Accuracy * total_scores.no_samples).sum() / total_scores.no_samples.sum(),
                   "F1": (total_scores.F1 * total_scores.no_samples).sum() / total_scores.no_samples.sum()}
    print(mean_scores)


if __name__ == '__main__':
    # Configure CPU or GPU using CUDA if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\n\n----------------------LOADING DATA-----------------------")
    timbre_CNN_type = SingleNoteTimbreCNN
    #timbre_CNN_type = MelodyTimbreCNN
    if timbre_CNN_type == SingleNoteTimbreCNN:
        hyperparams = hyperparams_single
        loader = InstrumentLoader(data_dir, note_range=[48, 72], set_velocity=None, normalise_wavs=True, load_MIDIsampled=True)
        total_data = loader.preprocess(fmin=20, fmax=20000, n_mels=300, normalisation="statistics")
    elif timbre_CNN_type == MelodyTimbreCNN:
        hyperparams = hyperparams_melody
        loader = MelodyInstrumentLoader(data_dir, note_range=[48, 72], set_velocity=None, normalise_wavs=True, load_MIDIsampled=True)
        total_data = loader.preprocess_melodies(midi_dir, normalisation="statistics")
    else:
        raise Exception(str(timbre_CNN_type)+" doesn't exist")

    batch_size = hyperparams["batch_size"]
    epochs = hyperparams["epochs"]
    learning_rate = hyperparams["learning_rate"]
    loss_function = hyperparams["loss_function"]

    # Split into seen and unseen subsets
    data_seen = total_data[total_data.dataset == "MIDIsampled"]
    data_unseen = total_data[total_data.dataset != "MIDIsampled"]
    gc.collect()

    dataset_seen = TimbreDataset(data_seen)

    train_indices, val_indices, _ = generate_split_indices(data_seen, partition_ratios=[0.8, 0.2],
                                                                      mode="segment-instruments-manual")
    loader_train = DataLoader(dataset_seen, batch_size=batch_size, shuffle=False,
                              sampler=sampler.SubsetRandomSampler(train_indices),
                              pin_memory=True)
    loader_val = DataLoader(dataset_seen, batch_size=evaluation_bs, shuffle=False,
                            sampler=sampler.SubsetRandomSampler(val_indices),
                            pin_memory=True)

    if perform_cross_val:
        print("\n\n---------------------CROSS-VALIDATION---------------------")
        cross_validate(cnn_type=timbre_CNN_type,
                       cross_val_subset=data_seen, #data_seen.iloc[train_indices],
                       cv_mode="5fold",
                       partition_mode="segment-instruments-random-balanced")

    print("\n\n-------------------RE-TRAINED MODEL-----------------------")
    model_filename = "model_"+str(batch_size)+"_"+str(epochs)+"_"+str(learning_rate)+"_"+model_name
    saved_model_path = os.path.join(model_dir, timbre_CNN_type.__name__, model_filename+".pth")
    if not os.path.isfile(saved_model_path):
        print("\nCreating and training new model")
        model, loss_plot = train_model(cnn_type=timbre_CNN_type, train_set=loader_train, val_set=loader_val,
                            plot_title="Re-trained model:\n"+model_filename)
        # Save model
        torch.save(model, saved_model_path)
        print("Saved trained model to", saved_model_path)
        # Save loss plot
        loss_plot.savefig(os.path.join(model_dir, timbre_CNN_type.__name__, model_filename+".svg"))
    else:
        print("\nLoading pre-trained model from", saved_model_path)
        model = torch.load(saved_model_path)
    print(model)
    model.count_parameters()

    print("\n\n-------------Evaluation on the validation set-------------")
    scores_seen, per_inst_scores_seen = evaluate_CNN(model, loader_val)
    print("---------Per-instrument scores---------")
    print(per_inst_scores_seen)
    per_inst_scores_seen.to_csv(os.path.join(result_dir, timbre_CNN_type.__name__, model_filename + ".csv"))
    print("---Overall validation set performance---")
    display_scores(scores_seen, "Validation set")

    print("\n\n--------------Evaluation on the unseen set---------------")
    dataset_unseen = TimbreDataset(data_unseen)
    loader_unseen = DataLoader(dataset_unseen, batch_size=evaluation_bs, shuffle=False, pin_memory=True)
    scores_unseen, per_inst_scores_unseen = evaluate_CNN(model, loader_unseen)
    print("---------Per-instrument scores---------")
    print(per_inst_scores_unseen)
    per_inst_scores_unseen.to_csv(os.path.join(result_dir, timbre_CNN_type.__name__, model_filename + ".csv"), mode="a")
    print("--------Overall unseen set performance--------")
    display_scores(scores_unseen, "Unseen test set")

