import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler






def data_prep(data): 
    data.sort_values(['Patient_ID','Eye_ID', 'Visit_Number'])
    patients = pd.unique(data['Patient_ID'])
    patient_sequence = []
    eye_sequence = []
    metadf = pd.DataFrame(columns = data.columns[:-1])
    row_count = 0
    
    for patient, id_number in enumerate(patients):
        patient_level_data = data[data['Patient_ID'] == id_number]
        
        visits = pd.unique(patient_level_data['Visit_Number'])
        
        visits_sequence = []
        for visit in visits:
            visit_level_data = patient_level_data[patient_level_data['Visit_Number'] == visit]
            eyes = pd.unique(visit_level_data['Eye_ID'])
            
            for eye in eyes:

            
                eye_level_data = visit_level_data[visit_level_data['Eye_ID' ]== eye]

                #print(visit_level_data)
                #print(visit_level_data['Image'])
            
                image_array =  np.stack( eye_level_data['Image'].to_numpy() / 255,axis = 0 ) #normalize color
                if image_array.shape[0] != 46:
                    print(image_array.shape)
                    print(id_number,eye,visit)
                else:
                    eye_sequence += [image_array]
                    column_count =  len(eye_level_data.columns)
                    #print(column_count)
                    metadata = eye_level_data.iloc[0,0:(column_count-1)]
                    print(metadata)
                    #print(len(metadata.columns) == len(metadf.columns))
                    #print(metadf.columns)
                    #print(metadata.columns)
                    metadf.loc[row_count] = metadata
                    row_count += 1 

           # visits_sequence += [image_array]
        #patient_sequence += [ np.stack(eye_sequence, axis= 0)]
    volumes = len(eye_sequence)    
    split_size = int(volumes/4)
    print(len(metadf))
    metadf.to_csv('volumeLevels.csv')
    for split in range(4):
        print([split, split_size])
        #np.save(f'dataset_{split}.npy', np.stack(eye_sequence[split * split_size:(split +1) *split_size],axis = 0))
        
    return "finished"

class OCTDataset(Dataset):

    def __init__(self, file_paths, metadata):

        self.arrays = [
            np.load(path, mmap_mode="r")
            for path in file_paths
        ]


        # Cumulative volume counts
        self.offsets = np.cumsum(
            [0] + [len(arr) for arr in self.arrays]
        )
        self.metadata = metadata.reset_index(drop=True)
        assert len(self.metadata) == self.offsets[-1]

        

        #definitions
        future_cst = self.metadata["CST_t_plus_1"]

        self.metadata["Meets_TREX_Criteria"] = (
            (future_cst >= 325) |
            (future_cst >= 1.2 * self.metadata["nadir_CST"])
        ).astype(np.float32)

        # Do not classify missing future outcomes as negative
        self.metadata.loc[
            future_cst.isna() | self.metadata["nadir_CST"].isna(),
            "Meets_TREX_Criteria"
        ] = np.nan





        self.metadata["First_Visit"] = (
            self.metadata["delta_CST"].isna()
        ).astype(np.float32)

        self.metadata["delta_CST"] = (
            self.metadata["delta_CST"].fillna(0)
        )

        self.metadata["pct_change_CST"] = (
            self.metadata["pct_change_CST"].fillna(0)
        )
        self.clinical_predictors = [ 'CST','BCVA','nadir_CST','delta_CST','pct_change_CST','pct_above_nadir']

            #add biomarkers when i figure out wtf to do
    def __len__(self):
        return self.offsets[-1]



    def __getitem__(self, idx):

            # Identify which file contains the volume
            file_idx = np.searchsorted(
                self.offsets[1:], idx, side="right"
            )

            # Find index within that file
            local_idx = idx - self.offsets[file_idx]

            # Load only the requested volume
            volume = np.array(
                self.arrays[file_idx][local_idx],
                dtype=np.float32,
                copy=True
            )

            # Add channel dimension
            X_images = torch.from_numpy(volume).unsqueeze(0)

            X_clinical = torch.tensor(
                self.metadata.iloc[idx][self.clinical_predictors]
                .to_numpy(dtype=np.float32)
            )

            # Continuous target: CST at t+1
            y_reg = torch.tensor(
                [self.metadata.iloc[idx]["CST_t_plus_1"]],
                dtype=torch.float32
            )

            # Binary target: meets TREX criteria at t+1
            y_class = torch.tensor(
                [self.metadata.iloc[idx]["Meets_TREX_Criteria"]],
                dtype=torch.float32
            )

            return X_images, X_clinical, y_reg, y_class






def get_cst_predictions(model, loader, device):
    model.eval()

    actual = []
    predicted = []

    with torch.no_grad():
        for images, clinical, y_reg, y_class in loader:
            images = images.to(device)
            clinical = clinical.to(device)

            cst_pred, _ = model(images, clinical)

            actual.extend(y_reg.cpu().numpy().flatten())
            predicted.extend(cst_pred.cpu().numpy().flatten())

    return np.array(actual), np.array(predicted)





def plot_cst_results(actual, predicted):

    residuals = actual - predicted

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Predicted vs actual
    axes[0].scatter(actual, predicted, alpha=0.5)

    limits = [
        min(actual.min(), predicted.min()),
        max(actual.max(), predicted.max())
    ]

    axes[0].plot(limits, limits, "k--")
    axes[0].set_xlabel("Actual CST (µm)")
    axes[0].set_ylabel("Predicted CST (µm)")
    axes[0].set_title("Predicted vs Actual CST")

    # Residuals
    axes[1].scatter(predicted, residuals, alpha=0.5)
    axes[1].axhline(0, linestyle="--", color="black")
    axes[1].set_xlabel("Predicted CST (µm)")
    axes[1].set_ylabel("Residual (µm)")
    axes[1].set_title("Residual Plot")

    # Error distribution
    axes[2].hist(residuals, bins=25, edgecolor="black")
    axes[2].set_xlabel("Prediction Error (µm)")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Error Distribution")

    plt.tight_layout()
    plt.show()

    print("MAE:", mean_absolute_error(actual, predicted))
    print("R²:", r2_score(actual, predicted))








def get_trex_predictions(model, loader, device):
    model.eval()

    actual = []
    probabilities = []

    with torch.no_grad():
        for images, clinical, y_reg, y_class in loader:
            images = images.to(device)
            clinical = clinical.to(device)

            _, class_logits = model(images, clinical)

            probs = torch.sigmoid(class_logits)

            actual.extend(y_class.cpu().numpy().flatten())
            probabilities.extend(probs.cpu().numpy().flatten())

    return np.array(actual), np.array(probabilities)


from sklearn.metrics import (
roc_curve,
auc,
ConfusionMatrixDisplay,
confusion_matrix
)
from sklearn.calibration import calibration_curve

def plot_trex_results(actual, probabilities):

    predicted = (probabilities >= 0.5).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # ROC curve
    if len(np.unique(actual)) == 2:
        fpr, tpr, _ = roc_curve(actual, probabilities)
        roc_auc = auc(fpr, tpr)

        axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        axes[0].plot([0, 1], [0, 1], "k--")
        axes[0].legend()
    else:
        axes[0].text(
            0.5, 0.5, "ROC requires both classes",
            ha="center", va="center"
        )

    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")

    # Confusion matrix
    cm = confusion_matrix(actual, predicted, labels=[0, 1])

    ConfusionMatrixDisplay(
        cm, display_labels=["Negative", "Positive"]
    ).plot(ax=axes[1], colorbar=False)

    axes[1].set_title("Confusion Matrix")

    # Calibration curve
    prob_true, prob_pred = calibration_curve(
        actual, probabilities, n_bins=5,
        strategy="quantile"
    )

    axes[2].plot(prob_pred, prob_true, marker="o")
    axes[2].plot([0, 1], [0, 1], "k--")
    axes[2].set_xlabel("Mean Predicted Probability")
    axes[2].set_ylabel("Observed Positive Frequency")
    axes[2].set_title("Probability Calibration")

    plt.tight_layout()
    plt.show()





class OCT_CNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten()
        )

        # Regression output: future CST
        self.regression = nn.Linear(38, 1)
        self.classification = nn.Linear(38, 1)

    def forward(self, images, clinical):

        image_features = self.features(images)

        combined_features = torch.cat(
            [image_features, clinical],
            dim=1
        )

        cst = self.regression(combined_features)
        classification = self.classification(combined_features)

        return cst, classification


class OCT_CNN_CLASSIFIER(nn.Module):

    def __init__(self):
        super().__init__()

        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten()
        )

         #Regression output: future CST
        self.regression = nn.Linear(38, 1)
        self.classification = nn.Linear(38, 1)

    def forward(self, images, clinical):

        image_features = self.features(images)

        combined_features = torch.cat(
            [image_features, clinical],
            dim=1
        )

        cst = self.regression(combined_features)
        classification = self.classification(combined_features)

        return cst, classification

os.chdir(r'F:\Data\Edemea Models Personal')
#train_pos = np.load('train_indices.npy')
test_pos = np.load('test_indices.npy')



pickle = np.load( 'dataset.pk1', allow_pickle= True)



data_files = ['dataset_0.npy','dataset_1.npy','dataset_2.npy','dataset_3.npy']
metadata = pd.read_csv('volumeLevels.csv')

dataset = OCTDataset(data_files,metadata)

target_cols = [
    "CST_t_plus_1",
    "Meets_TREX_Criteria"
]

valid_mask = dataset.metadata[target_cols].notna().all(axis=1)
valid_indices = np.flatnonzero(valid_mask.to_numpy())

filtered_dataset = Subset(dataset, valid_indices)


test_dataset = Subset(filtered_dataset, test_pos)




device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

cst_model = OCT_CNN().to(device)
dict_state = torch.load('weights.torch')
cst_model.load_state_dict(dict_state)

cst_test_loader = DataLoader(test_dataset,2,True)





actual_cst, predicted_cst = get_cst_predictions(
    cst_model,
    cst_test_loader,
    device
)






continuous_cols = [
    "CST",
    "BCVA",
    "nadir_CST",
    "delta_CST",
    "pct_change_CST",
    'pct_above_nadir'
]

scaler = StandardScaler()

scaler.fit(
    test_dataset.metadata.iloc[test_pos][continuous_cols]
)

dataset.metadata.loc[:, continuous_cols] = scaler.transform(
    test_dataset.metadata[continuous_cols]
)



trex_model = OCT_CNN_CLASSIFIER()
trex_model.load_state_dict('weights_classifier.torch')



actual_trex, predicted_probs = get_trex_predictions(
    trex_model,
    trex_test_loader,
    device
)

plot_trex_results(actual_trex, predicted_probs)

plot_trex_results(actual_trex, predicted_probs)