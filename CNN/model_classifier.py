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

        self.clinical_predictors = [ 'CST','BCVA','nadir_CST','delta_CST','pct_change_CST','pct_above_nadir']



        self.metadata["First_Visit"] = (
            self.metadata["delta_CST"].isna()
        ).astype(np.float32)

        self.metadata["delta_CST"] = (
            self.metadata["delta_CST"].fillna(0)
        )

        self.metadata["pct_change_CST"] = (
            self.metadata["pct_change_CST"].fillna(0)
        )

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
def train_model(model, train_loader, optimizer, device):

    model.train()

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    total_loss = 0

    for images, clinical, y_reg, y_class in train_loader:

        images = images.to(device)
        clinical = clinical.to(device)
        y_reg = y_reg.to(device)
        y_class = y_class.to(device)
        optimizer.zero_grad()

        cst_pred, class_pred = model(images,clinical)

        loss_reg = mse(cst_pred, y_reg)
        loss_class = bce(class_pred, y_class)

        # Weight the two losses
        loss =  loss_reg # + (.5 * loss_class)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

os.chdir(r'F:\Data\Edemea Models Personal')
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

valid_metadata = dataset.metadata.iloc[valid_indices]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_pos, test_pos = next(
    splitter.split(
        valid_metadata,
        groups=valid_metadata["Patient_ID"]
    )
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
    dataset.metadata.iloc[train_pos][continuous_cols]
)

dataset.metadata.loc[:, continuous_cols] = scaler.transform(
    dataset.metadata[continuous_cols]
)

train_dataset = Subset(filtered_dataset, train_pos)



loader = DataLoader(train_dataset,2,shuffle = True)





images, clinical, y_reg, y_class = next(iter(loader))

print("Images:", images.shape)
print("Clinical:", clinical.shape)
print("Regression:", y_reg.shape)
print("Classification:", y_class.shape)







# device = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )

# model = OCT_CNN().to(device)

# optimizer = torch.optim.Adam(
#     model.parameters(),
#     lr=0.001
# )

# for epoch in range(20):

#     loss = train_model(
#         model,
#         loader,
#         optimizer,
#         device
#     )

#     print(f"Epoch {epoch+1}: Loss = {loss:.4f}")


# torch.save(model.state_dict(),'weights.torch')








from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def evaluate_model(model, test_loader, device):

    model.eval()

    actual_cst = []
    predicted_cst = []

    actual_class = []
    predicted_prob = []

    with torch.no_grad():

        for images, clinical, y_reg, y_class in test_loader:

            images = images.to(device)
            clinical = clinical.to(device)
            # Model predictions
            cst_pred, class_logits = model(images,clinical)

            # Convert logits to probabilities
            probabilities = torch.sigmoid(class_logits)

            # Store predictions and actual values
            actual_cst.extend(y_reg.cpu().numpy().flatten())
            predicted_cst.extend(cst_pred.cpu().numpy().flatten())

            actual_class.extend(y_class.cpu().numpy().flatten())
            predicted_prob.extend(probabilities.cpu().numpy().flatten())

    # Convert to NumPy arrays
    actual_cst = np.array(actual_cst)
    predicted_cst = np.array(predicted_cst)

    actual_class = np.array(actual_class)
    predicted_prob = np.array(predicted_prob)

    # Convert probabilities to binary predictions
    predicted_class = (predicted_prob >= 0.5).astype(int)

    # Regression metrics
    mae = mean_absolute_error(actual_cst, predicted_cst)
    rmse = np.sqrt(mean_squared_error(actual_cst, predicted_cst))
    r2 = r2_score(actual_cst, predicted_cst)

    # Classification metrics
    accuracy = accuracy_score(actual_class, predicted_class)
    precision = precision_score(
        actual_class, predicted_class, zero_division=0
    )
    recall = recall_score(
        actual_class, predicted_class, zero_division=0
    )
    f1 = f1_score(
        actual_class, predicted_class, zero_division=0
    )

    # ROC-AUC requires both classes to be present
    auc = (
        roc_auc_score(actual_class, predicted_prob)
        if len(np.unique(actual_class)) == 2
        else np.nan
    )

    results = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc
    }

    return results


# test_dataset = Subset(filtered_dataset, test_pos)

# test_loader = DataLoader(test_dataset,2,shuffle = True)
# results = evaluate_model(model,test_loader,device)
# results








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

def train_model_classifier(model, train_loader, optimizer, device):

    model.train()

    #mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    total_loss = 0

    for images, clinical, y_reg, y_class in train_loader:

        images = images.to(device)
        clinical = clinical.to(device)
        y_reg = y_reg.to(device)
        y_class = y_class.to(device)
        optimizer.zero_grad()

        cst_pred, class_pred = model(images,clinical)

        #loss_reg = mse(cst_pred, y_reg)
        loss_class = bce(class_pred, y_class)

        # Weight the two losses
        loss =   loss_class

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)





model_classifier  = OCT_CNN_CLASSIFIER().to(device)

optimizer = torch.optim.Adam(
    model_classifier.parameters(),
    lr=0.001
)

for epoch in range(15):

    loss = train_model_classifier(
        model_classifier,
        loader,
        optimizer,
        device
    )

    print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

torch.save(model_classifier.state_dict(),'weights_classifier.torch')

test_dataset = Subset(filtered_dataset, test_pos)

test_loader = DataLoader(test_dataset,2,shuffle = True)
results = evaluate_model(model_classifier,test_loader,device)
results
