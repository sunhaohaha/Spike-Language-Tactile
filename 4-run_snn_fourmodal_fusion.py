import os
import copy
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("PyTorch version:", torch.__version__)
print("PyTorch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

USE_PCA = True
PCA_COMPONENTS = 50

USE_OVERSAMPLING = False

def find_excel_file(stem):
    names = [
        stem,
        stem + ".xlsx",
        stem + ".xls",
        stem + ".xlsm"
    ]

    search_dirs = [
        os.getcwd(),
        DATA_DIR,
        SCRIPT_DIR
    ]

    candidates = []
    for name in names:
        if os.path.isabs(name):
            candidates.append(name)
        else:
            candidates.extend(os.path.join(search_dir, name) for search_dir in search_dirs)

    for file in candidates:
        if os.path.exists(file):
            return file

    raise FileNotFoundError(f"File not found: {stem}. Please check the file name and path.")

modalities = {
    "temperature": {
        "train_candidates": ["temperature_X_train"],
        "test_candidates": ["temperature_X_test"]
    },
    "weight": {
        "train_candidates": ["weight_X_train"],
        "test_candidates": ["weight_X_test"]
    },
    "materials": {
        "train_candidates": ["materials_X_train"],
        "test_candidates": ["materials_X_test"]

    },
    "shape": {
        "train_candidates": ["shape_X_train"],
        "test_candidates": ["shape_X_test"]
    }
}

Y_train_file = find_excel_file("Y_train")
Y_test_file = find_excel_file("Y_test")

class LIFNeuronLayer(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_neurons,
        tau=20,
        threshold=0.3,
        refractory_period=5,
        dropout_rate=0.2
    ):
        super(LIFNeuronLayer, self).__init__()

        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.tau = tau
        self.threshold = threshold
        self.refractory_period = refractory_period
        self.dropout = nn.Dropout(dropout_rate)

        self.weights = nn.Parameter(torch.randn(num_inputs, num_neurons) * 0.1)

    def forward(self, x):
        membrane_potential = torch.matmul(x, self.weights)

        hard_spikes = (membrane_potential >= self.threshold).float()

        surrogate_scale = 10.0
        soft_spikes = torch.sigmoid(
            surrogate_scale * (membrane_potential - self.threshold)
        )

        spikes = hard_spikes.detach() - soft_spikes.detach() + soft_spikes
        spikes = self.dropout(spikes)

        return spikes

class SNN(nn.Module):
    def __init__(self, num_inputs, num_neurons, num_hidden_neurons, num_classes):
        super(SNN, self).__init__()

        self.lif_layer1 = LIFNeuronLayer(num_inputs, num_hidden_neurons)
        self.bn1 = nn.BatchNorm1d(num_hidden_neurons)

        self.lif_layer2 = LIFNeuronLayer(num_hidden_neurons, num_neurons)
        self.bn2 = nn.BatchNorm1d(num_neurons)

        self.fc = nn.Linear(num_neurons, num_classes)

    def forward(self, x):
        x = self.lif_layer1(x)
        x = self.bn1(x)
        x = self.lif_layer2(x)
        x = self.bn2(x)
        x = self.fc(x)
        return x

def evaluate(model, X, Y, criterion=None):
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        outputs = model(X)
        loss = criterion(outputs, Y)
        _, predicted = torch.max(outputs, 1)

        y_true = Y.cpu().numpy()
        y_pred = predicted.cpu().numpy()

        acc = accuracy_score(y_true, y_pred)

    return outputs.cpu(), predicted.cpu(), acc, loss.item()

def resolve_modality_file(file_info, split_name):
    key = f"{split_name}_candidates"

    for candidate in file_info[key]:
        try:
            return find_excel_file(candidate)
        except FileNotFoundError:
            pass

    raise FileNotFoundError(f"Could not find the {split_name} file: {file_info[key]}")

def load_all_modalities():
    all_data = {}

    for modality_name, file_info in modalities.items():
        train_file = resolve_modality_file(file_info, "train")
        test_file = resolve_modality_file(file_info, "test")

        X_train = pd.read_excel(train_file, header=None).values
        X_test = pd.read_excel(test_file, header=None).values

        if np.isnan(X_train).any() or np.isnan(X_test).any():
            raise ValueError(f"{modality_name}: NaN values were found. Please check the Excel file first.")

        all_data[modality_name] = {
            "train_file": train_file,
            "test_file": test_file,
            "X_train": X_train,
            "X_test": X_test,
            "num_train": X_train.shape[0],
            "num_test": X_test.shape[0],
            "num_features": X_train.shape[1]
        }

        print(
            f"{modality_name}: "
            f"train={X_train.shape}, test={X_test.shape}, "
            f"train_file={train_file}, test_file={test_file}"
        )

    return all_data

def prepare_labels(output_root):
    Y_train_orig = pd.read_excel(Y_train_file, header=None).values.squeeze()
    Y_test_orig = pd.read_excel(Y_test_file, header=None).values.squeeze()

    original_classes = np.unique(Y_train_orig)
    original_classes = np.sort(original_classes)

    label_to_index = {label: idx for idx, label in enumerate(original_classes)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    Y_train_mapped = np.array(
        [label_to_index[y] for y in Y_train_orig],
        dtype=np.int64
    )

    for y in np.unique(Y_test_orig):
        if y not in label_to_index:
            raise ValueError(f"Test label {y} does not appear in the training set and cannot be mapped.")

    Y_test_mapped = np.array(
        [label_to_index[y] for y in Y_test_orig],
        dtype=np.int64
    )

    num_classes = len(original_classes)

    class_mapping_df = pd.DataFrame({
        "model_class_index": list(index_to_label.keys()),
        "original_label": list(index_to_label.values()),
        "display_class": [f"Class {i + 1}" for i in range(num_classes)]
    })

    class_mapping_df.to_excel(
        os.path.join(output_root, "class_mapping_global.xlsx"),
        index=False
    )

    return Y_train_orig, Y_test_orig, Y_train_mapped, Y_test_mapped, index_to_label, num_classes

def check_data_consistency(all_data, Y_train_mapped, Y_test_mapped):
    train_nums = []
    test_nums = []

    for modality_name, data in all_data.items():
        train_nums.append(data["num_train"])
        test_nums.append(data["num_test"])

        if data["num_train"] != len(Y_train_mapped):
            raise ValueError(
                f"{modality_name}: X_train and Y_train have inconsistent sample counts: "
                f"{data['num_train']} vs {len(Y_train_mapped)}"
            )

        if data["num_test"] != len(Y_test_mapped):
            raise ValueError(
                f"{modality_name}: X_test and Y_test have inconsistent sample counts: "
                f"{data['num_test']} vs {len(Y_test_mapped)}"
            )

    if len(set(train_nums)) != 1:
        raise ValueError(f"The train sample counts are inconsistent across modalities: {train_nums}")

    if len(set(test_nums)) != 1:
        raise ValueError(f"The test sample counts are inconsistent across modalities: {test_nums}")

    print("Data consistency check passed: all modalities have matching train/test sample counts.")

def random_oversample_if_needed(X, y, random_state=SEED):
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    if len(set(counts)) == 1:
        print("The training subset is already balanced. Oversampling skipped.")
        return X, y

    rng = np.random.default_rng(random_state)
    sampled_indices = []

    for cls, count in zip(classes, counts):
        cls_indices = np.flatnonzero(y == cls)
        extra_indices = rng.choice(
            cls_indices,
            size=max_count - count,
            replace=True
        )
        sampled_indices.extend(cls_indices.tolist())
        sampled_indices.extend(extra_indices.tolist())

    sampled_indices = np.array(sampled_indices)
    rng.shuffle(sampled_indices)

    return X[sampled_indices], y[sampled_indices]

def run_fourmodal_fusion(
    all_data,
    Y_train_mapped,
    Y_test_mapped,
    index_to_label,
    num_classes,
    output_root,
    num_epochs=1000,
    patience=50,
    lr=0.0001,
    weight_decay=1e-4,
    num_neurons=1024,
    num_hidden_neurons=2048
):

    modality_names = list(modalities.keys())

    fusion_name = "+".join(modality_names)
    folder_name = "_".join(modality_names)

    print("\n" + "=" * 90)
    print(f"Start four-modal fusion: {fusion_name}")
    print("=" * 90)

    fusion_dir = os.path.join(output_root, folder_name)
    os.makedirs(fusion_dir, exist_ok=True)

    X_train_list = []
    X_test_list = []
    feature_info = {}

    for modality_name in modality_names:
        X_train_modality = all_data[modality_name]["X_train"]
        X_test_modality = all_data[modality_name]["X_test"]

        X_train_list.append(X_train_modality)
        X_test_list.append(X_test_modality)

        feature_info[modality_name] = X_train_modality.shape[1]

        print(f"{modality_name} feature dim: {X_train_modality.shape[1]}")

    X_train_orig = np.concatenate(X_train_list, axis=1)
    X_test_orig = np.concatenate(X_test_list, axis=1)

    num_features_fused = X_train_orig.shape[1]

    print(f"Fused feature dim: {num_features_fused}")

    fusion_info = {
        "fusion_name": fusion_name,
        "fusion_method": "feature_concatenation",
        "num_features_fused": num_features_fused
    }

    for modality_name in modality_names:
        fusion_info[f"{modality_name}_train_file"] = all_data[modality_name]["train_file"]
        fusion_info[f"{modality_name}_test_file"] = all_data[modality_name]["test_file"]
        fusion_info[f"{modality_name}_num_features"] = feature_info[modality_name]

    fusion_info_df = pd.DataFrame([fusion_info])

    fusion_info_df.to_excel(
        os.path.join(fusion_dir, "fusion_info.xlsx"),
        index=False
    )

    class_mapping_df = pd.DataFrame({
        "model_class_index": list(index_to_label.keys()),
        "original_label": list(index_to_label.values()),
        "display_class": [f"Class {i + 1}" for i in range(num_classes)]
    })

    class_mapping_df.to_excel(
        os.path.join(fusion_dir, "class_mapping.xlsx"),
        index=False
    )

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_orig,
        Y_train_mapped,
        test_size=0.25,
        stratify=Y_train_mapped,
        random_state=SEED
    )

    if USE_OVERSAMPLING:
        X_train, Y_train = random_oversample_if_needed(X_train, Y_train)
    else:
        print("USE_OVERSAMPLING=False. Oversampling skipped.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test_orig)

    pca_info = {
        "use_pca": USE_PCA,
        "pca_components_requested": PCA_COMPONENTS,
        "pca_components_used": 0,
        "pca_explained_variance_ratio_sum": np.nan
    }

    if USE_PCA:
        n_components = min(
            PCA_COMPONENTS,
            X_train_scaled.shape[0] - 1,
            X_train_scaled.shape[1]
        )

        pca = PCA(n_components=n_components, random_state=SEED)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_val_scaled = pca.transform(X_val_scaled)
        X_test_scaled = pca.transform(X_test_scaled)

        pca_info.update({
            "pca_components_used": n_components,
            "pca_explained_variance_ratio_sum": float(
                pca.explained_variance_ratio_.sum()
            )
        })

        print(
            f"PCA enabled: {num_features_fused} -> {n_components}, "
            f"explained variance={pca_info['pca_explained_variance_ratio_sum']:.4f}"
        )
    else:
        print("PCA disabled.")

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long).to(device)

    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.long).to(device)

    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    Y_test_tensor = torch.tensor(Y_test_mapped, dtype=torch.long).to(device)

    model = SNN(
        num_inputs=X_train_tensor.shape[1],
        num_neurons=num_neurons,
        num_hidden_neurons=num_hidden_neurons,
        num_classes=num_classes
    ).to(device)

    print("Model device:", next(model.parameters()).device)
    print("X_train device:", X_train_tensor.device)
    print("Input dim:", X_train_tensor.shape[1])
    print("Num classes:", num_classes)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_state = None
    wait = 0
    history = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train_tensor)
        train_loss = criterion(outputs, Y_train_tensor)

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        train_outputs, train_pred, train_acc, train_loss_eval = evaluate(
            model, X_train_tensor, Y_train_tensor, eval_criterion
        )

        val_outputs, val_pred, val_acc, val_loss = evaluate(
            model, X_val_tensor, Y_val_tensor, eval_criterion
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss_eval,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc
        })

        is_better = (
            val_acc > best_val_acc
            or (val_acc == best_val_acc and val_loss < best_val_loss)
        )

        if is_better:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"[{fusion_name}] "
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Train Loss: {train_loss_eval:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Train Acc: {train_acc * 100:.2f}%, "
                f"Val Acc: {val_acc * 100:.2f}%"
            )

        if wait >= patience:
            print(
                f"[{fusion_name}] Early stopping at epoch {epoch + 1}. "
                f"Best Val Loss: {best_val_loss:.4f}"
            )
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    torch.save(
        model.state_dict(),
        os.path.join(fusion_dir, "best_model.pth")
    )

    train_outputs, train_predicted, train_accuracy, train_loss_final = evaluate(
        model, X_train_tensor, Y_train_tensor, eval_criterion
    )

    val_outputs, val_predicted, val_accuracy, val_loss_final = evaluate(
        model, X_val_tensor, Y_val_tensor, eval_criterion
    )

    test_outputs, test_predicted, test_accuracy, test_loss_final = evaluate(
        model, X_test_tensor, Y_test_tensor, eval_criterion
    )

    print(f"[{fusion_name}] Final Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"[{fusion_name}] Final Val Accuracy: {val_accuracy * 100:.2f}%")
    print(f"[{fusion_name}] Final Test Accuracy: {test_accuracy * 100:.2f}%")

    history_df = pd.DataFrame(history)

    final_record = {
        "fusion_name": fusion_name,
        "num_train_after_oversampling": len(X_train),
        "num_val": len(X_val),
        "num_test": len(X_test_orig),
        "num_features_fused": num_features_fused,
        "num_features_after_preprocessing": X_train_scaled.shape[1],
        "num_classes": num_classes,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "final_train_loss": train_loss_final,
        "final_val_loss": val_loss_final,
        "final_test_loss": test_loss_final,
        "final_train_acc": train_accuracy,
        "final_val_acc": val_accuracy,
        "final_test_acc": test_accuracy,
        "final_train_acc_percent": train_accuracy * 100,
        "final_val_acc_percent": val_accuracy * 100,
        "final_test_acc_percent": test_accuracy * 100
    }

    for modality_name in modality_names:
        final_record[f"{modality_name}_num_features"] = feature_info[modality_name]

    final_record.update(pca_info)

    final_record_df = pd.DataFrame([final_record])

    acc_record_path = os.path.join(fusion_dir, "acc_record.xlsx")

    with pd.ExcelWriter(acc_record_path) as writer:
        final_record_df.to_excel(writer, sheet_name="final_acc", index=False)
        history_df.to_excel(writer, sheet_name="epoch_history", index=False)

    test_pred_np = test_predicted.numpy()
    test_true_np = Y_test_mapped

    pred_df = pd.DataFrame({
        "sample_index": np.arange(len(test_true_np)),
        "true_model_class_index": test_true_np,
        "pred_model_class_index": test_pred_np,
        "true_original_label": [index_to_label[i] for i in test_true_np],
        "pred_original_label": [index_to_label[i] for i in test_pred_np],
        "correct": test_true_np == test_pred_np
    })

    pred_df.to_excel(
        os.path.join(fusion_dir, "predictions.xlsx"),
        index=False
    )

    report = classification_report(
        test_true_np,
        test_pred_np,
        labels=list(range(num_classes)),
        target_names=[str(index_to_label[i]) for i in range(num_classes)],
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_excel(
        os.path.join(fusion_dir, "classification_report.xlsx")
    )

    cm = confusion_matrix(
        test_true_np,
        test_pred_np,
        labels=list(range(num_classes))
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"True_{index_to_label[i]}" for i in range(num_classes)],
        columns=[f"Pred_{index_to_label[i]}" for i in range(num_classes)]
    )

    cm_df.to_excel(
        os.path.join(fusion_dir, "confusion_matrix.xlsx")
    )

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[index_to_label[i] for i in range(num_classes)],
        yticklabels=[index_to_label[i] for i in range(num_classes)]
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{fusion_name} Confusion Matrix\nAcc = {test_accuracy * 100:.2f}%")
    plt.tight_layout()

    plt.savefig(
        os.path.join(fusion_dir, "confusion_matrix.png"),
        dpi=300
    )
    plt.close()

    del model
    del X_train_tensor, Y_train_tensor
    del X_val_tensor, Y_val_tensor
    del X_test_tensor, Y_test_tensor

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "fusion_name": fusion_name,
        "num_features_fused": num_features_fused,
        "num_features_after_preprocessing": X_train_scaled.shape[1],
        "num_classes": num_classes,
        "train_acc": train_accuracy,
        "val_acc": val_accuracy,
        "test_acc": test_accuracy,
        "train_acc_percent": train_accuracy * 100,
        "val_acc_percent": val_accuracy * 100,
        "test_acc_percent": test_accuracy * 100,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "output_folder": fusion_dir
    }

    for modality_name in modality_names:
        result[f"{modality_name}_num_features"] = feature_info[modality_name]

    result.update(pca_info)

    return result

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_root = f"fourmodal_fusion_results_{timestamp}"
os.makedirs(output_root, exist_ok=True)

print("\nLoading modality data...")
all_data = load_all_modalities()

print("\nLoading and processing labels...")
Y_train_orig, Y_test_orig, Y_train_mapped, Y_test_mapped, index_to_label, num_classes = prepare_labels(output_root)

print("\nChecking data consistency...")
check_data_consistency(all_data, Y_train_mapped, Y_test_mapped)

print("\nFour-modal combination to run:")
print("temperature + weight + materials + shape")

result = run_fourmodal_fusion(
    all_data=all_data,
    Y_train_mapped=Y_train_mapped,
    Y_test_mapped=Y_test_mapped,
    index_to_label=index_to_label,
    num_classes=num_classes,
    output_root=output_root,
    num_epochs=1000,
    patience=50,
    lr=0.0001,
    weight_decay=1e-4,
    num_neurons=1024,
    num_hidden_neurons=2048
)

summary_df = pd.DataFrame([result])

summary_path = os.path.join(output_root, "all_fourmodal_fusion_summary.xlsx")
summary_df.to_excel(summary_path, index=False)

print("\n" + "=" * 90)
print("Four-modal fusion experiment finished.")
print(f"Results saved in folder: {output_root}")
print(f"Summary saved to: {summary_path}")
print("=" * 90)

print(summary_df[[
    "fusion_name",
    "num_features_fused",
    "num_features_after_preprocessing",
    "train_acc_percent",
    "val_acc_percent",
    "test_acc_percent",
    "output_folder"
]])
