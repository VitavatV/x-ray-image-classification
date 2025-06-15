import re

def extract_log_info(filepath):
    info = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Extract mean and std (first line)
    mean_std_match = re.match(r"([0-9.]+)\s+([0-9.]+)", lines[0])
    if mean_std_match:
        info['mean'] = float(mean_std_match.group(1))
        info['std'] = float(mean_std_match.group(2))

    # Extract class names
    for line in lines:
        if line.startswith("Class Names:"):
            info['class_names'] = eval(line.split(":", 1)[1].strip())
        if line.startswith("train size:"):
            info['train_size'] = int(line.split(":")[1].strip())
        if line.startswith("val size:"):
            info['val_size'] = int(line.split(":")[1].strip())
        if line.startswith("Start training"):
            break

    # Extract epoch results
    info['epochs'] = []
    for line in lines:
        epoch_match = re.match(
            r"Epoch (\d+)/(\d+), Train Loss: ([0-9.]+), Train Acc: ([0-9.]+), Val Loss: ([0-9.]+), Val Acc: ([0-9.]+)", line)
        if epoch_match:
            info['epochs'].append({
                'epoch': int(epoch_match.group(1)),
                'train_loss': float(epoch_match.group(3)),
                'train_acc': float(epoch_match.group(4)),
                'val_loss': float(epoch_match.group(5)),
                'val_acc': float(epoch_match.group(6)),
            })

    # Extract final reports
    def extract_report(report_name):
        start = None
        for i, line in enumerate(lines):
            if line.strip() == report_name:
                start = i + 1
                break
        if start is not None:
            report_lines = []
            for line in lines[start:]:
                if (line.strip() == "") or ("Model" in line) or ("val_report" in line):
                    break
                report_lines.append(line.rstrip())
            return "\n".join(report_lines)
        return None

    info['train_report'] = extract_report('train_report')
    info['val_report'] = extract_report('val_report')

    # Extract model and config info at the end
    for line in lines[::-1]:
        if line.startswith("data_dir :"):
            info['data_dir'] = line.split(":", 1)[1].strip()
        if line.startswith("image (H,W,C) :"):
            info['image_shape'] = line.split(":", 1)[1].strip()
        if line.startswith("datadataset_name_dir :"):
            info['dataset_name'] = line.split(":", 1)[1].strip()
        if line.startswith("transform_mode :"):
            info['transform_mode'] = line.split(":", 1)[1].strip()
        if line.startswith("model_name :"):
            info['model_name'] = line.split(":", 1)[1].strip()
        if line.startswith("num_classes :"):
            info['num_classes'] = int(line.split(":", 1)[1].strip())
        if line.startswith("batch_size :"):
            info['batch_size'] = int(line.split(":", 1)[1].strip())
        if line.startswith("epochs :"):
            info['epochs_total'] = int(line.split(":", 1)[1].strip())
        if line.startswith("num_workers :"):
            info['num_workers'] = int(line.split(":", 1)[1].strip())
        if line.startswith("Model saved as"):
            info['model_file'] = line.split("Model saved as", 1)[1].strip()

    return info

if __name__ == "__main__":
    import os
    file_list = os.listdir("results")
    file_list.sort()  # Sort files to maintain order
    information = {}
    for log_file in file_list:
        if log_file.endswith(".txt"):
            log_path = os.path.join("results", log_file)
            print(f"Extracting info from {log_path}...")
            info = extract_log_info(log_path)
            nlog_file = log_file.replace('chestx_ray14', 'chestx-ray14')
            nlog_file = nlog_file.replace('efficientnet_b0', 'efficientnet-b0')
            nlog_file = nlog_file.replace('mimic_cxr', 'mimic-cxr')
            nlog_file = nlog_file.replace('.txt', '')
            nlog_file = nlog_file.replace('log_', '')
            information[nlog_file] = info
    for k, v in information[list(information.keys())[0]].items():
        print(f"{k}: {v}")

    print()
    print("\nFinal Epoch Results:")

# ...existing code...
import pandas as pd
import matplotlib.pyplot as plt

header = ["dataset", "preprocessing", "model_name", "train_precision", "train_recall", "train_f1score", "Train_accuracy", "val_precision", "val_recall", "val_f1score", "Val_accuracy"]
data = []

for k, v in information.items():
    dataset, preprocessing, model_name = k.split('_')
    print(f"\nDataset: {dataset}, Preprocessing: {preprocessing}, Model: {model_name}")
    train_report_acc_line = [v_i for v_i in v['train_report'].split('\n')[3].split(' ') if v_i != '']
    val_report_acc_line = [v_i for v_i in v['val_report'].split('\n')[3].split(' ') if v_i != '']
    data.append([
        dataset,
        preprocessing,
        model_name,
        train_report_acc_line[1],
        train_report_acc_line[2],
        train_report_acc_line[3],
        train_report_acc_line[4],
        val_report_acc_line[1],
        val_report_acc_line[2],
        val_report_acc_line[3],
        val_report_acc_line[4]
    ])
    
# Create DataFrame
df = pd.DataFrame(data, columns=header)
df.to_csv("results/summary_f1.csv", index=False)
print(df)

# # Prepare data for DataFrame
# header = ["dataset", "preprocessing", "model_name", "train_acc", "val_acc", "train_loss", "val_loss"]
# data = []
# for k, v in information.items():
#     dataset, preprocessing, model_name = k.split('_')
#     data.append([
#         dataset,
#         preprocessing,
#         model_name,
#         v['epochs'][-1]['train_acc'],
#         v['epochs'][-1]['val_acc'],
#         v['epochs'][-1]['train_loss'],
#         v['epochs'][-1]['val_loss']
#     ])

# # Create DataFrame
# df = pd.DataFrame(data, columns=header)
# df.to_csv("results/summary.csv", index=False)
# print(df)

# # Compare val_acc vs train_acc and val_loss vs train_loss for each dataset
# for dataset in df['dataset'].unique():
#     subset_df = df[df['dataset'] == dataset]

#     # Plot Accuracy Comparison
#     plt.figure(figsize=(10, 6))
#     x_labels = [f"{row['model_name']}\n{row['preprocessing']}" for _, row in subset_df.iterrows()]
#     plt.bar(x_labels, subset_df['train_acc'], width=0.4, label='Train Acc', align='edge')
#     plt.bar(x_labels, subset_df['val_acc'], width=-0.4, label='Val Acc', align='edge')
#     plt.ylabel('Accuracy')
#     plt.title(f'Train vs Val Accuracy ({dataset})')
#     plt.xticks(rotation=45)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f"results/acc_compare_{dataset}.png")
#     plt.close()

#     # Plot Loss Comparison
#     plt.figure(figsize=(10, 6))
#     plt.bar(x_labels, subset_df['train_loss'], width=0.4, label='Train Loss', align='edge')
#     plt.bar(x_labels, subset_df['val_loss'], width=-0.4, label='Val Loss', align='edge')
#     plt.ylabel('Loss')
#     plt.title(f'Train vs Val Loss ({dataset})')
#     plt.xticks(rotation=45)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f"results/loss_compare_{dataset}.png")
#     plt.close()

# fig, ax = plt.subplots(figsize=(16, 2 + len(df) * 0.5))
# ax.axis('off')
# table = ax.table(
#     cellText=df.values,
#     colLabels=df.columns,
#     cellLoc='center',
#     loc='center'
# )
# table.auto_set_font_size(False)
# table.set_fontsize(12)
# table.auto_set_column_width(col=list(range(len(df.columns))))
# plt.title("Summary Table", fontsize=14, pad=20)
# plt.tight_layout()
# plt.savefig("results/summary_table.png")
# plt.close()