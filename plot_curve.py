import pandas as pd
import matplotlib.pyplot as plt

loss_file = "logs_data/coarse_loss.csv"
psnr_file = "logs_data/coarse_train_PSNR.csv"
# loss_file = "logs_data/fine_loss.csv"
# psnr_file = "logs_data/fine_train_PSNR.csv"

loss_df = pd.read_csv(loss_file)
psnr_df = pd.read_csv(psnr_file)

print(loss_df.head())
print(psnr_df.head())

loss_iter_col = loss_df.columns[1]
loss_val_col = loss_df.columns[2]
psnr_iter_col = psnr_df.columns[1]
psnr_val_col = psnr_df.columns[2]

# === Plot Training Loss ===
plt.figure(figsize=(14, 6))
plt.plot(loss_df[loss_iter_col], loss_df[loss_val_col], color='tab:blue')
plt.title("Training Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("coarse_training_loss_curve.png", dpi=600, bbox_inches='tight')
# plt.savefig("fine_training_loss_curve.png", dpi=600, bbox_inches='tight')
plt.show()

# === Plot Training PSNR ===
plt.figure(figsize=(14, 6))
plt.plot(psnr_df[psnr_iter_col], psnr_df[psnr_val_col], color='tab:blue')
plt.title("Training PSNR Curve")
plt.xlabel("Iteration")
plt.ylabel("PSNR")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("coarse_training_PSNR_curve.png", dpi=600, bbox_inches='tight')
# plt.savefig("fine_training_PSNR_curve.png", dpi=600, bbox_inches='tight')
plt.show()
