import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

base_dir = "logs/blender_lego_fine"

iterations = ["000500", "002000", "005000", "010000", "020000"]

test_images = ["000", "001", "002"]


fig, axes = plt.subplots(6, 5, figsize=(7, 8))
plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0.02, right=0.98, top=0.93, bottom=0.02)

for row_idx, img_id in enumerate(test_images):
    for col_idx, it in enumerate(iterations):
        folder = f"testset_{it}"
        rgb_path = os.path.join(base_dir, folder, f"{img_id}_rgb.png")
        depth_path = os.path.join(base_dir, folder, f"{img_id}_depth.png")

        # === RGB image ===
        rgb_img = mpimg.imread(rgb_path)
        axes[row_idx * 2, col_idx].imshow(rgb_img)
        axes[row_idx * 2, col_idx].axis('off')
        if row_idx == 0:
            axes[row_idx * 2, col_idx].set_title(f"iter: {int(it)}", fontsize=10)

        # === Depth map ===
        depth_img = mpimg.imread(depth_path)
        im = axes[row_idx * 2 + 1, col_idx].imshow(depth_img, cmap='plasma')
        axes[row_idx * 2 + 1, col_idx].axis('off')

    label_positions = [0.81, 0.49, 0.17]

    for i, ypos in enumerate(label_positions):
        fig.text(
            0.015, ypos,                     # (x, y) position in figure coordinates
            f"Test image {i}",               # Label text
            fontsize=10,
            rotation=90,                     # Rotate vertically
            va="center", ha="center"         # Center alignment
        )

plt.tight_layout()
plt.savefig("fine_rendered_results.png", dpi=900)
plt.show()
