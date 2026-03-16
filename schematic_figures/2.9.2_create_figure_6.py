import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Output files
output_png = "figure1_pipeline_overview_two_rows_final.png"
output_pdf = "figure1_pipeline_overview_two_rows_final.pdf"

# Figure setup
fig, ax = plt.subplots(figsize=(13.5, 7.2))
ax.set_xlim(0, 13.5)
ax.set_ylim(0, 7.2)
ax.axis("off")

# Box styling
box_width = 2.25
box_height = 1.25
box_fontsize = 12
group_fontsize = 11

# Step definitions: (label, x, y, color)
steps = [
    ("Raw whole-brain\nimages", 1.8, 5.45, "#f2f2f2"),
    ("LAS X\nTIFF export", 4.5, 5.45, "#f2f2f2"),
    ("ImageJ\npreprocessing", 7.2, 5.45, "#f2f2f2"),
    ("Cellpose\nnuclei model training", 9.9, 5.45, "#e8f1fb"),
    ("QuPath\nc-Fos classifier training", 9.9, 2.55, "#e8f1fb"),
    ("ABBA + DeepSlice\natlas registration", 6.7, 2.55, "#eaf7ea"),
    ("Region-wise\nquantification", 3.5, 2.55, "#fff1dd"),
]

# Helper function to draw boxes
def draw_box(label, x, y, color):
    box = FancyBboxPatch(
        (x - box_width / 2, y - box_height / 2),
        box_width,
        box_height,
        boxstyle="round,pad=0.03,rounding_size=0.09",
        linewidth=1.5,
        edgecolor="black",
        facecolor=color
    )
    ax.add_patch(box)
    ax.text(
        x, y, label,
        ha="center", va="center",
        fontsize=box_fontsize
    )

# Draw all boxes
for label, x, y, color in steps:
    draw_box(label, x, y, color)

# Helper function to draw arrows
def draw_arrow(x1, y1, x2, y2):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=14,
        linewidth=1.2,
        color="black"
    )
    ax.add_patch(arrow)

# Top row arrows
for i in range(3):
    x1 = steps[i][1] + box_width / 2
    y1 = steps[i][2]
    x2 = steps[i + 1][1] - box_width / 2
    y2 = steps[i + 1][2]
    draw_arrow(x1, y1, x2, y2)

# Vertical arrow from Cellpose to QuPath
draw_arrow(
    steps[3][1], steps[3][2] - box_height / 2,
    steps[4][1], steps[4][2] + box_height / 2
)

# Bottom row arrows (right to left)
draw_arrow(
    steps[4][1] - box_width / 2, steps[4][2],
    steps[5][1] + box_width / 2, steps[5][2]
)

draw_arrow(
    steps[5][1] - box_width / 2, steps[5][2],
    steps[6][1] + box_width / 2, steps[6][2]
)

# Helper function to draw group brackets
def draw_group(x_start, x_end, y, label):
    ax.plot([x_start, x_end], [y, y], color="black", linewidth=1.2)
    ax.plot([x_start, x_start], [y, y - 0.16], color="black", linewidth=1.2)
    ax.plot([x_end, x_end], [y, y - 0.16], color="black", linewidth=1.2)
    ax.text(
        (x_start + x_end) / 2,
        y + 0.14,
        label,
        ha="center", va="bottom",
        fontsize=group_fontsize,
        fontweight="bold"
    )

# Group brackets
draw_group(
    steps[0][1] - box_width / 2,
    steps[2][1] + box_width / 2,
    6.25,
    "Image preparation"
)

draw_group(
    steps[4][1] - box_width / 2,
    steps[3][1] + box_width / 2,
    6.25,
    "Model preparation"
)

draw_group(
    steps[5][1] - box_width / 2,
    steps[5][1] + box_width / 2,
    3.45,
    "Atlas registration"
)

draw_group(
    steps[6][1] - box_width / 2,
    steps[6][1] + box_width / 2,
    3.45,
    "Quantification"
)

plt.subplots_adjust(left=0.03, right=0.97, top=0.96, bottom=0.06)
plt.savefig(output_png, dpi=300)
plt.savefig(output_pdf)
plt.show()
