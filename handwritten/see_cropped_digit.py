import scipy.io
import matplotlib.pyplot as plt

# Load the .mat file
mat = scipy.io.loadmat('C:/Users/User/Desktop/handwritten/train_32x32.mat')

# Access images and labels
X = mat['X']
y = mat['y']

# Display a few images
num_images_to_display = 20
for i in range(num_images_to_display):
    plt.subplot(1, num_images_to_display, i+1)
    plt.imshow(X[:, :, :, i])
    plt.title(f"Label: {y[i]}")
    plt.axis('off')

plt.show()
