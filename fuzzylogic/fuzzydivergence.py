import numpy as np
import matplotlib.pyplot as plt

# 1. Define Membership Function
def sigmoid_membership(pixel_values, center, slope):
    """
    Compute fuzzy membership values using a sigmoid function.
    """
    return 1 / (1 + np.exp(-(pixel_values - center) / slope))

# 2. Compute Fuzzy Divergence
def fuzzy_divergence(image, reference, membership_func, p=1.5, q=0.5):
    """
    Compute fuzzy divergence between an image and its reference.
    Args:
        image: Input image as a 2D numpy array.
        reference: Reference image (ideal segmentation).
        membership_func: Membership function to calculate fuzzy values.
        p: Concentration exponent.
        q: Dilation exponent.
    Returns:
        Total fuzzy divergence between the two images.
    """
    # Compute membership values
    mu_image = membership_func(image)
    mu_ref = membership_func(reference)
    
    # Apply hedge operators
    mu_image_con = mu_image ** p
    mu_ref_dil = mu_ref ** q

    # Compute D1 and D2

    # aggregate divergnece values over the image
    D1 = np.sum(
        1 - (1 - mu_image_con) * np.exp(mu_image_con - mu_ref_dil) - 
        mu_image_con * np.exp(mu_ref_dil - mu_image_con)
    )
    
    D2 = np.sum(
        1 - (1 - mu_ref_dil) * np.exp(mu_ref_dil - mu_image_con) - 
        mu_ref_dil * np.exp(mu_image_con - mu_ref_dil)
    )
    
    # Total divergence
    return D1 + D2

def main():
    # Generate a synthetic grayscale image and reference
    np.random.seed(42)
    image = np.random.randint(0, 256, (100, 100))  # Simulated noisy grayscale image
    reference = np.where(image > 128, 255, 0)       # Simulated ground truth segmentation

    # Membership function parameters
    center = 128
    slope = 30

    # Compute fuzzy divergence
    membership_func = lambda img: sigmoid_membership(img, center, slope)
    divergence = fuzzy_divergence(image, reference, membership_func)
    
    print(f"Fuzzy Divergence: {divergence:.2f}")

    # Compute concentrated and eroded membership images
    mu_image = membership_func(image)
    concentrated_image = np.clip(mu_image ** 1.5, 0, 1)  # Concentration with p = 1.5
    eroded_image = np.clip(mu_image ** 0.8, 0, 1)        # Erosion (dilation with q = 0.8)

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1)
    plt.title("Original Image, noisy grayscale")
    plt.imshow(image, cmap='gray')
    plt.colorbar()

    plt.subplot(2, 3, 2)
    plt.title("Reference, ground truth")
    plt.imshow(reference, cmap='gray')
    plt.colorbar()

    plt.subplot(2, 3, 3)
    plt.title("Membership Values, for original image")
    plt.imshow(mu_image, cmap='Reds')
    plt.colorbar(label="Membership")

    plt.subplot(2, 3, 4)
    plt.title("Concentrated Image, sharpened")
    plt.imshow(concentrated_image, cmap='Reds')
    plt.colorbar(label="Concentration")

    plt.subplot(2, 3, 5)
    plt.title("Eroded Image")
    plt.imshow(eroded_image, cmap='Reds')
    plt.colorbar(label="Erosion")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()




