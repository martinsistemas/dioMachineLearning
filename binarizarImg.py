import numpy as np
import cv2

def rgb_to_grayscale(image):
    """Converte uma imagem colorida para níveis de cinza."""
    gray_image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    return gray_image.astype(np.uint8)

def binarize_image(gray_image, threshold=128):
    """Binariza uma imagem em escala de cinza."""
    binary_image = np.where(gray_image >= threshold, 255, 0)
    return binary_image.astype(np.uint8)

def process_image(image_path, threshold=128):
    # Carregar a imagem colorida
    image = cv2.imread(image_path)
    
    if image is None:
        print("Erro ao carregar a imagem.")
        return
    
    # Converter para escala de cinza manualmente
    gray_image = rgb_to_grayscale(image)
    
    # Binarizar a imagem manualmente
    binary_image = binarize_image(gray_image, threshold)
    
    # Exibir imagens
    cv2.imshow("Original", image)
    cv2.imshow("Escala de Cinza", gray_image)
    cv2.imshow("Binarizada", binary_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Salvar as imagens processadas
    cv2.imwrite("gray_image.jpg", gray_image)
    cv2.imwrite("binary_image.jpg", binary_image)

# Exemplo de uso
image_path = "lena.jpg"  # Substituir pelo caminho da imagem
process_image(image_path)
