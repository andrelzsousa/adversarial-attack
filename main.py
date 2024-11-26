

import os
from PIL import Image
import requests
import numpy as np

DISTURBED_IMAGES_DIR = 'disturbed_images'
APPROVED_IMAGES_DIR = 'approved_images'

def disturb_image(image, epsilon, noise_type):
  image_array = np.array(image).astype(np.float32) / 255
  
  if noise_type == 'uniform':
    noise = np.random.uniform(-epsilon, epsilon, image_array.shape)
  elif noise_type == 'gaussian':
    noise = np.random.normal(0, epsilon, image_array.shape)
  else :
    raise ValueError("Invalid noise type")
  
  disturbed_array = np.clip(image_array + noise * 2, 0, 1)
  disturbed_array = (disturbed_array - 0.5) * 1.5 + 0.5
  disturbed_array = np.clip(disturbed_array, 0, 1)
  
  return Image.fromarray((disturbed_array * 255).astype(np.uint8))

def test_classifier(path):
  try:
    with open(path, 'rb') as image:
      files = {'file': image}

      headers = {
        'Accept': '*/*',
        'User-Agent': 'python-requests'
      }

      res = requests.post(
        'http://ec2-54-85-67-162.compute-1.amazonaws.com:8080/classify',
        headers=headers,
        files=files,
      )
      
      res.raise_for_status()
      return res.text        
  except Exception as e:
      return "Error: " + str(e)

def save_image(image, directory, name):
  final_path = os.path.join(directory, name)
  image.save(final_path)

  return final_path

def main():
  if not os.path.exists(DISTURBED_IMAGES_DIR):
    os.makedirs(DISTURBED_IMAGES_DIR)

  if not os.path.exists(APPROVED_IMAGES_DIR):
    os.makedirs(APPROVED_IMAGES_DIR)

  original_image = Image.open("images/reprovado.jpg")
  types = ['gaussian', 'uniform']

  for noise_type in types:
    for i in range(5):    
      epsilon = (i + 1) * 0.1
      image_name = f"{noise_type}-{i}.jpg"

      disturbed_image = disturb_image(original_image, epsilon=epsilon  , noise_type=noise_type)
      final_path = save_image(disturbed_image, DISTURBED_IMAGES_DIR, image_name)
      result = test_classifier(final_path).lower()

      if "aprovado" in result:
        print(f"Aprroved Image ({image_name}): {noise_type} with epsilon={epsilon:.2f}")
        save_image(disturbed_image, APPROVED_IMAGES_DIR, image_name)
        break

if __name__ == '__main__':
    main()