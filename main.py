

import os
from PIL import Image
from disturb_image import disturb_image
from save_image import save_image
from test_classifier import test_classifier
import requests
import numpy as np

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

      request = requests.post(
        'http://ec2-54-85-67-162.compute-1.amazonaws.com:8080/classify',
        headers=headers,
        files=files,
      )
      
      
      request.raise_for_status()
      return request.text        
  except Exception as e:
      return "Error: " + str(e)

def save_image(image, directory, name):
  final_path = os.path.join(directory, name)
  image.save(final_path)

  return final_path

def main():
  disturbed_images_dir = 'disturbed_images'
  approved_images_dir = 'approved_images'

  if not os.path.exists(disturbed_images_dir):
    os.makedirs(disturbed_images_dir)

  if not os.path.exists(approved_images_dir):
    os.makedirs(approved_images_dir)

  original_image = Image.open("images/reprovado.jpg")
  types = ['gaussian', 'uniform']

  for noise_type in types:
    for i in range(5):    
      epsilon = 0.1 * (i + 1)

      image_name = f"{noise_type}-{i}.jpg"

      disturbed_image = disturb_image(original_image, epsilon=epsilon  , noise_type=noise_type)
      final_path = save_image(disturbed_image, disturbed_images_dir, image_name)
      result = test_classifier(final_path).lower()

      if "aprovado" in result:
        print(f"Aprroved Image ({image_name}): {noise_type} with epsilon={epsilon:.2f}")
        save_image(disturbed_image, approved_images_dir, image_name)
        break

if __name__ == '__main__':
    main()