import os

from meltingpot import substrate
from PIL import Image
from PIL import ImageDraw
import utils


class MPSubstratesPreprocessor():
  def __init__(self, substrate_name, output_folder, scale_factor=5):
      self.substrate_name = substrate_name
      env_config = substrate.get_config(substrate_name)
      roles = env_config.default_player_roles
      self._num_players = len(roles)
      self._env = utils.env_creator({
          'substrate': substrate_name,
          'roles': roles,
      })
      self.output_folder = output_folder
      self.scale_factor = scale_factor

  def run_step(self):
    """Test step() returns rewards for all agents."""
    obs, _ = self._env.reset()
    # Create dummy actions
    actions = {}
    # turn left 4 times to get all orientations
    for player_idx in range(0, self._num_players):
      actions['player_' + str(player_idx)] = 5
    for step in range(1, 5):
      base_output_folder = os.path.join(
          self.output_folder, self.substrate_name, f'step_{step}')
      if not os.path.exists(base_output_folder):
          os.makedirs(base_output_folder)
      obs, rewards, _, _, _ = self._env.step(actions)
      world_rgb = obs['player_0']['WORLD.RGB']
      patch_coords = self.split_obs_into_patches(world_rgb, base_output_folder)

  def split_obs_into_patches(self, image, base_output_folder, sprite_size=8):
      """
      Split an image into patches of individual sprites. Scaling is used to
      make manual annotation easier. But for patch comparison during text generation
      the original patches are used.
      """
      def split_image_into_patches(image, patch_size, output_folder):
        width, height = image.size
        patch_coords = []
        patch_number = 0
        for y in range(0, height, patch_size):
          for x in range(0, width, patch_size):
              box = (x, y, x + patch_size, y + patch_size)
              # crop patches
              patch = image.crop(box)
              patch_save_path = f"{output_folder}/patch_{patch_number}.jpg"
              patch.save(patch_save_path)
              patch_coords.append((x, y))
              patch_number += 1
        return patch_coords

      # output original and scaled patches into separate folders
      output_folder = os.path.join(
        base_output_folder, 'patches')
      scaled_folder = os.path.join(
        base_output_folder, 'scaled_patches')
      if not os.path.exists(output_folder):
          os.makedirs(output_folder)
          os.makedirs(scaled_folder)
      # split the image into patches
      image = Image.fromarray(image)
      patch_coords = split_image_into_patches(image, sprite_size, output_folder)
      # split resized image into patches
      patch_size = sprite_size * self.scale_factor
      scaled_image = image.resize(
         (image.size[0] * self.scale_factor, image.size[1] * self.scale_factor))
      scaled_patch_coords = split_image_into_patches(
         scaled_image, patch_size, scaled_folder)
      # draw patch numbers on scaled image
      draw = ImageDraw.Draw(scaled_image)
      font_size = 10  # Adjust as needed
      for num, (x, y) in enumerate(patch_coords):
          draw.text(
             (x * self.scale_factor, y * self.scale_factor),
             str(num), fill="white", font_size=font_size)
      step = base_output_folder.split('/')[-1]
      scaled_image.save(f"{base_output_folder}/{step}.jpg")
      return patch_coords


if __name__ == '__main__':
    #substrate_name = 'collaborative_cooking__asymmetric'
    # substrate_name = 'prisoners_dilemma_in_the_matrix__repeated'
    substrate_name = 'prisoners_dilemma_in_the_matrix__arena'
    output_folder = './patch_labels'
    scale_factor = 5
    preprocessor = MPSubstratesPreprocessor(substrate_name, output_folder, scale_factor)
    preprocessor.run_step()
    # preprocessor.split_image_into_patches('collaborative_cooking__asymmetric', 'output_folder')
