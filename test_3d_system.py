import numpy as np
import matplotlib.pyplot as plt
from rasterio import Affine
from terrain_3d_particles import render_surface, ParticleSystem

# Load the processed elevation data
elevation = np.load("Data/processed_dem.npy")
print(f"Loaded elevation data with shape: {elevation.shape}")
print(f"Elevation range: {np.min(elevation):.1f} to {np.max(elevation):.1f} meters")

# Create transform matching the processed data
# Based on the previous processing: 60m pixels (5m * 12 downsample factor)
pixel_size = 60  
origin_x = 825875.0  # Approximate, adjust if needed
origin_y = 835125.0

transform = Affine(pixel_size, 0, origin_x,
                  0, -pixel_size, origin_y)

# Test 1: Render basic terrain without particles
print("\nRendering basic terrain...")
fig1 = render_surface(elevation, transform, reveal_map=None, 
                     backend='matplotlib', save_path='test_basic_terrain.png')
plt.close(fig1)
print("Basic terrain saved as 'test_basic_terrain.png'")

# Test 2: Create a simple reveal map and render
print("\nTesting reveal map functionality...")
reveal_map = np.zeros_like(elevation)
# Create a circular revealed area in the center
center_row, center_col = elevation.shape[0] // 2, elevation.shape[1] // 2
radius = 20
for i in range(elevation.shape[0]):
    for j in range(elevation.shape[1]):
        distance = np.sqrt((i - center_row)**2 + (j - center_col)**2)
        if distance <= radius:
            reveal_map[i, j] = 1.0 - (distance / radius) * 0.7  # Fade from center

fig2 = render_surface(elevation, transform, reveal_map=reveal_map,
                     backend='matplotlib', save_path='test_reveal_terrain.png')
plt.close(fig2)
print("Reveal test saved as 'test_reveal_terrain.png'")

# Test 3: Initialize particle system and run a few steps
print("\nTesting particle system...")
particles = ParticleSystem(
    elevation, transform,
    particle_count=100,
    wind_vector=(15, 5),
    reveal_radius=80,
    reveal_alpha=0.1
)

print(f"Initialized {particles.particle_count} particles")
print(f"Active particles: {np.sum(particles.active)}")

# Run a few simulation steps
for step in range(10):
    particles.step(dt=2.0)
    active_count = np.sum(particles.active)
    revealed_pixels = np.sum(particles.reveal_map > 0)
    print(f"Step {step+1}: {active_count} active particles, {revealed_pixels} pixels revealed")

# Render result after particle simulation
fig3 = render_surface(elevation, transform, reveal_map=particles.reveal_map,
                     backend='matplotlib', save_path='test_particle_reveal.png')
plt.close(fig3)
print("Particle reveal test saved as 'test_particle_reveal.png'")

print("\nAll tests completed! Check the generated PNG files.")