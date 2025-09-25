import numpy as np
import matplotlib.pyplot as plt
from rasterio import Affine
from terrain_3d_particles import render_surface, ParticleSystem, create_video
from pathlib import Path

# Load the processed elevation data
elevation = np.load("Data/processed_dem.npy")
print(f"Loaded elevation data with shape: {elevation.shape}")

# Create transform for the processed data
pixel_size = 60  # 60m pixels after downsampling
origin_x = 825875.0
origin_y = 835125.0

transform = Affine(pixel_size, 0, origin_x,
                  0, -pixel_size, origin_y)

# Initialize particle system with optimized parameters
particles = ParticleSystem(
    elevation, transform,
    particle_count=500,          # Moderate particle count for performance
    wind_vector=(20, 8),         # Strong eastward wind, slight northward
    reveal_radius=120,           # Larger reveal radius for better visibility
    reveal_alpha=0.05            # Moderate reveal strength
)

# Create frames directory
Path("frames").mkdir(exist_ok=True)

# Run short simulation with frame saving
num_frames = 60  # 60 frames for a short demo
save_every = 2   # Save every 2nd frame
saved_frames = []

print(f"Running simulation for {num_frames} frames...")

for frame in range(num_frames):
    # Step simulation
    particles.step(dt=3.0)  # 3 second time steps
    
    # Print progress
    if frame % 10 == 0:
        active_count = np.sum(particles.active)
        revealed_pixels = np.sum(particles.reveal_map > 0)
        revealed_percent = (revealed_pixels / particles.reveal_map.size) * 100
        print(f"Frame {frame}: {active_count} particles, {revealed_percent:.1f}% terrain revealed")
    
    # Save frame periodically
    if frame % save_every == 0:
        frame_path = f"frames/frame_{frame:04d}.png"
        
        # Render with slowly rotating view
        fig = render_surface(
            elevation, transform, 
            reveal_map=particles.reveal_map,
            elev=35,                    # Fixed elevation angle
            azim=45 + frame * 1.0,      # Rotate 1 degree per frame
            backend='matplotlib',
            save_path=frame_path
        )
        plt.close(fig)
        
        saved_frames.append(frame_path)
        if frame % 10 == 0:
            print(f"  Saved frame {len(saved_frames)}")

print(f"\nSimulation complete! Generated {len(saved_frames)} frames.")

# Create MP4 video
if saved_frames:
    print("Creating video...")
    create_video(saved_frames, "frames/terrain_reveal_demo.mp4", fps=8)

# Save final state
print("Saving final revealed terrain...")
final_reveal_percent = (np.sum(particles.reveal_map > 0) / particles.reveal_map.size) * 100
print(f"Final terrain revealed: {final_reveal_percent:.1f}%")

fig_final = render_surface(
    elevation, transform, 
    reveal_map=particles.reveal_map,
    elev=35, azim=45,
    backend='matplotlib',
    save_path='final_terrain_reveal.png'
)
plt.close(fig_final)

print("Demo complete! Check:")
print("  - frames/terrain_reveal_demo.mp4 (animation)")
print("  - final_terrain_reveal.png (final state)")
print("  - Individual frames in frames/ directory")