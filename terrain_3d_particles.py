import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import plotly.graph_objects as go
import plotly.express as px
from rasterio.transform import xy
import imageio
import os
from pathlib import Path

def render_surface(elevation, transform, reveal_map=None, elev=45, azim=45, 
                  backend='matplotlib', save_path=None):
    """
    Render 3D terrain surface with optional reveal mapping.
    
    Args:
        elevation: 2D numpy array of elevation data
        transform: rasterio transform for coordinate mapping
        reveal_map: 2D array (0-1) showing revealed areas, None for full reveal
        elev: elevation angle for 3D view
        azim: azimuth angle for 3D view
        backend: 'matplotlib' or 'plotly'
        save_path: path to save the rendered image
    
    Returns:
        figure object
    """
    height, width = elevation.shape
    
    # Create coordinate grids
    x_coords = np.zeros((height, width))
    y_coords = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            x, y = xy(transform, i, j)
            x_coords[i, j] = x
            y_coords[i, j] = y
    
    if backend == 'matplotlib':
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create base colormap
        if reveal_map is None:
            colors = plt.cm.terrain(plt.Normalize()(elevation))
        else:
            # Blend between hidden (gray) and revealed (terrain) colors
            hidden_color = np.array([0.3, 0.3, 0.3, 1.0])  # Dark gray
            terrain_colors = plt.cm.terrain(plt.Normalize()(elevation))
            colors = np.zeros_like(terrain_colors)
            
            for i in range(height):
                for j in range(width):
                    alpha = reveal_map[i, j]
                    colors[i, j] = (1 - alpha) * hidden_color + alpha * terrain_colors[i, j]
        
        # Plot surface
        surf = ax.plot_surface(x_coords, y_coords, elevation, 
                              facecolors=colors, alpha=0.9, 
                              linewidth=0, antialiased=True)
        
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        ax.set_zlabel('Elevation (m)')
        ax.set_title('3D Terrain with Wind Particle Revelation')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    elif backend == 'plotly':
        # Plotly implementation for interactive 3D
        if reveal_map is None:
            colorscale = 'terrain'
            surfacecolor = elevation
        else:
            # Create custom colorscale based on reveal map
            surfacecolor = elevation * reveal_map + (1 - reveal_map) * np.min(elevation)
            colorscale = 'terrain'
        
        fig = go.Figure(data=[
            go.Surface(
                x=x_coords, y=y_coords, z=elevation,
                surfacecolor=surfacecolor,
                colorscale=colorscale,
                opacity=0.9
            )
        ])
        
        fig.update_layout(
            title='3D Terrain with Wind Particle Revelation',
            scene=dict(
                xaxis_title='Easting (m)',
                yaxis_title='Northing (m)',
                zaxis_title='Elevation (m)',
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=0.8)
                )
            ),
            width=1200, height=800
        )
        
        if save_path:
            fig.write_image(save_path)
        
        return fig

class ParticleSystem:
    """
    Particle-based wind system that reveals terrain through contact.
    """
    
    def __init__(self, elevation, transform, particle_count=1000, 
                 wind_vector=(10, 5), reveal_radius=50, reveal_alpha=0.05):
        """
        Initialize particle system.
        
        Args:
            elevation: 2D elevation array
            transform: rasterio transform
            particle_count: number of particles
            wind_vector: (u, v) wind velocity in m/s
            reveal_radius: radius of influence for terrain revelation (meters)
            reveal_alpha: strength of revelation per contact
        """
        self.elevation = elevation
        self.transform = transform
        self.particle_count = particle_count
        self.wind_vector = np.array(wind_vector)
        self.reveal_radius = reveal_radius
        self.reveal_alpha = reveal_alpha
        
        # Get terrain bounds
        height, width = elevation.shape
        min_x, min_y = xy(transform, height-1, 0)
        max_x, max_y = xy(transform, 0, width-1)
        
        self.bounds = {
            'min_x': min_x, 'max_x': max_x,
            'min_y': min_y, 'max_y': max_y
        }
        
        # Initialize particles at upwind edge
        self.reset_particles()
        
        # Initialize reveal map
        self.reveal_map = np.zeros_like(elevation)
        
    def reset_particles(self):
        """Reset particles to starting positions at terrain edge."""
        # Place particles at upwind edge based on wind direction
        if self.wind_vector[0] > 0:  # Wind blowing east
            start_x = self.bounds['min_x']
        else:  # Wind blowing west
            start_x = self.bounds['max_x']
            
        if self.wind_vector[1] > 0:  # Wind blowing north
            start_y = self.bounds['min_y']
        else:  # Wind blowing south
            start_y = self.bounds['max_y']
        
        # Randomize particle positions along the edge
        self.positions = np.zeros((self.particle_count, 2))
        self.positions[:, 0] = start_x + np.random.uniform(-500, 500, self.particle_count)
        self.positions[:, 1] = np.random.uniform(self.bounds['min_y'], self.bounds['max_y'], self.particle_count)
        
        # Initialize particle heights above terrain
        self.heights = np.random.uniform(50, 200, self.particle_count)  # meters above ground
        
        # Track which particles are active
        self.active = np.ones(self.particle_count, dtype=bool)
    
    def elevation_at_coord(self, x, y):
        """Get elevation at world coordinates using bilinear interpolation."""
        try:
            # Convert world coords to array indices
            height, width = self.elevation.shape
            
            # Calculate fractional indices
            col_f = (x - self.transform.c) / self.transform.a
            row_f = (y - self.transform.f) / self.transform.e
            
            # Check bounds
            if col_f < 0 or col_f >= width-1 or row_f < 0 or row_f >= height-1:
                return 0  # Return sea level for out-of-bounds
            
            # Bilinear interpolation
            col_i = int(col_f)
            row_i = int(row_f)
            col_frac = col_f - col_i
            row_frac = row_f - row_i
            
            # Get four corner values
            tl = self.elevation[row_i, col_i]      # top-left
            tr = self.elevation[row_i, col_i+1]    # top-right
            bl = self.elevation[row_i+1, col_i]    # bottom-left
            br = self.elevation[row_i+1, col_i+1]  # bottom-right
            
            # Interpolate
            top = tl * (1 - col_frac) + tr * col_frac
            bottom = bl * (1 - col_frac) + br * col_frac
            elevation = top * (1 - row_frac) + bottom * row_frac
            
            return elevation
            
        except:
            return 0
    
    def update_reveal_map(self, positions):
        """Update reveal map based on particle positions."""
        height, width = self.elevation.shape
        
        for pos in positions:
            x, y = pos
            
            # Convert to array coordinates
            col = (x - self.transform.c) / self.transform.a
            row = (y - self.transform.f) / self.transform.e
            
            if 0 <= col < width and 0 <= row < height:
                # Calculate reveal radius in pixels
                pixel_size = abs(self.transform.a)  # meters per pixel
                radius_pixels = int(self.reveal_radius / pixel_size)
                
                # Update reveal map in circular area
                center_row, center_col = int(row), int(col)
                for dr in range(-radius_pixels, radius_pixels + 1):
                    for dc in range(-radius_pixels, radius_pixels + 1):
                        r = center_row + dr
                        c = center_col + dc
                        if (0 <= r < height and 0 <= c < width and
                            dr*dr + dc*dc <= radius_pixels*radius_pixels):
                            distance_factor = 1 - np.sqrt(dr*dr + dc*dc) / radius_pixels
                            self.reveal_map[r, c] = np.minimum(
                                1.0, 
                                self.reveal_map[r, c] + self.reveal_alpha * distance_factor
                            )
    
    def step(self, dt=1.0):
        """Advance particle simulation by one time step."""
        # Simple advection with wind
        displacement = self.wind_vector * dt
        self.positions[self.active] += displacement
        
        # Check terrain contact and update heights
        for i, pos in enumerate(self.positions):
            if not self.active[i]:
                continue
                
            x, y = pos
            terrain_height = self.elevation_at_coord(x, y)
            
            # Check if particle is near surface
            if self.heights[i] <= terrain_height + 10:  # 10m threshold
                # Particle contacts terrain - update reveal map
                self.update_reveal_map([pos])
                
                # Bounce particle up or deactivate
                if np.random.random() < 0.3:  # 30% chance to bounce
                    self.heights[i] = terrain_height + np.random.uniform(20, 100)
                else:
                    self.active[i] = False
            else:
                # Gravity effect
                self.heights[i] -= 2 * dt  # 2 m/s downward
        
        # Remove particles that left the domain
        for i, pos in enumerate(self.positions):
            x, y = pos
            if (x < self.bounds['min_x'] - 1000 or x > self.bounds['max_x'] + 1000 or
                y < self.bounds['min_y'] - 1000 or y > self.bounds['max_y'] + 1000):
                self.active[i] = False
        
        # Respawn some particles if too few are active
        active_count = np.sum(self.active)
        if active_count < self.particle_count * 0.3:
            inactive_indices = np.where(~self.active)[0]
            respawn_count = min(len(inactive_indices), self.particle_count // 10)
            respawn_indices = np.random.choice(inactive_indices, respawn_count, replace=False)
            
            # Reset selected particles
            for idx in respawn_indices:
                self.active[idx] = True
                if self.wind_vector[0] > 0:
                    self.positions[idx, 0] = self.bounds['min_x'] + np.random.uniform(-500, 500)
                else:
                    self.positions[idx, 0] = self.bounds['max_x'] + np.random.uniform(-500, 500)
                self.positions[idx, 1] = np.random.uniform(self.bounds['min_y'], self.bounds['max_y'])
                self.heights[idx] = np.random.uniform(50, 200)

def run_simulation(elevation, transform, output_dir='frames', 
                  particle_count=1000, wind_vector=(15, 8), 
                  num_frames=200, dt=2.0, save_every=5):
    """
    Run the complete particle simulation and save frames.
    
    Args:
        elevation: 2D elevation array
        transform: rasterio transform
        output_dir: directory to save frames
        particle_count: number of particles
        wind_vector: (u, v) wind velocity
        num_frames: total simulation frames
        dt: time step in seconds
        save_every: save frame every N steps
    """
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Initialize particle system
    particles = ParticleSystem(
        elevation, transform, 
        particle_count=particle_count,
        wind_vector=wind_vector,
        reveal_radius=100,
        reveal_alpha=0.02
    )
    
    saved_frames = []
    
    print(f"Running simulation for {num_frames} frames...")
    
    for frame in range(num_frames):
        # Step simulation
        particles.step(dt)
        
        # Save frame periodically
        if frame % save_every == 0:
            frame_path = f"{output_dir}/frame_{frame:04d}.png"
            
            fig = render_surface(
                elevation, transform, 
                reveal_map=particles.reveal_map,
                elev=45, azim=45 + frame * 0.5,  # Slowly rotate view
                backend='matplotlib',
                save_path=frame_path
            )
            plt.close(fig)
            
            saved_frames.append(frame_path)
            print(f"Saved frame {frame}/{num_frames}")
    
    # Create MP4 video
    if saved_frames:
        create_video(saved_frames, f"{output_dir}/terrain_reveal.mp4")
    
    return particles.reveal_map

def create_video(frame_paths, output_path, fps=10):
    """Create MP4 video from frame images."""
    try:
        with imageio.get_writer(output_path, fps=fps) as writer:
            for frame_path in frame_paths:
                image = imageio.imread(frame_path)
                writer.append_data(image)
        print(f"Video saved: {output_path}")
    except Exception as e:
        print(f"Error creating video: {e}")

# Example usage
if __name__ == "__main__":
    # Load processed elevation data
    elevation = np.load("Data/processed_dem.npy")
    
    # Create a simple transform for the processed data
    # (This should match your actual transform from preprocessing)
    from rasterio import Affine
    
    # Example transform - adjust based on your actual data
    pixel_size = 60  # 60m pixels after downsampling (5m * 12)
    origin_x = 825875.0  # Approximate origin after cropping
    origin_y = 835125.0
    
    transform = Affine(pixel_size, 0, origin_x,
                      0, -pixel_size, origin_y)
    
    # Run simulation
    final_reveal_map = run_simulation(
        elevation, transform,
        output_dir='frames',
        particle_count=800,
        wind_vector=(20, 10),  # Strong eastward wind with northward component
        num_frames=150,
        dt=3.0,
        save_every=3
    )
    
    # Save final revealed terrain
    fig = render_surface(elevation, transform, final_reveal_map, 
                        backend='matplotlib', save_path='final_revealed_terrain.png')
    plt.show()