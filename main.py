import os
import sys
import numpy as np
import trimesh
import vispy.scene
from vispy.scene import visuals
from vispy import app

def load_model():
    """Load the 3D model file."""
    # model_path = os.path.join(os.path.dirname(__file__), 'Resources', 'Emotiv Epoc_SplitJoints_0001.fbx')
    model_path = os.path.join(os.path.dirname(__file__), 'Resources', 'Full_scene_3.2_bundled.obj')
    try:
        mesh = trimesh.load(model_path)
        return mesh
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def render_model(scene):
    """Render the 3D model using vispy."""
    # Create a canvas with a 3D viewport
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    
    # Handle scene with multiple meshes
    if isinstance(scene, trimesh.Scene):
        # Extract all mesh geometry from the scene
        meshes = []
        for name, geometry in scene.geometry.items():
            if isinstance(geometry, trimesh.Trimesh):
                meshes.append(geometry)
        
        # If we have meshes, combine them into one
        if meshes:
            combined_mesh = trimesh.util.concatenate(meshes)
            vertices = combined_mesh.vertices
            faces = combined_mesh.faces
        else:
            print("No valid meshes found in the scene")
            sys.exit(1)
    else:
        # It's already a single mesh
        vertices = scene.vertices
        faces = scene.faces
    
    # Add a 3D mesh to the scene
    mesh_visual = visuals.Mesh(vertices=vertices, faces=faces, shading='smooth')
    view.add(mesh_visual)
    
    # Add camera and lighting
    view.camera = 'turntable'
    view.camera.fov = 45
    view.camera.distance = 2
    
    # Set up lighting - Fixed: Don't set light_dir directly
    # Create a directional light
    light = vispy.scene.DirectionalLight((0, 10, 0), color='white')
    view.add(light)
    
    # Center the camera on the scene
    view.camera.center = (0, 0, 0)
    
    return canvas


def main():
    """Main function to run the application."""
    print("Hello from phoemotivepocsensor3d!")
    print("Loading 3D model of Emotiv EPOC headset...")
    
    mesh = load_model()
    canvas = render_model(mesh)
    
    # Run the application
    app.run()


if __name__ == "__main__":
    main()
