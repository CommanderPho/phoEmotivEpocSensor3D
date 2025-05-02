from copy import deepcopy
import numpy as np
import pyvista as pv
import time
import sys
import random
import queue
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display, clear_output


"""
uv venv ./.venv_viz --python=3.9
.\.venv_viz\Scripts\activate

"""

# Create a queue for EEG data
tasks = queue.Queue()

# Electrode names (adjust based on your headset)
electrode_names = ['AF3', 'AF4', 'AuxCMS', 'AuxDRL', 'CMS', 'DRL', 'F3', 'F4', 'F7', 'F8', 'FC5', 'FC6', 'O1', 'O2', 'P7', 'P8', 'T7', 'T8']

electrode_names_dict = {
    0:"AF3",
    1:"AF4",
    2:"Arm_R_Body",
    3:"ArmL_Body",
    4:"AuxCMS",
    5:"AuxDRL",
    6:"CMS",
    7:"DRL",
    8:"F3",
    9:"F4",
    10:"F7",
    11:"F8",
    12:"FC5",
    13:"FC6",
    14:"Headset_Back_Body",
    15:"O1",
    16:"O2",
    17:"P7",
    18:"P8",
    19:"T7",
    20:"T8",	
}

electrode_name_to_position_center = {
    'AF3': [-0.02849366665483203, -0.0731334302412427, -0.06309279727001485],
    'AF4': [0.02849366665483203, -0.07313343024976739, -0.06309279727001485],
    'Arm_R_Body': [0.05551410322883508, -0.027015664734477904, -0.03079282724493873],
    'ArmL_Body': [-0.05551410425847653, -0.027015660031019486, -0.030792982403052636],
    'AuxCMS': [-0.06634027689047482, -0.0030793339151980194, 0.012765326982607012],
    'AuxDRL': [0.06634027689047482, -0.0030793339151980194, 0.01276943341869375],
    'CMS': [-0.06563927965347782, -0.006030545767433053, -0.029458098261013078],
    'DRL': [0.06563927965347782, -0.006030545767433053, -0.029458098261013078],
    'F3': [-0.02363707810785255, -0.05494754518077494, -0.0826450100923021],
    'F4': [0.0236370781035902, -0.05494754518077494, -0.0826450100923021],
    'F7': [-0.04149985112566543, -0.0717646495774555, -0.019034005747378804],
    'F8': [0.04149985112566543, -0.0717646495774555, -0.019034005747378804],
    'FC5': [-0.05378159934215332, -0.04925838264772732, -0.050193148788608866],
    'FC6': [0.05378159934215332, -0.04925838264772732, -0.05019314879279459],
    'Headset_Back_Body': [0.0041755319549092755, 0.07179735000299484, -0.013438814943172376],
    'O1': [-0.023960385178350026, 0.08606975499185385, -0.014604999520661032],
    'O2': [0.023960385178350026, 0.08606975499185385, -0.014604999520661032],
    'P7': [-0.05509887202880335, 0.036132210952347504, -0.009576722499429188],
    'P8': [0.05509887202880335, 0.036132210952347504, -0.009576722499429188],
    'T7': [-0.06534174108801341, -0.02824034937506554, -0.013662302267152341],
    'T8': [0.06534174108801341, -0.02824034937506554, -0.013662302267152341],
 }



class EEGVisualizer:
    def __init__(self, model_path=None, update_interval=0.5, is_notebook=True):
        """
        Initialize the EEG visualizer
        
        Parameters:
        -----------
        model_path : str or Path, optional
            Path to the 3D model file (.glb or .obj)
        update_interval : float, optional
            Time interval between updates in seconds (default: 0.5)
        """
        self.update_interval = update_interval
        self.running = False
        
        # Set up PyVista visualization for Jupyter
        if is_notebook:
            pv.set_jupyter_backend('trame')  # Use static backend for compatibility
            self.plotter = pv.Plotter(notebook=True)
        else:
            ## use pyqt to display in a separate window
            # pv.set_jupyter_backend('trame')  # Use static backend for compatibility
            self.plotter = pv.Plotter()

        # Load the 3D model
        if model_path is None:
            model_path = Path('EXTERNAL/meshes/CompleteEmotivEpocEEG.glb').resolve()
        else:
            model_path = Path(model_path).resolve()
            
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        print(f"Loading 3D model from: {model_path}")
        self.headset = pv.read(model_path)
        

        # Create electrode spheres at predefined positions
        # These are approximate positions for the Emotiv EPOC headset
        # You'll need to adjust these based on your actual model
        self.electrode_names = deepcopy(electrode_names)
        self.electrode_names_dict = deepcopy(electrode_names_dict)
        self.electrode_name_to_position_center = deepcopy(electrode_name_to_position_center)
        
        # Create electrode spheres
        self.electrodes = {}
        for name in electrode_names:
            try:
                # Try to get block by name
                self.electrodes[name] = self.headset.get_block_by_name(name)
                print(f"Found electrode: {name}")
            except Exception as e:
                print(f"Warning: Could not find electrode '{name}' in the model: {e}")
                # If we can't find the electrode, we'll create a placeholder
                # This is just so the script doesn't crash if the model doesn't have all electrodes
                sphere = pv.Sphere(radius=0.01, center=(0, 0, 0))
                sphere.name = name
                self.electrodes[name] = sphere
                            
                if name in self.electrode_name_to_position_center:
                    # Create a sphere at the electrode position
                    position = self.electrode_name_to_position_center[name]
                    sphere = pv.Sphere(radius=0.01, center=position)
                    sphere.name = name
                    # Add the sphere to the plotter with a name
                    self.electrodes[name] = sphere
                    print(f"Created electrode sphere: {name} at position {position}")
                else:
                    print(f"Warning: No position defined for electrode '{name}'")
        
        # Add model to the scene
        self.plotter.add_mesh(self.headset, color='lightgray')
        
        # Add electrodes with initial color (yellow)
        for name, electrode in self.electrodes.items():
            self.plotter.add_mesh(electrode, color='yellow', name=name)
        
        # Add a title
        self.plotter.add_text("EEG Electrode Quality Visualization\nRed = Poor Quality, Green = Good Quality", 
                              position="upper_left", font_size=12, color='white')
        
        # Create interactive widgets
        self.create_widgets()
        
    def create_widgets(self):
        """Create interactive widgets for the visualization"""
        # Create sliders for each electrode
        self.sliders = {}
        self.slider_layout = widgets.Layout(width='300px')
        
        slider_widgets = []
        for name in electrode_names:
            slider = widgets.FloatSlider(
                value=0.5,
                min=0,
                max=1,
                step=0.01,
                description=f'{name}:',
                disabled=False,
                continuous_update=True,
                orientation='horizontal',
                readout=True,
                readout_format='.2f',
                layout=self.slider_layout
            )
            self.sliders[name] = slider
            slider_widgets.append(slider)
        
        # Create buttons for control
        self.random_button = widgets.Button(
            description='Randomize Values',
            button_style='info',
            tooltip='Generate random quality values'
        )
        self.random_button.on_click(self.on_random_button_click)
        
        self.start_stop_button = widgets.Button(
            description='Start Animation',
            button_style='success',
            tooltip='Start/Stop automatic updates'
        )
        self.start_stop_button.on_click(self.on_start_stop_button_click)
        
        self.reset_button = widgets.Button(
            description='Reset Values',
            button_style='warning',
            tooltip='Reset all values to 0.5'
        )
        self.reset_button.on_click(self.on_reset_button_click)
        
        # Create update interval slider
        self.interval_slider = widgets.FloatSlider(
            value=self.update_interval,
            min=0.1,
            max=2.0,
            step=0.1,
            description='Update Interval (s):',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
            layout=self.slider_layout
        )
        self.interval_slider.observe(self.on_interval_change, names='value')
        
        # Create output widget for status messages
        self.output = widgets.Output()
        
        # Arrange widgets
        self.button_box = widgets.HBox([self.random_button, self.start_stop_button, self.reset_button])
        self.slider_box = widgets.VBox(slider_widgets)
        self.control_box = widgets.VBox([
            widgets.HTML("<h3>Electrode Quality Controls</h3>"),
            self.button_box,
            self.interval_slider,
            widgets.HTML("<h4>Individual Electrode Quality</h4>"),
            self.slider_box,
            self.output
        ])
        
        # Connect sliders to update function
        for name, slider in self.sliders.items():
            slider.observe(self.on_slider_change, names='value')
        
    def on_slider_change(self, change):
        """Handle slider value changes"""
        # Get all current values
        quality_values = [self.sliders[name].value for name in electrode_names]
        # Update visualization
        self.update_visualization(quality_values)
        
    def on_random_button_click(self, b):
        """Handle random button click"""
        quality_values = self.generate_random_quality()
        # Update sliders
        for name, value in zip(electrode_names, quality_values):
            self.sliders[name].value = value
        
    def on_start_stop_button_click(self, b):
        """Handle start/stop button click"""
        if self.running:
            self.running = False
            self.start_stop_button.description = 'Start Animation'
            self.start_stop_button.button_style = 'success'
            with self.output:
                clear_output()
                print("Animation stopped")
        else:
            self.running = True
            self.start_stop_button.description = 'Stop Animation'
            self.start_stop_button.button_style = 'danger'
            with self.output:
                clear_output()
                print("Animation started")
            # Start animation in a separate thread
            import threading
            self.animation_thread = threading.Thread(target=self.animate)
            self.animation_thread.daemon = True
            self.animation_thread.start()
            
    def on_reset_button_click(self, b):
        """Handle reset button click"""
        # Reset all sliders to 0.5
        for name in electrode_names:
            self.sliders[name].value = 0.5
        with self.output:
            clear_output()
            print("Values reset to 0.5")
            
    def on_interval_change(self, change):
        """Handle interval slider change"""
        self.update_interval = change['new']
        
    def animate(self):
        """Run animation loop"""
        while self.running:
            quality_values = self.generate_random_quality()
            # Update sliders (which will trigger visualization update)
            for name, value in zip(electrode_names, quality_values):
                self.sliders[name].value = value
            time.sleep(self.update_interval)
    
    def update_visualization(self, quality_values):
        """
        Update the visualization with new quality values
        This creates a new plot each time instead of updating in-place
        """
        # Create a new plotter
        plotter = pv.Plotter(notebook=True)
        
        # Add the headset
        plotter.add_mesh(self.headset, color='lightgray')
        
        # Add electrodes with updated colors
        for i, name in enumerate(electrode_names):
            if i < len(quality_values) and name in self.electrodes:
                quality = quality_values[i]
                color = [1-quality, quality, 0]  # R,G,B
                plotter.add_mesh(self.electrodes[name], color=color)
        
        # Add title
        plotter.add_text("EEG Electrode Quality Visualization\nRed = Poor Quality, Green = Good Quality", 
                         position="upper_left", font_size=12, color='white')
        
        # Display the updated plot
        with self.output:
            clear_output(wait=True)
            display(plotter.show(jupyter_backend='static'))
            quality_str = ", ".join([f"{name}: {quality:.2f}" for name, quality in zip(electrode_names, quality_values)])
            print(f"Quality values: {quality_str}")
    
    def generate_random_quality(self):
        """Generate random quality values for testing"""
        return [random.uniform(0, 1) for _ in range(len(electrode_names))]
    
    def show(self):
        """Display the visualization and widgets in the notebook"""
        # Initial visualization
        # plot_widget = self.plotter.show(jupyter_backend='static')
        plot_widget = self.plotter.show(jupyter_backend='trame', return_viewer=True)

        # Create dashboard
        dashboard = widgets.VBox([
            plot_widget,
            self.control_box,
            self.output
        ])
        display(dashboard)
        
        # Show initial quality values
        with self.output:
            print("Ready. Adjust sliders or click buttons to update visualization.")


# Run the visualizer
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='EEG Electrode Quality Visualizer')
    parser.add_argument('--model', type=str, help='Path to the 3D model file (.glb or .obj)')
    parser.add_argument('--interval', type=float, default=0.5, help='Update interval in seconds (default: 0.5)')
    
    args = parser.parse_args()
    
    try:
        visualizer = EEGVisualizer(model_path=args.model, update_interval=args.interval, is_notebook=False)
        # visualizer.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
        
