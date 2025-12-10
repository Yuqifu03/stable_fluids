# Fluid Simulation

A modular, high-performance fluid simulation framework built with [Taichi Lang](https://github.com/taichi-dev/taichi). This project supports GPU-accelerated physics, real-time interactive GUIs, and offline rendering. 

## 📚 References

Core fluid simulation algorithms in this project are based on the classic method described in:

* **Stable Fluids**, Jos Stam (SIGGRAPH 1999).
  * [Read Paper (PDF)](https://pages.cs.wisc.edu/~chaol/data/cs777/stam-stable_fluids.pdf)
 
## ✨ Features

* **GPU Acceleration**: Utilizes Taichi's backend (CUDA/Vulkan/Metal) for high-performance physics calculations.
* **Dual Run Modes**:
    * **GUI Mode**: Real-time visualization. Includes a Tkinter-based interactive 2D canvas (draw to disturb fluid) and a standard Taichi 3D viewer.
    * **Render Mode**: Exports simulation steps to GIF or MP4 video files without opening a window.
* **Modular Scenarios**: Easily switch between different simulation logic (e.g., `starry_water`) defined in `src.scenarios`.

## 🛠 Installation

### 1. Prerequisites
* Python 3.8 or higher.
* A GPU compatible with Taichi (recommended for performance).

### 2. Install Dependencies
Install the required Python packages:

```bash
pip install requirements.txt
```

### 3. Project Structure
Ensure your directory looks like this so the imports work correctly:

```bash
project_root/
├── main.py             # Entry point
├── result/
└── src/
    ├── __init__.py
    └── fluid_solver.py
    └── scenarios.py
    └── utils.py
```

## 🚀 Usage
Run the simulation using main.py with the following command-line arguments.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--scenario` | `str` | `starry_water` | The key of the scenario to load (must be defined in `SCENARIO_MAP`). |
| `--mode` | `str` | `gui` | `gui`: Real-time window.<br>`render`: Export results to file. |
| `--frames` | `int` | `300` | Total number of frames to generate (only used in `render` mode). |

Examples
1. Interactive Mode (Default) Run the default "Starry Water" simulation in an interactive Tkinter window
```bash
python main.py
```
2. 3D Simulation Mode Run a specific scenario (e.g., quicksand or others defined in your map) in the Taichi 3D visualizer:
```bash
python main.py --scenario quick_sand --mode gui
```
3.  Rendering to File Render 500 frames of the simulation and save as a GIF/Video to the output directory:
```bash
python main.py --scenario starry_water --mode render --frames 500
```

## 🎮 Controls
Interactive 2D (Starry Water)
Mouse Drag: Draw on the canvas to inject force and disturb the fluid.

## 📝 Extending
To add new simulations, edit `src/scenarios.py`:

1. Create a new class for your simulation.
2. Implement the required methods (e.g., `step()`, `setup_gui()`, `render()`).
3. Register the class in the `SCENARIO_MAP` dictionary.
