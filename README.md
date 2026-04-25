# SwarmGPT

![swarm_gpt_banner](/docs/img/swarm_gpt_banner.png)
[![Format Check](https://github.com/utiasDSL/swarmGPT/actions/workflows/ruff.yaml/badge.svg)](https://github.com/utiasDSL/swarmGPT/actions/workflows/ruff.yaml)
[![website](https://github.com/utiasDSL/swarmGPT/actions/workflows/website.yaml/badge.svg)](https://github.com/utiasDSL/swarmGPT/actions/workflows/website.yaml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

SwarmGPT integrates large language models (LLMs) with safe swarm motion planning, providing an automated and novel approach to deployable drone swarm choreography. Users can automatically generate synchronized drone performances through natural language instructions. Emphasizing safety and creativity, the system combines the creative power of generative models with the effectiveness and safety of model-based planning algorithms. For more information, visit the [project website](https://utiasdsl.github.io/swarm_GPT/) or read our [paper](https://ieeexplore.ieee.org/document/11197931/).

- [Installation](#installation)
- [How to run SwarmGPT](#how-to-run-swarmgpt)
- [Deployment](#deployment)
- [Citing](#citing)

## Installation

[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![ROS Noetic](https://img.shields.io/badge/ROS-Noetic-blue.svg)](http://wiki.ros.org/noetic)

SwarmGPT uses [Pixi](https://pixi.sh) for dependency management and environment setup. Pixi provides a fast, reliable package manager that handles both conda and PyPI dependencies seamlessly.

### Prerequisites

- Linux x64 system (required for ROS Noetic support)
- [Pixi package manager](https://pixi.sh) - see [installation instructions](https://pixi.sh/latest/installation/)

### Setting up SwarmGPT

Clone the repository and activate the environment:
```bash
git clone git@github.com:utiasDSL/swarmGPT.git
cd swarmGPT
pixi shell
```

Note: You will see an error message, that `setup.sh` and `openai_api_key.sh` were not found. This is fine and fixed in the next steps.

This project is build on crazyswarm. Since the original (legacy) version is broken, please use our fork:
```bash
git clone --recurse-submodules git@github.com:utiasDSL/crazyswarm.git submodules/crazyswarm
cd submodules/crazyswarm && ./build.sh
exit # to force sourcing of setup.sh
```

To test the crazyswarm installation, you can try to run the unit tests after reactivating the shell with `pixi shell`:
```bash
cd submodules/crazyswarm/ros_ws/src/crazyswarm/scripts
python -m pytest
```
Note: Those tests will fail after completion of the installation, since they require `numpy<2`.

Crazyswarm needs tracking information (for **deployment only**). At the Learning Systems Lab, we use a Vicon motion capture and therefore need this `vicon_bridge` package:
```bash
git clone git@github.com:ethz-asl/vicon_bridge.git submodules/catkin_ws/src/vicon_bridge
cd submodules/catkin_ws && catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5
exit # to force sourcing of setup.bash
```
You need to set the IP in `submodules/catkin_ws/src/vicon_bridge/launch/vicon.launch` (`datastream_hostport`) and in `submodules/crazyswarm/ros_ws/src/crazyswarm/launch/hover_swarm.launch` (`motion_capture_host_name`).

Lastly, we rely on the VLC media player to play the music. In case you don't have it installed, run:
```bash
sudo apt install vlc
```

Next we can install axswarm and swarmGPT, given an active environment, with:
```bash
git clone git@github.com:utiasDSL/axswarm.git submodules/axswarm
pip install -e .
pip install -e ./submodules/axswarm
```

Note: We are installing axswarm last to force `numpy>=2.0`, which is needed for some of our packages.

Your setup is ready now. If you are unsure if the installation was successful, you can run tests after exporting the API key (see below).
```bash
python -m pytest tests
```

The environment includes:
- **Python 3.11** with essential scientific computing packages
- **ROS Noetic Desktop** for robot communication and control
- **Build tools** (cmake, ninja, catkin_tools) for ROS workspace compilation
- **Development tools** (ruff for linting, uv for fast Python package management)
- **Point Cloud Library (PCL)** for 3D processing

### Documentation Environment

To work with documentation, use the docs environment:

```bash
# Serve documentation locally
pixi run -e docs docs-serve

# Build documentation
pixi run -e docs docs-build
```
## How to run SwarmGPT

### Prerequisites

Before running SwarmGPT, start your pixi shell with `pixi shell`. Then, ensure you have:

1. **OpenAI API Key**: Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   For convinience, you can create a `key.sh` with the command above, which is automatically executed whenever you start you `pixi shell`.

2. **Crazyswarm Configuration**: Configure your drone swarm by editing the `crazyflies.yaml` file in your Crazyswarm installation. SwarmGPT automatically locates this file at:
   ```
   submodules/crazyswarm/ros_ws/src/crazyswarm/launch/crazyflies.yaml
   ```
   
   This file defines:
   - Drone IDs and radio addresses
   - Initial positions for each drone in the swarm
   - Flight area boundaries

### Launching the Interface

1. **Activate the Pixi environment** (if not already active):
   ```bash
   pixi shell
   ```

2. **Launch SwarmGPT**:
   ```bash
   python swarm_gpt/launch.py
   ```
   
   Optional parameters:
   ```bash
   # Use different LLM model
   python swarm_gpt/launch.py --model_id="gpt-3.5-turbo"
   
   # Disable motion primitives (use raw waypoints)
   python swarm_gpt/launch.py --use_motion_primitives=False
      ```

3. **Access the web interface**: The terminal will display a local URL (typically `http://127.0.0.1:7860`). Open this link in your web browser.

### Using the Interface

1. **Select a song** from the available music library
2. **Generate choreography** - SwarmGPT will create a first synchronized drone performance automatically
3. **Preview the results** in the simulation viewer
4. **Refine as needed** by providing additional prompts or modifications
5. **Deploy when satisfied** with the generated choreography

The system will automatically:
- Analyze the selected music for beats, rhythm, and musical features
- Generate safe, collision-free trajectories for your drone swarm
- Ensure all movements stay within the configured flight boundaries
- Synchronize drone movements with the musical timeline

### Ready for Deployment

Once you're happy with your generated choreography, you can proceed to deploy it on your physical drone swarm.

## Deployment

To deploy the generated choreography on your physical drone swarm, crazyswarm has to be running **before** starting the SwarmGPT interface!

1. **Start the Crazyswarm server**:
   ```bash
   roslaunch crazyswarm hover_swarm.launch
   ```
2. **Launch SwarmGPT** as described in the [Launching the Interface](#launching-the-interface) section.
3. **Generate and preview choreography** using the web interface.
4. **Deploy to drones**: Once satisfied with the choreography, click the "Let the Crazyflies dance" button in the web interface to execute the performance on your physical drone swarm.


## Citing
If you find this work useful, compare it with other approaches or use some components, please cite
us as follows:

```bibtex
@article{schuck2025swarmgpt,
  title={SwarmGPT: Combining Large Language Models with Safe Motion Planning for Drone Swarm Choreography},
  author={Schuck, Martin and Dahanaggamaarachchi, Dinushka Orrin and Sprenger, Ben and Vyas, Vedant and Zhou, Siqi and Schoellig, Angela P.},
  journal={IEEE Robotics and Automation Letters},
  year={2025},
  publisher={IEEE}
}
```