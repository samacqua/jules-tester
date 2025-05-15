# PyTorch Model Visualizer & Hook Manager (Tauri Edition)

## Overview

This application provides an interactive desktop interface for visualizing PyTorch model architectures and dynamically attaching hooks to model layers. It leverages the [Tauri framework](https://tauri.app/) for a cross-platform user interface built with web technologies, and a Python backend for PyTorch model analysis using the `torchlens` library.

Users can load their PyTorch models, view the computational graph, inspect properties of individual nodes (layers), and attach custom PyTorch hooks to modify or inspect layer behavior during runtime.

## Features

*   **Dynamic Model Loading:** Load PyTorch models defined in Python files.
*   **Graph Visualization:** View the model's computational graph interactively.
*   **Node Inspection:** Click on graph nodes (layers) to see detailed properties.
*   **Dynamic Hook Attachment:** Attach custom PyTorch forward or backward hooks to any layer directly from the UI.
*   **Cross-Platform:** Built with Tauri for compatibility with macOS, Windows, and Linux.

## Architecture

The application consists of two main parts:

1.  **Tauri Frontend:**
    *   Built using web technologies (HTML, CSS, JavaScript).
    *   Provides the user interface for model loading, graph display, node inspection, and hook definition.
    *   Communicates with the Python backend via Tauri's IPC / sidecar mechanism.
2.  **Python Backend (Sidecar):**
    *   A Python script (`tauri_backend.py`) that acts as a sidecar process managed by Tauri.
    *   Handles core PyTorch operations:
        *   Dynamically loading user-defined `nn.Module` classes from Python files.
        *   Analyzing models using the `torchlens` library to extract graph structure.
        *   Parsing the `torchlens` output into a structured graph format.
        *   Compiling and attaching user-defined PyTorch hooks to specified layers.
    *   Communicates with the frontend by exchanging JSON messages over stdin/stdout.

## Prerequisites

To develop or build this application, you will need:

*   **Node.js and npm:** For managing frontend dependencies and running Tauri commands. (Download from [nodejs.org](https://nodejs.org/))
*   **Rust:** For the Tauri backend and build process. (Install via [rustup.rs](https://rustup.rs/))
*   **Python:** Version 3.8 or higher for the Python sidecar.
*   **Tauri CLI:** Install globally after Node.js:
    ```bash
    npm install -g @tauri-apps/cli
    ```
*   **Operating System Specific Build Tools:**
    *   **macOS:** Xcode Command Line Tools.
    *   **Windows:** Microsoft C++ Build Tools (available via Visual Studio Installer).
    *   **Linux:** `build-essential`, `libwebkit2gtk-4.0-dev`, `librsvg2-dev` (and other dependencies listed in Tauri documentation).

## Quick Start / Installation (End-Users)

Once the application is built and packaged (see "Building for Production"), platform-specific installers will be available:

*   **macOS:** A `.dmg` file. Open it and drag the application icon to your Applications folder.
*   **Windows:** An `.msi` installer. Run it and follow the installation prompts.
*   **Linux:** An `.AppImage` or `.deb` file.
    *   `.AppImage`: Make it executable (`chmod +x *.AppImage`) and run it.
    *   `.deb`: Install using your system's package manager (e.g., `sudo dpkg -i *.deb` followed by `sudo apt-get install -f`).

(Typically, these packaged files would be available from a "Releases" page on a GitHub repository or a project website.)

## Development Setup

Follow these steps to set up a local development environment:

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Python Backend Setup:**
    *   Navigate to the project root directory.
    *   Create a Python virtual environment:
        ```bash
        python3 -m venv .venv
        ```
    *   Activate the virtual environment:
        *   macOS/Linux: `source .venv/bin/activate`
        *   Windows: `.\.venv\Scriptsctivate`
    *   Install Python dependencies:
        ```bash
        pip install -r requirements.txt
        ```
    *(Ensure `requirements.txt` includes `torch`, `torchvision`, `torchlens`, `networkx`)*

3.  **Frontend Setup & Running in Development Mode:**
    *   Ensure you are in the project root directory (where `src-tauri` and `package.json` are located).
    *   Install frontend dependencies (if any, e.g., JavaScript libraries for graph visualization):
        ```bash
        npm install
        ```
    *   Run the Tauri development server:
        ```bash
        npm run tauri dev
        ```
    This command will:
        *   Compile the Rust parts of Tauri (if changed).
        *   Start a development server for the frontend (usually with hot reloading).
        *   Open the application window.
        *   The Python sidecar (`tauri_backend.py`) will be automatically invoked by the frontend as needed.

## Building for Production

To build the application for production, creating distributable packages (including `.dmg` for macOS):

1.  Ensure your development setup is complete (Python dependencies installed, frontend dependencies installed).
2.  Run the build command from the project root:
    ```bash
    npm run tauri build
    ```
3.  After the build completes, you will find the packaged application in:
    *   `src-tauri/target/release/bundle/dmg/` for macOS (`.dmg` file).
    *   `src-tauri/target/release/bundle/msi/` for Windows (`.msi` file).
    *   `src-tauri/target/release/bundle/appimage/` or `src-tauri/target/release/bundle/deb/` for Linux.

## How It Works

The application flow is as follows:

1.  The user interacts with the Tauri UI (built with HTML/JS/CSS).
2.  When an action requiring Python backend logic is triggered (e.g., "Analyze Model", "Attach Hook"):
    *   The JavaScript frontend constructs a JSON request.
    *   It uses Tauri's API (`window.__TAURI__.shell.Command` or a custom `invoke` command) to execute the `tauri_backend.py` script as a sidecar.
    *   The JSON request is passed to the Python script's standard input.
3.  `tauri_backend.py`:
    *   Reads the JSON request from stdin.
    *   Performs the requested action (e.g., loads the PyTorch model, analyzes it with `torchlens`, compiles and attaches a hook).
    *   Sends a JSON response (containing data or success/error status) to its standard output.
4.  The JavaScript frontend receives the JSON response from the sidecar's stdout.
5.  The UI is updated based on the response (e.g., render the graph, confirm hook attachment, display an error).

## Directory Structure (Simplified)

```
.
├── app/                        # Python application modules
│   └── core/
│       └── graph_parser.py     # Logic to parse torchlens output
│   └── sample_model.py         # An example PyTorch model
├── src/                        # Frontend code (HTML, CSS, JavaScript)
│   ├── index.html
│   ├── style.css
│   └── script.js               # Main JavaScript for UI logic & backend communication
├── src-tauri/                  # Tauri-specific files
│   ├── Cargo.toml              # Rust dependencies
│   ├── build.rs                # Tauri build script (Rust)
│   ├── icons/                  # Application icons
│   ├── src/                    # Rust source code for Tauri core
│   │   └── main.rs
│   └── tauri.conf.json         # Tauri application configuration (incl. sidecar)
├── tests/                      # Python unit tests
│   ├── __init__.py
│   ├── test_graph_parser.py
│   └── test_tauri_backend.py
├── main.py                     # (Original REPL/Tkinter app - may be removed or kept separate)
├── tauri_backend.py            # Python sidecar entry point
├── requirements.txt            # Python dependencies
├── package.json                # Frontend (Node.js) dependencies & scripts
└── README.md                   # This file
```

---

*(This README assumes a standard Tauri project setup with a JavaScript frontend and Python sidecar. Specific file names or directory structures for the frontend part might vary based on the chosen JavaScript framework, if any.)*
