# Yolonet (YOLO) — Real‑Time Object Detection & Neural Network Framework

This repository contains  a lightweight neural network framework written in **C** with optional **CUDA/cuDNN** acceleration, plus scripts and configs for running and training **YOLO** object detectors (YOLOv4/YOLOv3 and related variants).

> **What’s in this repo?**
> - C/CUDA sources (`src/`, `include/`) and build systems (**Makefile** + **CMake**)
> - Configs (`cfg/`) and sample data (`data/`)
> - CLI executable `Yolonet` and shared library (`libYolonet.so` on Linux, `Yolonet.dll` on Windows when built)
> - Python ctypes wrapper (`Yolonet.py`) + helper scripts (`Yolonet_images.py`, `Yolonet_video.py`)
> - Dockerfiles for CPU/GPU builds

---

## Quick start (Linux)

This project bundle already includes **`yolov4-tiny.cfg`** and **`yolov4-tiny.weights`**, so you can run a detection demo without downloading anything else.

### 1) Build (or use the included binary)

```bash
# From the repo root
make
```

> Want video/webcam windows via OpenCV?
>
> Edit `Makefile` and set `OPENCV=1` (and optionally `GPU=1`, `CUDNN=1`, `CUDNN_HALF=1`), then rebuild:
>
> ```bash
> make clean && make
> ```

### 2) Run object detection on an image

```bash
./Yolonet detector test cfg/coco.data yolov4-tiny.cfg yolov4-tiny.weights data/dog.jpg -thresh 0.25
```

Yolonet saves an annotated image as **`predictions.jpg`** by default.

---

## Python demo (images / video)

### Install Python deps

```bash
python3 -m pip install --upgrade pip
python3 -m pip install opencv-python numpy
```

### Run the image script

```bash
python3 Yolonet_images.py   --input data/dog.jpg   --weights yolov4-tiny.weights   --config_file yolov4-tiny.cfg   --data_file cfg/coco.data   --thresh 0.25
```

### Run the webcam / video script

```bash
python3 Yolonet_video.py   --input 0   --weights yolov4-tiny.weights   --config_file yolov4-tiny.cfg   --data_file cfg/coco.data   --thresh 0.25
```

> Note: The Python wrapper loads **`libYolonet.so`** (Linux/macOS) or **`Yolonet.dll`** (Windows). Make sure you have built Yolonet and that the library is in the repo root (or on your library path).

---

## Downloading additional pretrained weights (optional)

This repo already includes **YOLOv4‑tiny weights**. If you want other models, download weights and place them in the repo root (next to `./Yolonet`).

### YOLOv4 (full)

```bash
# weights (~258 MB)
wget -O yolov4.weights \
  https://sourceforge.net/projects/Yolonet-yolo.mirror/files/Yolonet_yolo_v4_pre/yolov4.weights/download
```

Then run:

```bash
./Yolonet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights data/dog.jpg -thresh 0.25
```

### YOLOv3 (classic)

```bash
wget -O yolov3.weights https://pjreddie.com/media/files/yolov3.weights
```

Then run:

```bash
./Yolonet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights data/dog.jpg -thresh 0.25
```

---

## Building

### Option A — Makefile (Linux/macOS)

1. Open `Makefile` and enable what you need:

- `GPU=1` — CUDA
- `CUDNN=1` — cuDNN
- `CUDNN_HALF=1` — FP16 / Tensor Cores (supported GPUs)
- `OPENCV=1` — OpenCV (video, webcam, imshow windows)
- `OPENMP=1`, `AVX=1` — faster CPU inference (if supported)

2. Build:

```bash
make clean
make -j
```

### Option B — CMake (Linux/macOS/Windows)

```bash
mkdir -p build_cmake
cd build_cmake
cmake .. -DENABLE_OPENCV=ON -DENABLE_CUDA=OFF
cmake --build . --config Release
```

To enable CUDA:

```bash
cmake .. -DENABLE_CUDA=ON -DENABLE_CUDNN=ON -DENABLE_OPENCV=ON
```

### Windows (Visual Studio)

This repo includes Visual Studio solution files under `build/Yolonet/`.

- Open `build/Yolonet/Yolonet.sln` in Visual Studio
- Select **Release | x64**
- Build the solution
- The resulting `Yolonet.exe` / `Yolonet.dll` will be in the corresponding `x64/` output folder

---

## Training a custom detector (high level)

1. **Prepare your dataset**
   - Images (e.g., `.jpg`) + YOLO label files (`.txt`) for each image
   - A `.names` file with one class name per line
   - `train.txt` / `valid.txt` listing image paths

2. **Create a data file** (example: `data/obj.data`)
   ```ini
   classes = <NUM_CLASSES>
   train   = data/train.txt
   valid   = data/valid.txt
   names   = data/obj.names
   backup  = backup/
   ```

3. **Create/update a model config**
   - Start from `cfg/yolov4-tiny-custom.cfg` or `cfg/yolov4-custom.cfg`
   - Set `classes=<NUM_CLASSES>` in each `[yolo]` layer
   - Update the preceding `[convolutional]` `filters` to `(NUM_CLASSES + 5) * 3`

4. **Train**
   ```bash
   ./Yolonet detector train data/obj.data cfg/yolov4-tiny-custom.cfg yolov4-tiny.conv.29
   ```

---

## Repository layout (partial)

- `cfg/` — network configuration files (YOLOv4, YOLOv3, custom templates)
- `data/` — sample images + class name lists
- `src/`, `include/` — Yolonet source code
- `scripts/` — helper scripts
- `Dockerfile.cpu`, `Dockerfile.gpu`, `docker-compose.yml` — container builds
- `Yolonet.py`, `Yolonet_images.py`, `Yolonet_video.py` — Python wrapper & demos

---

## License

This project uses the “YOLO LICENSE” (public domain / do‑what‑you‑want). See `LICENSE` in this repository for the full text.

---

## Acknowledgements

- Yolonet originally by **Joseph Redmon**
- Many improvements and YOLOv4/Windows/Linux tooling popularized by the community Yolonet/YOLO fork ecosystem
