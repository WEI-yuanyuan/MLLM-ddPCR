# MLLM-ddPCR

A digital PCR (dPCR) analysis tool powered by deep learning and large language models for automated droplet detection, analysis, and professional report generation.

## Requirements

- Python 3.9+
- CUDA-enabled GPU (recommended)

Key dependencies:
- PyQt5
- OpenCV
- NumPy
- Matplotlib
- Pandas
- Pillow
- Requests
- Markdown
- Segment Anything
- Ultralytics
- OpenAI

## Configuration

Before using, configure the following parameters in `config.py`:

- OpenAI API key and base URL
- Model paths and types
- Other path configurations

## Usage

1. Launch the application:

```bash
python gui.py
```

2. Input URLs for three images in the interface:
   - Fluorescence image
   - Bright field image
   - Merged image

3. Click "Analyze" to start the analysis

## Important Notes

1. Ensure image URLs are accessible
2. Model files will be downloaded on first run:
   c-net:https://1drv.ms/u/c/bdeebb56c930dee1/EeGmqop56_NOrGKBcGnU1u0BUxaYSmbAUWJ23sfe2AE2kw?e=eIEfZ9
   SAM:https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
3. Valid OpenAI API key required
4. GPU usage recommended for better performance

## Image Requirements

The tool requires three types of ddPCR images:
1. Fluorescence image
2. Bright field image
3. Merged image


Example URLs:
- Fluorescence: blob:https://onedrive.live.com/fc336e58-68b5-4d5e-9dd1-e38f395be852
- Bright field: blob:https://onedrive.live.com/b70d1ab8-37d1-4f44-bcba-c4748f2fb1bd
- Merged: blob:https://onedrive.live.com/e109b077-4cbe-4f53-a374-08f0288ac50e