# Neural 3D SLAM + Real-time profiling

## Colab Quickstart
Paste this in the first cell:

```python
REPO = "https://github.com/<your-username>/neural_slam_rt.git"; BRANCH="main"

from google.colab import drive; drive.mount('/content/drive')
%cd /content
!git clone -b {BRANCH} {REPO} || (cd neural_slam_rt && git pull)
%cd /content/neural_slam_rt

# Point data/ to your Drive folder (create it once and add TUM there if you have it)
!rm -rf data && ln -s "/content/drive/MyDrive/tum_rgbd" data

!pip -q install -r requirements.txt
!python scripts/run_baseline.py
