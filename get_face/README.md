## Get face

This folder has two face detection models. One uses a neural network, and another uses OpenCV's haar
cascades. 

If using the neural network, you need to run<br/>
`pip install face_recognition`

### Usage
To extract face images from a video (or videos), use like so <br/>
`python3 get_face.py --output output_folder/ video1.mp4 video2.mp4`<br/>

### Installation

If using `from_youtube.sh` you need to install youtube-dl

For Ubuntu
```
sudo add-apt-repository ppa:nilarimogard/webupd8
sudo apt-get update
sudo apt-get install youtube-dl
```

For Mac
```
brew install youtube-dl
```
