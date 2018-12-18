## Get face

This folder has two face detection models. One uses a neural network, and another uses OpenCV's haar
cascades. It also contains a script to extract faces from videos.

### Usage
To extract face images from a video (or videos), use like so <br/>
`python3 get_face.py --output output_folder/ video1.mp4 video2.mp4`<br/>

To extract face images from a youtube video, use like so <br/>
`bash from_youtube.sh youtubelink1.com youtubelink2.com `<br/>
The images will automatically go to output/_i_, where _i_ is the _ith_ video in the input.

### Installation

If using `from_youtube.sh` you need to install youtube-dl

For Ubuntu
```
sudo add-apt-repository ppa:nilarimogard/webupd8
sudo apt-get update
sudo apt-get install youtube-dl
```
