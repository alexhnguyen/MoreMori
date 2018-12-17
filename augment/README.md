## Segment face(s)

Uses https://github.com/YuvalNirkin/face_segmentation

### Installation
```
sudo apt-get install libboost-all-dev
sudo apt install caffe-cuda
pip install opencv-python
```

Download the model from 
[here](https://github.com/YuvalNirkin/face_segmentation/releases/download/1.0/face_seg_fcn8s.zip).
Unzip the folder and place it in the same directory as `segmnent_face.py`.

### Usage

Segment an image<br/>
`python3 segment_face.py -f image.jpg -o output_folder`

Segment a directory with images<br/>
`python3 segment_face.py -d input_folder -o output_folder`
