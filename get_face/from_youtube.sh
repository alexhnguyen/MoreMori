#! /bin/bash
set -e

count=0
for url in "$@" 
do
    IFS='/' read -ra TOKEN <<< $url
    IFS='=' read -ra LAST <<< ${TOKEN[-1]}
    prefix=${LAST[-1]}
    subdir="$count"
    mkdir output/$subdir
    youtube-dl --abort-on-error -w -f mp4 -o from_youtube_tmp.mp4 $url 
    python3 get_face.py --skipframe 23 --size 256 --factor 1.8 --detector deeplearning \
            --output output/$subdir/ from_youtube_tmp.mp4 --prefix $prefix
    rm from_youtube_tmp.mp4
    count=$((count+1))
done
