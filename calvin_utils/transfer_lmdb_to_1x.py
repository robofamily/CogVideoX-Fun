#!/usr/bin/env python3
import os
from pathlib import Path
import argparse
import av
import json
from tqdm import tqdm
import numpy as np
import torch
from torchvision.transforms.functional import resize
from einops import rearrange
import lmdb
from pickle import loads
from torchvision.io import write_jpeg, decode_jpeg

ORIGINAL_STATIC_RES = 200

def parse_args():
    parser = argparse.ArgumentParser(description="Transfer lmdb dataset format to 1x memmap dataset format")
    parser.add_argument(
        "--in_dir",
        type=str,
        help="Path of directory of the lmdb dataset",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Path of directory of the 1x dataset",
    )
    parser.add_argument(
        "--max_length", default=4000000, type=int, help="Maximum number of frames in the dataset"
    )
    parser.add_argument(
        "--start_ratio", default=0, type=float,
    )
    parser.add_argument(
        "--end_ratio", default=0, type=float,
    )
    args = parser.parse_args()
    return args

def save_video(file, video):
    container = av.open(file, 'w', 'mp4')
    stream = container.add_stream('libx264', rate=5)
    stream.pix_fmt = 'yuv420p'
    for i in range(len(video)):
        frame = av.VideoFrame.from_ndarray(video[i], format='rgb24')
        frame = frame.reformat(format=stream.pix_fmt)
        for packet in stream.encode(frame):
            container.mux(packet)
    # Flush stream
    for packet in stream.encode():
        container.mux(packet)
    container.close()

def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    meta_data = []
    env = lmdb.open(args.in_dir, readonly=True, create=False, lock=False)
    with env.begin() as txn:
        dataset_len = loads(txn.get('cur_step'.encode())) + 1
        start_frame_id = int(dataset_len*args.start_ratio)
        start_ep_idx = loads(txn.get(f'cur_episode_{start_frame_id}'.encode()))
        end_frame_id = int(dataset_len*args.end_ratio)
        last_ep_idx = 0
        frames_in_an_episode = []
        for frame_idx in tqdm(range(start_frame_id, end_frame_id, 1), desc="convert frame"):
            ep_idx = loads(txn.get(f'cur_episode_{frame_idx}'.encode()))
            if ep_idx > last_ep_idx:
                if frames_in_an_episode != []:
                    lang = loads(txn.get(f'inst_{last_ep_idx}'.encode()))
                    save_video(out_dir/f"{last_ep_idx}.mp4", frames_in_an_episode)
                    meta_data.append(
                        {
                            "file_path": str(out_dir/f"{last_ep_idx}.mp4"),
                            "text": lang,
                            "type": "video",
                        },
                    )
                    frames_in_an_episode = []
                last_ep_idx = ep_idx
            rgb_static = decode_jpeg(loads(txn.get(f'rgb_static_{frame_idx}'.encode()))).permute(1,2,0).numpy()
            frames_in_an_episode.append(rgb_static)
    env.close()
    
    meta_json_path = open(out_dir/"metadata.json", "w")
    json.dump(meta_data, meta_json_path, indent=4)

if __name__ == "__main__":
    main()