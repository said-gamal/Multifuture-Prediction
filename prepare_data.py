import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("end_frame", type=int)
parser.add_argument("--obs_length", default=10, type=int)
parser.add_argument("--every", default=10, type=int)
parser.add_argument("video_name")
parser.add_argument("annotations_path")
parser.add_argument("outpath_obs")
parser.add_argument("outpath_multifuture")

if __name__ == "__main__":
  args = parser.parse_args()
  for start_frame in range(0, args.end_frame, args.every):
      end_frame = start_frame + ((args.obs_length - 1) * args.every) + 1
      subprocess.run(f'python get_prepared_data_multifuture.py --start_frame {start_frame} --end_frame {end_frame} {args.video_name} {args.annotations_path} {args.outpath_obs} {args.outpath_multifuture}')