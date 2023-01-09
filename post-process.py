import os
import subprocess
from argparse import ArgumentParser

argparser = ArgumentParser(
    description="Post-process simulation results, convert png files to video"
)
argparser.add_argument(
    "-P",
    "--path",
    default="ffmpeg",
    type=str,
    help="ffmpeg to use",
)
argparser.add_argument(
    "-R",
    "--framerate",
    default=30,
    type=int,
    help="Frames rate (frames per second)",
)
argparser.add_argument(
    "-L",
    "--loop",
    action="store_true",
    help="Loop over the input",
)
argparser.add_argument(
    "-F",
    "--force",
    action="store_true",
    help="Overwrites existing output files",
)
argparser.add_argument(
    "-D",
    "--directory",
    default="fig",
    type=str,
    help="Output directory",
)
argparser.add_argument(
    "-O",
    "--output",
    default="Population.mp4",
    type=str,
    help="Output video file name",
)

args = argparser.parse_args()
print(args)

commands = [
    args.path,
    "-f",
    "image2",
    "-framerate",
    str(args.framerate),
    "-i",
    os.path.join(args.directory, "frames", "GEN-%d.png"),
    "-loop",
    str(int(args.loop)),
    os.path.join(args.directory, args.output),
    "-y" if args.force else "",
]
if subprocess.run(commands).returncode == 0:
    print("Convertion done")
else:
    print("Convertion failed")
