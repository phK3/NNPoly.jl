import sys
import os
import glob

def main(vnnlib_file, outfile):
    print(f"rewriting {vnnlib_file} to {outfile}")
    lines = []
    # since we only care about bounds after one symbolic pass,
    # we can put any property here, we'll just ignore it later on
    new_prop = '(assert (<= Y_0 0.0))'

    last_idx = -1
    with open(vnnlib_file, 'r') as f:
        for i, line in enumerate(f):
            lines.append(line)
            if 'assert' in line:
                last_idx = i

    with open(outfile, 'a') as f:
        for line in lines[:last_idx]:
            f.write(line)

        f.write('; useless property that can be ignored later on\n')
        f.write(new_prop)


if __name__ == '__main__':
    assert len(sys.argv) >= 3, "expected 2 args: vnnlib_file and outfile"

    if len(sys.argv) == 3:
        # interpret args as vnnlib_file, outfile
        vnnlib_file = sys.argv[1]
        outfile = sys.argv[2]
        main(vnnlib_file, outfile)
    else:
        # interpret args as vnnlib_directory, out_dir, out_suffix
        vnnlib_dir = sys.argv[1]
        out_dir = sys.argv[2]
        suffix = sys.argv[3]
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for vnnlib_file in glob.glob(vnnlib_dir + '/*.vnnlib'):
            filename = vnnlib_file.split('/')[-1]
            
            if suffix == "":
            	outfile = out_dir + '/' + filename
            else:
            	outfile = out_dir + '/' + filename.split('.')[0] + '_' + suffix + '.vnnlib'
            
            main(vnnlib_file, outfile)

