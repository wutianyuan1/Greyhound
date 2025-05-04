import re
import sys
import logging
import json


def generate_config_from_source(src_prefix: str, output_prefix: str):
    config = {}
    with open(f"{src_prefix}/config.hpp") as header_file:
        for line in header_file.readlines():
            if line.startswith("#define"):
                line.rstrip()
                m = re.search(r'#define\s+([A-Za-z]\w+)\s+(.*)', line)
                if m:
                    content = m.group(2)
                    config[m.group(1)] = int(content, base=16 if '0x' in content else 10)
    with open(f"{src_prefix}/shm_storage.hpp") as storage_file:
        all_lines = storage_file.read().replace("\n", "")
        all_lines = re.sub(r" +", r" ", all_lines)
        m = re.search(r'struct Record\{\s*uint64\_t\s([A-Za-z0-9_]+,\s*)*', all_lines)
        numfields = m.group(0).count(',') + 1
        config['NUM_FIELDS'] = numfields
    logging.critical("Config: " + str(config))
    with open(f"{output_prefix}/config.json", 'w') as config_file:
        json.dump(config, config_file)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        src_prefix = "./"
        out_prefix = "./control_plane/"
    else:
        src_prefix = sys.argv[1]
        out_prefix = sys.argv[2]
    generate_config_from_source(src_prefix, out_prefix)
