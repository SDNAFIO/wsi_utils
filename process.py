import os
import tarfile


def extract_archives():
    manifest = open('gdc_manifest_kidney.txt')
    entries = [x for x in manifest][1:]

    for idx, entry in enumerate(entries):
        entry_dat = entry.split('\t')
        target_folder = entry_dat[1]
        target_filename = f'{target_folder}.tar.gz'
        target_folder = target_folder.split('.svs')[0]
        print(f'Processing: {target_folder}')

        if not os.path.exists(target_filename):
            print(f'{target_filename} does not exist')
        else:
            if os.path.exists(target_folder):
                print('Target folder already exists')
                continue

            file = tarfile.open(target_filename)
            print(file.getnames())
            file.extractall(f'./{target_folder}')
            print('Extracted files')
            file.close()


if __name__ == '__main__':
    extract_archives()