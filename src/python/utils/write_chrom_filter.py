
HUNDRED_KB = 100 * 1000

def load_chroms(chromsizes_filepath):
    """Return chrom_name:size dict."""
    chroms = {}
    with open(chromsizes_filepath, 'r', encoding="utf-8") as chrom_file:
        for line in chrom_file:
            chrom_name, size = line.rstrip().split()
            chroms[chrom_name] = int(size)
    return chroms


def load_centromeres(bed_filepath):
    """Return chrom_name:(centromere_begin, centromere_end) dict."""
    centromeres = {}
    with open(bed_filepath, 'r', encoding="utf-8") as centromere_file:
        for line in centromere_file:
            chrom_name, begin, end = line.rstrip().split()
            centromeres[chrom_name] = (int(begin), int(end))
    return centromeres


def compute_second_arm_region(centromere_end, chromsize, width=303*HUNDRED_KB, resolution=HUNDRED_KB):
    """Return a region of given width (begin,end)
    in the middle of the bigger chrom arm.

    The given resolution fixes the position of the beginning to fit with the
    binning.
    """
    middle = (chromsize + centromere_end)//2

    if middle + width/2 > chromsize:
        raise AssertionError("Region width too big to fit in the second arm.")

    begin = middle - width//2
    begin = begin - begin % resolution #adjust to binning
    end = begin + width
    # print("begin:{}\nmiddle:{}\nend:{}\n".format(begin, middle, end))
    return (begin, end)


def print_chr14_filter():
    """Print in bed format 303 regions of 100kb of chr14."""
    million = 1000000
    step = 100*1000 # 1kb
    start = 50*million
    stop = start + 303*step #303 regions of 100kb
    for val in range(start, stop, step):
        print(f"chr14\t{val}\t{val+step}")


def main():
    """TODO : Write docstring"""

    chrom_filepath = "./chromsizes/hg19.noy.chrom.sizes"
    centromere_filepath = "./filter/hg19.centromeres.bed"

    chroms = load_chroms(chrom_filepath)
    centro = load_centromeres(centromere_filepath)

    for name, size in chroms.items():

        begin, end = compute_second_arm_region(centromere_end=centro[name][1],
                                               chromsize=size)

        with open(f"./temp/hg19.{name}_1.bed", "w", encoding="utf-8") as filter_file:
            filter_file.write(f"{name}\t{begin}\t{end}\n")


if __name__ == "__main__":
    main()
