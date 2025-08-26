
Input file *encode_full_metadata_2025-02_no_revoked.csv.gz* provided by JR
The file is comma separated both multiple fields contains comma in the description, required a lot of cleaning to extract correct columns for metadata extraction.

```bash
### clean
zcat encode_full_metadata_2025-02_no_revoked.csv.gz | sed 's/46,XY,-22,/46_XY_-22_/g' | sed 's$1d996a4e8727/, $1d996a4e8727/,$g' | sed 's/,SID/_SID/g' |sed 's/(rep3,4)/(rep3_4)/g' | sed 's/3151999,http:/3151999_http:/g' | sed 's/A,-11/A_-11/g' | sed 's/IVS22AS,G/IVS22AS_G/g' |sed 's/showing 69,XYY/showing 69_XYY/g' | sed 's/4,9,11-trien-3/4_9_11-trien-3/g' | sed 's/REMC 2,3,4,5/REMC 2_3_4_5/g' | sed 's/46,XY, t(3/46_XY_ t(3/g' | sed 's/,_/_/g' | sed 's/, /_/g' | sed 's/ /_/g' | sed 's/,/\t/g' | sed 's$/_$/\t$g' | sed 's$upper_arm_/\tlower_arm$upper_arm_lower_arm$' | sed 's$STL010\tSTL011$STL010_STL011$' | sed 's$"$$g'> encode_full_metadata_2025-02_no_revoked.tsv

### select column for cancer metadata and clean fields
Rscript curate_metadata.R

### extract metadata
Rscript extract_metadata.freeze1.R  freeze1
```
