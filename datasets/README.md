# Datasets

The raw datasets are not distributed with this repository.  Place the files of
each dataset in its own folder, as described below, and set `data.root` in the
configuration file if you keep them elsewhere.

```
datasets/
├── shuguang/
│   ├── shuguang_1.bmp     # SAR image of T1, (593, 921, 3)
│   ├── shuguang_2.bmp     # RGB image of T2, (593, 921, 3)
│   └── shuguang_gt.bmp    # change map,       (593, 921)
└── islandtown/
    ├── Img11-Ac.ppm       # RGB image of T1, (403, 415, 3)
    ├── Img11-B.ppm        # SAR image of T2, (403, 415, 3)
    └── Img11-C.ppm        # change map,      (403, 415)
```

The three "channel" bands of the SAR files are identical, i.e. the image is
stored as a greyscale picture.  The files are read with OpenCV and divided by
255, so that every image lies in `[0, 1]`.

## Change map

Both datasets are fully labelled: `0` = unchanged, `1` = changed, and every
pixel is used for training and for the reported metrics (`gt_encoding: binary`).

| dataset    | size     | changed | unchanged |
|------------|----------|---------|-----------|
| Shuguang   | 593 x 921 | 25 099  | 521 054   |
| Island Town| 403 x 415 | 6 846   | 160 399   |

## Sources

* **Shuguang** — RadarSat-2 C-band SAR and Google Earth optical imagery with an
  8 m spatial resolution, acquired over Shuguang Village, China, in June 2008 and
  September 2012; it documents construction evolution for urban expansion and
  land-use change studies.
* **Island Town** — RadarSat-2 C-band SAR and Google Earth optical imagery,
  acquired over a coastal town in China between June 2008 and June 2013; it is
  used for urbanisation monitoring and multi-source remote sensing synergy
  analysis.

Please cite the original providers of the datasets as well as this paper.
