# VioNet
VioNet is a CNN based Design Rule Violation (DRV) predictor that predicts the detailed routing wire short before routing take place. A hierarchical prediction approach is adopted where a low resolution prediction is followed by a high resolution prediction. Experiments performed on the ISPD19 benchmarks showed that, on average, VioNet can predict 74\% of the wire-short violations. The hierarchical strategy improved the prediction accuracy, true positive rate, and true negative rate by 11%, 7%, and 11% respectively, compared with the non-hierarchical approach.

## Benchmarks
Benchmarks from ISPD 2019 detailed routing contest were used. Details can be found in ISPD 2019 [website](http://www.ispd.cc/contests/19/).

## How to run
Run `prepare_data.py` to crop the hyper-image to tiles ready for training. Run `VioNet_low_res.py` for low resolution prediction model training. Run `VioNet_high_res.py` for high resolution model training. Run `prediction.py` to do high resolution prediction on test set.
