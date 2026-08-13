# Tremor Peak Annotation Utility

Install the dependency, then run the script with a folder containing `.mp4` videos:

```bash
python -m pip install -r requirements.txt
python tremor_annotator.py /path/to/videos
```

Keep the OpenCV video window focused. Only the centered five-second portion of
each video plays (the entire video plays if it is shorter than five seconds), at
the configured slow speed shown in the overlay. Press Space at each motion peak.
When the video ends, press `R` to redo it, `Enter` or `N` to accept it, or
`Q`/`Esc` to quit. At least two peaks are required to accept a video.

Accepted results are saved as `summary.csv` and as detailed JSON records in
`annotations/` inside the input folder. A later run skips already accepted
videos. Keypress time is measured with wall-clock timing and converted to the
original-video timeline, so frequency values remain in the video's actual
timebase; precision is limited by input and operating-system scheduling. Each
detailed JSON record includes the exact annotation-window start and end times.

Frequency is the reciprocal of the mean interval between peaks. Where there
are at least two intervals (three peaks), the saved uncertainty values include
the sample interval standard deviation, propagated frequency standard error,
and an approximate 95% normal confidence interval. With only two peaks, the
frequency is saved but those uncertainty fields are blank.
