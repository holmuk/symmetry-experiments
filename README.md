# symmetry-experiments

Predict the symmetry axis of an image:
```bash
python evaluate.py --path data\NYU\S\I104.png
```

Evaluate RANSAC-based method on the NYU dataset:
```bash
python evaluate.py --path data\NYU --database NYU --method ransac --print-json
```

Evaluate SIFT-based method on the NYU dataset:
```bash
python evaluate.py --path data\NYU --database NYU --method sift --print-json
```

Browse NYU data:
```bash
python browse.py --path data\NYU --database NYU
```

You can download NYU data from https://symmetry.cs.nyu.edu/